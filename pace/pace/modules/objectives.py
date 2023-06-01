import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
import json
import tqdm
import functools
import random
from collections import defaultdict

from torch.utils.data.distributed import DistributedSampler
from torch.nn.modules.loss import _Loss
from einops import rearrange
from sklearn.metrics import precision_score, recall_score, f1_score

from pace.modules.dist_utils import all_gather
from pace.utils.glossary import slot_tokens, slot_values_keys, open_slots, slot_values
from pace.utils.write_mmconv_dst import make_results
from pace.utils.eval_mmconv_rg import evaluate_mmconvrg

class LabelSmoothingLoss(_Loss):
    def __init__(self, label_smoothing=0, tgt_vocab_size=0, ignore_index=0, size_average=None, reduce=None, reduction='mean'):
        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        super(LabelSmoothingLoss, self).__init__(
            size_average=size_average, reduce=reduce, reduction=reduction)

        assert label_smoothing > 0
        assert tgt_vocab_size > 0

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.ignore_index] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))
        self.confidence = 1.0 - label_smoothing
        self.tgt_vocab_size = tgt_vocab_size

    def forward(self, output, target):
        assert self.tgt_vocab_size == output.size(2)
        batch_size, num_pos = target.size(0), target.size(1)
        output = output.view(-1, self.tgt_vocab_size)
        target = target.view(-1)
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)

        return F.kl_div(output, model_prob.type_as(output), reduction='none').view(batch_size, num_pos, -1).sum(2)

class ClassificationLabelSmoothingLoss(_Loss):
    def __init__(self, label_smoothing=0, tgt_size=0, size_average=None, reduce=None, reduction='mean'):
        assert 0.0 < label_smoothing <= 1.0
        super(ClassificationLabelSmoothingLoss, self).__init__(
            size_average=size_average, reduce=reduce, reduction=reduction)

        assert label_smoothing > 0
        assert tgt_size > 0

        smoothing_value = label_smoothing / (tgt_size-1)
        one_hot = torch.full((tgt_size,), smoothing_value)
        self.register_buffer('one_hot', one_hot.unsqueeze(0))
        self.confidence = 1.0 - label_smoothing
        self.tgt_size = tgt_size

    def forward(self, output, target):
        batch_size = target.size(0)
        output = output.view(-1, self.tgt_size)
        model_prob = self.one_hot.repeat(batch_size, 1).to(output)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        return F.kl_div(output, model_prob.type_as(output), reduction='batchmean')

def cost_matrix_cosine(x, y, eps=1e-5):
    """Compute cosine distnace across every pairs of x, y (batched)
    [B, L_x, D] [B, L_y, D] -> [B, Lx, Ly]"""
    assert x.dim() == y.dim()
    assert x.size(0) == y.size(0)
    assert x.size(2) == y.size(2)
    x_norm = F.normalize(x, p=2, dim=-1, eps=eps)
    y_norm = F.normalize(y, p=2, dim=-1, eps=eps)
    cosine_sim = x_norm.matmul(y_norm.transpose(1, 2))
    cosine_dist = 1 - cosine_sim
    return cosine_dist


def trace(x):
    """ compute trace of input tensor (batched) """
    b, m, n = x.size()
    assert m == n
    mask = torch.eye(n, dtype=torch.bool, device=x.device).unsqueeze(0).expand_as(x)
    trace = x.masked_select(mask).contiguous().view(b, n).sum(dim=-1, keepdim=False)
    return trace


@torch.no_grad()
def ipot(C, x_len, x_pad, y_len, y_pad, joint_pad, beta, iteration, k):
    """ [B, M, N], [B], [B, M], [B], [B, N], [B, M, N]"""
    b, m, n = C.size()
    sigma = torch.ones(b, m, dtype=C.dtype, device=C.device) / x_len.unsqueeze(1)
    T = torch.ones(b, n, m, dtype=C.dtype, device=C.device)
    A = torch.exp(-C.transpose(1, 2) / beta)

    # mask padded positions
    sigma.masked_fill_(x_pad, 0)
    joint_pad = joint_pad.transpose(1, 2)
    T.masked_fill_(joint_pad, 0)
    A.masked_fill_(joint_pad, 0)

    # broadcastable lengths
    x_len = x_len.unsqueeze(1).unsqueeze(2)
    y_len = y_len.unsqueeze(1).unsqueeze(2)

    # mask to zero out padding in delta and sigma
    x_mask = (x_pad.to(C.dtype) * 1e4).unsqueeze(1)
    y_mask = (y_pad.to(C.dtype) * 1e4).unsqueeze(1)

    for _ in range(iteration):
        Q = A * T  # bs * n * m
        sigma = sigma.view(b, m, 1)
        for _ in range(k):
            delta = 1 / (y_len * Q.matmul(sigma).view(b, 1, n) + y_mask)
            sigma = 1 / (x_len * delta.matmul(Q) + x_mask)
        T = delta.view(b, n, 1) * Q * sigma
    T.masked_fill_(joint_pad, 0)
    return T


def optimal_transport_dist(
    txt_emb, img_emb, txt_pad, img_pad, beta=0.5, iteration=50, k=1
):
    """ [B, M, D], [B, N, D], [B, M], [B, N]"""
    cost = cost_matrix_cosine(txt_emb, img_emb)
    # mask the padded inputs
    joint_pad = txt_pad.unsqueeze(-1) | img_pad.unsqueeze(-2)
    cost.masked_fill_(joint_pad, 0)

    txt_len = (txt_pad.size(1) - txt_pad.sum(dim=1, keepdim=False)).to(dtype=cost.dtype)
    img_len = (img_pad.size(1) - img_pad.sum(dim=1, keepdim=False)).to(dtype=cost.dtype)

    T = ipot(
        cost.detach(), txt_len, txt_pad, img_len, img_pad, joint_pad, beta, iteration, k
    )
    distance = trace(cost.matmul(T.detach()))
    return distance


def compute_mlm(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=True, mask_image=False)
    mlm_logits = pl_module.mlm_score(infer["text_feats"])
    mlm_labels = infer["text_labels"]

    mlm_loss = F.cross_entropy(
        mlm_logits.view(-1, pl_module.hparams.config["vocab_size"]),
        mlm_labels.view(-1),
        ignore_index=-100,
    )

    ret = {
        "mlm_loss": mlm_loss,
        "mlm_logits": mlm_logits,
        "mlm_labels": mlm_labels,
        "mlm_ids": infer["text_ids"],
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mlm_loss")(ret["mlm_loss"])
    acc = getattr(pl_module, f"{phase}_mlm_accuracy")(
        ret["mlm_logits"], ret["mlm_labels"]
    )
    pl_module.log(f"mlm/{phase}/loss", loss)
    pl_module.log(f"mlm/{phase}/accuracy", acc)

    return ret

@torch.no_grad()
def generation_test_wrapup(pl_module):
    detokenize = pl_module.hparams.config["detokenize"]
    dms = pl_module.trainer.datamodule.dms
    dataset = dms[0].test_dataset
    tokenizer = dataset.tokenizer
    prompt_text = pl_module.hparams.config["decode_prompt"]
    decode_prompt = tokenizer(prompt_text , add_special_tokens=False).input_ids
    dist_sampler = DistributedSampler(dataset, shuffle=False)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=pl_module.hparams.config["per_gpu_batchsize"],
        num_workers=pl_module.hparams.config["num_workers"],
        pin_memory=True,
        sampler=dist_sampler,
        collate_fn=functools.partial(
            dataset.collate,
            mlm_collator=dms[0].mlm_collator,
        ),
    )

    outputs = []    
    iids = []
    source_texts = []
    for _b in tqdm.tqdm(dataloader, desc="generation loop"):
        iids.extend(_b["raw_index"])
        source_texts.extend(_b["text"])
        output  = pl_module(_b , decode_prompt=decode_prompt)['pred_seq']
        outputs.extend(output)
        
    torch.distributed.barrier()
    gather_iids = all_gather(iids)
    gather_outputs = all_gather(outputs)
    gather_sources = all_gather(source_texts)

    print('rank num:',len(gather_iids))

    output_sequences = {}
    for i in range(len(gather_outputs)):
        for j in range(len(gather_outputs[i])):
            # sent = text_dset.tokenizer.decode(output_ids[i])
            output_tokens = tokenizer.convert_ids_to_tokens(gather_outputs[i][j])
            sent = ' '.join(detokenize(output_tokens))
            splits = sent.split("[SEP]")
            result = ""
            for split in splits:
                split_sent = split.replace("[PAD]","").strip()
                if len(split_sent)>0:
                    result = split_sent
                    break
            output_sequences[gather_iids[i][j]] = {
                'pred':result,
                'source':gather_sources[i][j]
            }
    task = pl_module.hparams.config["datasets"][0] 
    model_path = pl_module.hparams.config["load_path"].split(".")[0]
    output_file_name = task + "_" + model_path
    output_file_name = output_file_name.replace("/","_") + '.json'
    if 0 == torch.distributed.get_rank():
        with open(output_file_name , "w") as output_file:
            json.dump(output_sequences , output_file)
        print(f"output file has been saved to {output_file_name}")

def compute_seq2seq(pl_module , batch):
    def gather_seq_out_by_pos(seq, pos):
        return torch.gather(seq, 1, pos.unsqueeze(2).expand(-1, -1, seq.size(-1)))
        
    infer = pl_module.infer(batch, mask_text=True, mask_image=False)
    masked_pos = batch["target_masked_pos"]
    masked_labels = batch["target_masked_ids"]
    masked_weights = batch["target_masked_weights"]

    text_feats_masked = gather_seq_out_by_pos(infer["text_feats"] , masked_pos)
    masked_logits = pl_module.mlm_score(text_feats_masked)

    masked_loss = F.cross_entropy(
        masked_logits.view(-1, pl_module.hparams.config["vocab_size"]),
        masked_labels.view(-1),
        ignore_index=-100
    )

    ret = {
        "seq2seq_loss": masked_loss,
        "seq2seq_logits": masked_logits,
        "seq2seq_labels": masked_labels,
        "seq2seq_ids": masked_labels,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_seq2seq_loss")(ret["seq2seq_loss"])
    acc = getattr(pl_module, f"{phase}_seq2seq_accuracy")(
        ret["seq2seq_logits"], ret["seq2seq_labels"]
    )
    pl_module.log(f"seq2seq/{phase}/loss", loss)
    pl_module.log(f"seq2seq/{phase}/accuracy", acc)

    return ret

def compute_mpp(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=False, mask_image=True)
    mpp_logits = pl_module.mpp_score(infer["image_feats"])
    mpp_logits = torch.stack(
        [
            mpp_logits[:, :, 0:256],
            mpp_logits[:, :, 256:512],
            mpp_logits[:, :, 512:768],
        ],
        dim=2,
    )
    mpp_labels = infer["image_labels"]

    mpp_loss = F.cross_entropy(
        mpp_logits.view(-1, 256),
        mpp_labels.view(-1),
        ignore_index=-100,
    )

    ret = {
        "mpp_loss": mpp_loss,
        "mpp_logits": mpp_logits,
        "mpp_labels": mpp_labels,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mpp_loss")(ret["mpp_loss"])
    acc = getattr(pl_module, f"{phase}_mpp_accuracy")(
        ret["mpp_logits"], ret["mpp_labels"]
    )
    pl_module.log(f"mpp/{phase}/loss", loss)
    pl_module.log(f"mpp/{phase}/accuracy", acc)

    return ret


def compute_mppd(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=False, mask_image=True)
    mppd_logits = pl_module.mppd_score(infer["image_feats"])
    mppd_labels = infer["image_labels_mppd"]
    filter_to_train = infer["image_labels"].float().mean(dim=-1) != -100

    labels = mppd_labels[filter_to_train]
    logits = mppd_logits[filter_to_train]
    mppd_loss = F.mse_loss(logits, labels)

    ret = {
        "mppd_loss": mppd_loss,
        "mppd_logits": mppd_logits,
        "mppd_labels": mppd_labels,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mppd_loss")(ret["mppd_loss"])
    pl_module.log(f"mppd/{phase}/loss", loss)

    return ret


def compute_mpfr(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=False, mask_image=True)
    mpfr_logits = pl_module.mpfr_score(infer["image_feats"])
    mpfr_labels = infer["image_labels_mpfr"]
    filter_to_train = infer["image_labels"].float().mean(dim=-1) != -100

    labels = mpfr_labels[filter_to_train]
    logits = mpfr_logits[filter_to_train]
    mpfr_loss = F.mse_loss(logits, labels)

    ret = {
        "mpfr_loss": mpfr_loss,
        "mpfr_logits": mpfr_logits,
        "mpfr_labels": mpfr_labels,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mpfr_loss")(ret["mpfr_loss"])
    pl_module.log(f"mpfr/{phase}/loss", loss)

    return ret


def compute_itm_wpa(pl_module, batch):
    pos_len = len(batch["text"]) // 2
    # pos_len = 1
    neg_len = len(batch["text"]) - pos_len
    itm_labels = torch.cat([torch.ones(pos_len), torch.zeros(neg_len)]).to(
        pl_module.device
    )
    itm_labels = itm_labels[torch.randperm(itm_labels.size(0))]

    itm_images = [
        torch.stack(
            [
                ti if itm_labels[i] == 1 else fi
                for i, (ti, fi) in enumerate(zip(bti, bfi))
            ]
        )
        for bti, bfi in zip(batch["image"], batch["false_image_0"])
    ]

    batch = {k: v for k, v in batch.items()}
    batch["image"] = itm_images

    infer = pl_module.infer(batch, mask_text=False, mask_image=False)

    with torch.cuda.amp.autocast(enabled=False):
        txt_emb, img_emb = infer["text_feats"], infer["image_feats"]
        txt_mask, img_mask = infer["text_masks"].bool(), infer["image_masks"].bool()
        for i, _len in enumerate(txt_mask.sum(dim=1)):
            txt_mask[i, _len - 1] = False
        txt_mask[:, 0] = False
        img_mask[:, 0] = False
        if "deit" in pl_module.hparams.config["vit"]:
            img_mask[:, 1] = False
        txt_pad, img_pad = ~txt_mask, ~img_mask

        cost = cost_matrix_cosine(txt_emb.float(), img_emb.float())
        joint_pad = txt_pad.unsqueeze(-1) | img_pad.unsqueeze(-2)
        cost.masked_fill_(joint_pad, 0)

        txt_len = (txt_pad.size(1) - txt_pad.sum(dim=1, keepdim=False)).to(
            dtype=cost.dtype
        )
        img_len = (img_pad.size(1) - img_pad.sum(dim=1, keepdim=False)).to(
            dtype=cost.dtype
        )
        T = ipot(
            cost.detach(), txt_len, txt_pad, img_len, img_pad, joint_pad, 0.5, 50, 1
        )
        distance = trace(cost.matmul(T.detach()))

    dist_pos = distance.masked_select(itm_labels == 1)
    dist_neg = distance.masked_select(itm_labels == 0)
    ot_loss = (dist_pos.sum() - dist_neg.sum()) / (dist_pos.size(0) + dist_neg.size(0))

    itm_logits = pl_module.itm_score(infer["cls_feats"])
    itm_loss = F.cross_entropy(itm_logits, itm_labels.long())

    ret = {
        "itm_loss": itm_loss,
        "itm_wpa_loss": 0.1 * ot_loss,
        "itm_logits": itm_logits,
        "itm_labels": itm_labels,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_itm_loss")(ret["itm_loss"])
    wpa_loss = getattr(pl_module, f"{phase}_itm_wpa_loss")(ret["itm_wpa_loss"])
    acc = getattr(pl_module, f"{phase}_itm_accuracy")(
        ret["itm_logits"], ret["itm_labels"]
    )
    pl_module.log(f"itm/{phase}/loss", loss)
    pl_module.log(f"itm/{phase}/wpa_loss", wpa_loss)
    pl_module.log(f"itm/{phase}/accuracy", acc)

    return ret


def compute_imgcls(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=False, mask_image=False)
    imgcls_logits = pl_module.img_classifier(infer["cls_feats"])
    imgcls_labels = batch["label"]
    imgcls_labels = torch.tensor(imgcls_labels).to(pl_module.device).long()
    imgcls_loss = F.cross_entropy(imgcls_logits, imgcls_labels)

    ret = {
        "imgcls_loss": imgcls_loss,
        "imgcls_logits": imgcls_logits,
        "imgcls_labels": imgcls_labels,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_imgcls_loss")(ret["imgcls_loss"])
    acc = getattr(pl_module, f"{phase}_imgcls_accuracy")(
        ret["imgcls_logits"], ret["imgcls_labels"]
    )
    pl_module.log(f"imgcls/{phase}/loss", loss)
    pl_module.log(f"imgcls/{phase}/accuracy", acc)

    return ret

@torch.no_grad()
def set_slot_tokens(pl_module):
    tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    for slot in slot_values_keys:
        if slot not in slot_tokens and slot not in open_slots:
            for candidate in slot_values[slot]:
                slot_tokens[slot].append((tokenizer(candidate, return_tensors='pt')['input_ids']))

def compute_dst(pl_module, batch):
    # loss_fn = nn.CrossEntropyLoss()
    ignore_index = pl_module.cross_entropy.ignore_index
    
    ## ====== prepare ====== ##
    batch['span'][batch['span'] == -1] = ignore_index
    for i, s in enumerate(batch['span']):
        if s[s != ignore_index].sum() == 0:
            batch['span'][i] = ignore_index
    batch['action'][batch['action'] == -1] = ignore_index
    extras = {'id': batch['id'], 'input_ids_len': batch['input_ids_len'], 'span': batch['span'],
        'gate': batch['gate'], 'action': batch['action'], 'slot_value': batch['slot value']}
    ## ====== infer ====== ##
    infer = pl_module.infer(batch, mask_text=False, mask_image=False)
    hidden_states = pl_module.dropout(infer["text_feats"])
    pooled_output = pl_module.dropout(infer["cls_feats"])

    span = pl_module.classifier_span(hidden_states)
    gate = pl_module.classifier_gate(pooled_output)
    action = pl_module.classifier_action(pooled_output)

    cosine_matching = []
    cache = True
    slot = batch["slot"]
    for i in range(len(slot)):
        candidate_tokens = slot_tokens[slot_values_keys[slot[i]]]
        cosine_matching.append(torch.zeros((len(candidate_tokens),)).to(pl_module.device))
        for j, candidate_token in enumerate(candidate_tokens):
            tuple_token = tuple(candidate_token.squeeze(0).numpy())
            if cache and tuple_token in pl_module.candidate_value_cache:
                candidate_output = pl_module.candidate_value_cache[tuple_token]
            else:
                candidate_token = candidate_token.to(pl_module.device)
                token_type_ids_curr = torch.ones_like(candidate_token)
                token_type_ids_curr[..., 0] = 0
                with torch.no_grad():
                    candidate_output = pl_module.pure_text_infer(candidate_token)["cls_feats"]
                if cache:
                    pl_module.candidate_value_cache[tuple_token] = candidate_output
            cosine_matching[i][j] = pooled_output[i].unsqueeze(0).mm(candidate_output.t()) / (pooled_output[i].norm() * candidate_output.norm())
    outputs = {'span': span, 'gate': gate, 'action': action, 'slot': cosine_matching}

    results = make_results(ignore_index, outputs, extras)
    ret = compute_mmconvdst_loss(pl_module, outputs, extras)

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_dst_loss")(ret["mmconv_dst_loss"])
    score = getattr(pl_module, f"{phase}_dst_DSTScore")(results)
    pl_module.log(f"mmconv_dst/{phase}/loss", loss)
    for m in ["ac", "os", "sl", "joint", "ga"]:
        pl_module.log(f"dst/{phase}/score_{m}", score[m])

    return ret


def compute_mmconvdst_loss(pl_module, outputs, extras):
    span_gt = extras['span'].to(pl_module.device)
    gate_gt = extras['gate'].to(pl_module.device)
    action_gt = extras['action'].to(pl_module.device)
    slot_gt = extras['slot_value']
    curr_maxlen = span_gt.shape[1]
    span_pred, gate_pred = outputs['span'][:, :curr_maxlen, :], outputs['gate']
    action_pred, slot_pred = outputs['action'], outputs['slot']

    batch_loss_ga = pl_module.cross_entropy(gate_pred, gate_gt)
    batch_loss_os = pl_module.cross_entropy(span_pred.reshape(-1, span_pred.shape[-1]), span_gt.view(-1))
    batch_loss_ac = pl_module.cross_entropy(action_pred, action_gt)
    batch_loss_sl = 0
    fixed_slot_sample_count = 0
    for i, slot_pd in enumerate(slot_pred):
        value_idx = slot_gt[i].item()
        if value_idx != -1:
            fixed_slot = slot_pd.detach().clone()
            fixed_slot[value_idx] = -1e7
            loss_fixed_slot = 0.2 - slot_pd[value_idx] + slot_pd[fixed_slot.argmax()]
            if loss_fixed_slot.item() > 0:
                batch_loss_sl += loss_fixed_slot
            fixed_slot_sample_count += 1

    loss = 2*batch_loss_ga + 10*batch_loss_os + batch_loss_ac
    if fixed_slot_sample_count and bool(batch_loss_sl != 0):
        batch_loss_sl /= fixed_slot_sample_count
        loss += batch_loss_sl

    return {"mmconv_dst_loss" : loss}

def compute_itm_intent(pl_module, batch):
    is_training_phase = pl_module.training
    _bs, _c, _h, _w = batch["image"][0].shape
    false_len = pl_module.hparams.config["draw_false_text"]
    intent_labels = torch.zeros(_bs, false_len+1).to(pl_module.device)
    for i, ignore_idx in enumerate(batch["ignore_idx"]):
        intent_labels[i, ignore_idx+1:] = -100
    intent_labels[:, 0] = 1
    text_ids = torch.stack(
        [batch[f"false_text_{i}_ids"] for i in range(false_len)], dim=1
    )
    text_masks = torch.stack(
        [batch[f"false_text_{i}_masks"] for i in range(false_len)], dim=1
    )
    text_labels = torch.stack(
        [batch[f"false_text_{i}_labels"] for i in range(false_len)], dim=1
    )

    text_ids = torch.cat([batch["text_ids"].unsqueeze(1), text_ids], dim=1)
    text_masks = torch.cat([batch["text_masks"].unsqueeze(1), text_masks], dim=1)
    text_labels = torch.cat([batch["text_labels"].unsqueeze(1), text_labels], dim=1)
    # text_ids = torch.cat([text_ids[:, :pos_index], batch["text_ids"].unsqueeze(1), text_ids[:, pos_index:]], dim=1)
    # text_masks = torch.cat([text_masks[:, :pos_index], batch["text_masks"].unsqueeze(1), text_masks[:, pos_index:]], dim=1)
    # text_labels = torch.cat([text_labels[:,:pos_index], batch["text_labels"].unsqueeze(1), text_labels[:,pos_index:]], dim=1)
    images = batch["image"][0].unsqueeze(1).expand(_bs, false_len + 1, _c, _h, _w)

    infer = pl_module.infer(
        {
            "image": [rearrange(images, "bs fs c h w -> (bs fs) c h w")],
            "text_ids": rearrange(text_ids, "bs fs tl -> (bs fs) tl"),
            "text_masks": rearrange(text_masks, "bs fs tl -> (bs fs) tl"),
            "text_labels": rearrange(text_labels, "bs fs tl -> (bs fs) tl")
        }
    )
    intent_logits = pl_module.itm_score(infer["cls_feats"])
    weight = torch.tensor([1.0, 5.0]).to(pl_module.device)
    intent_labels = intent_labels.reshape(-1, 1).squeeze(1)
    intent_loss = F.cross_entropy(intent_logits, intent_labels.long(), weight)
    
    ret = {
        "intent_loss": intent_loss,
        "intent_logits": intent_logits,
        "intent_labels": intent_labels,
    }
    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_intent_loss")(ret["intent_loss"])
    acc = getattr(pl_module, f"{phase}_intent_accuracy")(
        ret["intent_logits"], ret["intent_labels"]
    )
    pl_module.log(f"intent/{phase}/loss", loss)
    pl_module.log(f"intent/{phase}/accuracy", acc)
    # intent_test_wrapup([ret])
    return ret

@torch.no_grad()
def intent_test_wrapup(outs):
    ret = {"predictions":[],"labels":[]}
    for out in outs:
        logits = out["intent_logits"]
        target = out["intent_labels"]
        preds = logits.argmax(dim=-1)
        preds = preds[target != -100]
        target = target[target != -100]
        assert len(preds) == len(target)
        ret["predictions"] += preds.tolist()
        ret["labels"] += target.tolist()

    torch.distributed.barrier()
    labels = ret["labels"]
    predictions = ret["predictions"]

    precision = precision_score(labels, predictions, average="macro")
    recall = recall_score(labels, predictions, average="macro")
    f1 = f1_score(labels, predictions, average="macro")
    print("PRECISION : ", round(precision, 4), " RECALL : ",round(recall, 4), " F1-SCORE : ",round(f1, 4))

    torch.distributed.barrier()


def compute_irtr(pl_module, batch):
    is_training_phase = pl_module.training

    _bs, _c, _h, _w = batch["image"][0].shape
    false_len = pl_module.hparams.config["draw_false_text"]
    text_ids = torch.stack(
        [batch[f"false_text_{i}_ids"] for i in range(false_len)], dim=1
    )
    text_masks = torch.stack(
        [batch[f"false_text_{i}_masks"] for i in range(false_len)], dim=1
    )
    text_labels = torch.stack(
        [batch[f"false_text_{i}_labels"] for i in range(false_len)], dim=1
    )

    text_ids = torch.cat([batch["text_ids"].unsqueeze(1), text_ids], dim=1)
    text_masks = torch.cat([batch["text_masks"].unsqueeze(1), text_masks], dim=1)
    text_labels = torch.cat([batch["text_labels"].unsqueeze(1), text_labels], dim=1)
    images = batch["image"][0].unsqueeze(1).expand(_bs, false_len + 1, _c, _h, _w)
    
    infer_input = {
            "image": [rearrange(images, "bs fs c h w -> (bs fs) c h w")],
            "text_ids": rearrange(text_ids, "bs fs tl -> (bs fs) tl"),
            "text_masks": rearrange(text_masks, "bs fs tl -> (bs fs) tl"),
            "text_labels": rearrange(text_labels, "bs fs tl -> (bs fs) tl"),
    }

    if pl_module.hparams.config["use_segment_ids"] and false_len > 0:
        text_segment_ids = torch.stack(
            [batch[f"false_text_{i}_segment_ids"] for i in range(false_len)], dim=1
        )
        text_segment_ids = torch.cat([batch["text_segment_ids"].unsqueeze(1) , text_segment_ids], dim=1)
        infer_input["text_segment_ids"] = rearrange(text_segment_ids , "bs fs tl -> (bs fs) tl")

    infer = pl_module.infer(
        infer_input
    )
    score = pl_module.rank_output(infer["cls_feats"])[:, 0]
    score = rearrange(score, "(bs fs) -> bs fs", bs=_bs, fs=false_len + 1)
    answer = torch.zeros(_bs).to(score).long()
    irtr_loss = F.cross_entropy(score, answer)

    ret = {
        "irtr_loss": irtr_loss,
    }

    phase = "train" if pl_module.training else "val"
    irtr_loss = getattr(pl_module, f"{phase}_irtr_loss")(ret["irtr_loss"])

    pl_module.log(f"irtr/{phase}/irtr_loss", irtr_loss)

    return ret

@torch.no_grad()
def compute_mmdial_irtr_recall(pl_module):

    text_dset = pl_module.trainer.datamodule.dms[0].make_no_false_val_dset()
    image_mapper = text_dset.load_evalset()
    text_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    text_loader = torch.utils.data.DataLoader(
        text_dset,
        batch_size=64,
        num_workers=pl_module.hparams.config["num_workers"],
        pin_memory=True,
        collate_fn=functools.partial(
            text_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
    )
    rank_scores = list()
    text_preload = list()
    for _b in tqdm.tqdm(text_loader, desc="text prefetch loop"):
        text_preload.append(
            {
                "text_ids": _b["text_ids"].to(pl_module.device),
                "text_masks": _b["text_masks"].to(pl_module.device),
                "text_labels": _b["text_labels"].to(pl_module.device),
                "img_index": _b["img_index"],
                "image": _b["image"][0].to(pl_module.device),
                "negs_imgs": _b["negs_imgs"]
            }
        )
    for tbatch in tqdm.tqdm(text_preload, desc="text batch loop"):
        for i in range(len(tbatch["img_index"])):
            txt_batch = {
                "text_ids": tbatch["text_ids"][i].unsqueeze(0),
                "text_masks": tbatch["text_masks"][i].unsqueeze(0),
                "text_labels": tbatch["text_labels"][i].unsqueeze(0),
                "img_index": [tbatch["img_index"][i]],
                "image": tbatch["image"][i].unsqueeze(0),
                "negs_imgs": tbatch["negs_imgs"][i]
            }
            # Ground Truth Image 
            (pie, pim, _, _) = pl_module.transformer.visual_embed(
                    txt_batch["image"],
                    max_image_len=pl_module.hparams.config["max_image_len"],
                    mask_it=False,
                )
            with torch.cuda.amp.autocast():
                pos_score = pl_module.rank_output(
                    pl_module.infer(
                        {
                            "text_ids": txt_batch["text_ids"],
                            "text_masks": txt_batch["text_masks"],
                            "text_labels": txt_batch["text_labels"],
                        },
                        image_embeds=pie,
                        image_masks=pim,
                    )["cls_feats"]
                )[:, 0]
            # Negetive Images
            negs_images = txt_batch["negs_imgs"]
            negs_images_list = list()
            for i in negs_images:
                if i in image_mapper:            
                    negs_images_list.append(image_mapper[i])
            if len(negs_images_list) < len(negs_images):
                # print("negs_images_list : ", len(negs_images_list))
                negs_images_list = negs_images_list + negs_images_list[0:(len(negs_images)-len(negs_images_list))]

            image_dset = pl_module.trainer.datamodule.dms[0].make_no_false_val_dset(
                image_only=True,
                image_list=negs_images_list,
            )
            image_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
            dist_sampler = DistributedSampler(image_dset, shuffle=False)
            image_loader = torch.utils.data.DataLoader(
                image_dset,
                batch_size=64,
                num_workers=pl_module.hparams.config["num_workers"],
                sampler=dist_sampler,
                pin_memory=True,
                collate_fn=functools.partial(
                    image_dset.collate,
                    mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
                ),
            )
            image_preload = list()
            for _b in image_loader:
                (ie, im, _, _) = pl_module.transformer.visual_embed(
                    _b["image"][0].to(pl_module.device),
                    max_image_len=pl_module.hparams.config["max_image_len"],
                    mask_it=False,
                )
                image_preload.append((ie, im, _b["raw_index"][0]))

            neg_batch_score = list()
            for img_batch in image_preload:
                ie, im, _iid = img_batch
                b, _, _ = ie.shape
                _, l = txt_batch["text_ids"].shape

                text_ids = txt_batch["text_ids"].expand(b, l)
                text_masks = txt_batch["text_masks"].expand(b, l)
                text_labels = txt_batch["text_labels"].expand(b, l)

                with torch.cuda.amp.autocast():
                    score = pl_module.rank_output(
                        pl_module.infer(
                            {
                                "text_ids": text_ids,
                                "text_masks": text_masks,
                                "text_labels": text_labels,
                            },
                            image_embeds=ie,
                            image_masks=im,
                        )["cls_feats"]
                    )[:, 0]

                neg_batch_score.append(score)

            neg_batch_score = torch.cat(neg_batch_score)
            img_batch_score = torch.cat([pos_score, neg_batch_score])
            rank_scores.append(img_batch_score.cpu().tolist())
        
    torch.distributed.barrier()
    gather_rank_scores = all_gather(rank_scores)
    scores = torch.tensor(gather_rank_scores)
    scores = scores.view(scores.shape[1], -1)
    topk10 = scores.topk(10, dim=1)
    topk5 = scores.topk(5, dim=1)
    topk1 = scores.topk(1, dim=1)
    topk10_iids = topk10.indices
    topk5_iids = topk5.indices
    topk1_iids = topk1.indices
    ir_r1 = (0==topk1_iids).float().max(dim=1)[0].mean()
    ir_r5 = (0==topk5_iids).float().max(dim=1)[0].mean()
    ir_r10 = (0==topk10_iids).float().max(dim=1)[0].mean()
    return (ir_r1, ir_r5, ir_r10, -1,-1,-1)

@torch.no_grad()
def compute_old_irtr_recall(pl_module):
    text_dset = pl_module.trainer.datamodule.dms[0].make_no_false_val_dset()
    text_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    text_loader = torch.utils.data.DataLoader(
        text_dset,
        batch_size=64,
        num_workers=pl_module.hparams.config["num_workers"],
        pin_memory=True,
        collate_fn=functools.partial(
            text_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
    )

    image_dset = pl_module.trainer.datamodule.dms[0].make_no_false_val_dset(
        image_only=True
    )
    image_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    dist_sampler = DistributedSampler(image_dset, shuffle=False)
    image_loader = torch.utils.data.DataLoader(
        image_dset,
        batch_size=1,
        num_workers=pl_module.hparams.config["num_workers"],
        sampler=dist_sampler,
        pin_memory=True,
        collate_fn=functools.partial(
            image_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
    )

    text_preload = list()
    for _b in tqdm.tqdm(text_loader, desc="text prefetch loop"):
        text_preload.append(
            {
                "text_ids": _b["text_ids"].to(pl_module.device),
                "text_masks": _b["text_masks"].to(pl_module.device),
                "text_labels": _b["text_labels"].to(pl_module.device),
                "img_index": _b["img_index"],
            }
        )

    tiids = list()
    for pre in text_preload:
        tiids += pre["img_index"]
    tiids = torch.tensor(tiids)

    image_preload = list()
    for _b in tqdm.tqdm(image_loader, desc="image prefetch loop"):
        (ie, im, _, _) = pl_module.transformer.visual_embed(
            _b["image"][0].to(pl_module.device),
            max_image_len=pl_module.hparams.config["max_image_len"],
            mask_it=False,
        )
        image_preload.append((ie, im, _b["img_index"][0]))

    rank_scores = list()
    rank_iids = list()

    for img_batch in tqdm.tqdm(image_preload, desc="rank loop"):
        _ie, _im, _iid = img_batch
        _, l, c = _ie.shape

        img_batch_score = list()
        for txt_batch in text_preload:
            fblen = len(txt_batch["text_ids"])
            ie = _ie.expand(fblen, l, c)
            im = _im.expand(fblen, l)

            with torch.cuda.amp.autocast():
                score = pl_module.rank_output(
                    pl_module.infer(
                        {
                            "text_ids": txt_batch["text_ids"],
                            "text_masks": txt_batch["text_masks"],
                            "text_labels": txt_batch["text_labels"],
                        },
                        image_embeds=ie,
                        image_masks=im,
                    )["cls_feats"]
                )[:, 0]

            img_batch_score.append(score)

        img_batch_score = torch.cat(img_batch_score)
        rank_scores.append(img_batch_score.cpu().tolist())
        rank_iids.append(_iid)

    torch.distributed.barrier()
    gather_rank_scores = all_gather(rank_scores)
    gather_rank_iids = all_gather(rank_iids)

    iids = torch.tensor(gather_rank_iids)
    iids = iids.view(-1)
    scores = torch.tensor(gather_rank_scores)
    scores = scores.view(len(iids), -1)

    topk10 = scores.topk(10, dim=1)
    topk5 = scores.topk(5, dim=1)
    topk1 = scores.topk(1, dim=1)
    topk10_iids = tiids[topk10.indices]
    topk5_iids = tiids[topk5.indices]
    topk1_iids = tiids[topk1.indices]

    tr_r10 = (iids.unsqueeze(1) == topk10_iids).float().max(dim=1)[0].mean()
    tr_r5 = (iids.unsqueeze(1) == topk5_iids).float().max(dim=1)[0].mean()
    tr_r1 = (iids.unsqueeze(1) == topk1_iids).float().max(dim=1)[0].mean()

    topk10 = scores.topk(10, dim=0)
    topk5 = scores.topk(5, dim=0)
    topk1 = scores.topk(1, dim=0)
    topk10_iids = iids[topk10.indices]
    topk5_iids = iids[topk5.indices]
    topk1_iids = iids[topk1.indices]

    ir_r10 = (tiids.unsqueeze(0) == topk10_iids).float().max(dim=0)[0].mean()
    ir_r5 = (tiids.unsqueeze(0) == topk5_iids).float().max(dim=0)[0].mean()
    ir_r1 = (tiids.unsqueeze(0) == topk1_iids).float().max(dim=0)[0].mean()

    return (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10)

def compute_tr_recall_for_target_answer_set(pl_module):
    text_dset = pl_module.trainer.datamodule.dms[0].val_dataset
    is_test = pl_module.hparams.config["test_only"]
    if is_test: text_dset = pl_module.trainer.datamodule.dms[0].test_dataset
    text_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    text_dset.draw_false_text = 99
    mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator
    # image_dset = pl_module.trainer.datamodule.dms[0].make_no_false_val_dset(
    #     image_only=True
    # )
    # image_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    dist_sampler = DistributedSampler(text_dset, shuffle=False)
    option_len = text_dset.draw_false_text + 1
    text_loader = torch.utils.data.DataLoader(
        text_dset,
        batch_size = 1 ,#pl_module.hparams.config["batch_size"],
        num_workers=pl_module.hparams.config["num_workers"],
        sampler=dist_sampler,
        pin_memory=True,
        collate_fn=functools.partial(
            text_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
    )

    rank_scores = list()
    rank_iids = list()

    for dict_batch in tqdm.tqdm(text_loader, desc="rank loop"):
        _bs,_c,_h,_w = dict_batch["image"][0].shape
        text_ids = torch.stack(
            [dict_batch[f"false_text_{i}_ids"] for i in range(option_len-1)], dim=1
        )
        text_masks = torch.stack(
            [dict_batch[f"false_text_{i}_masks"] for i in range(option_len-1)], dim=1
        )
        text_labels = torch.stack(
            [dict_batch[f"false_text_{i}_labels"] for i in range(option_len-1)], dim=1
        )

        text_ids = torch.cat([dict_batch["text_ids"].unsqueeze(1), text_ids], dim=1).to(pl_module.device)
        text_masks = torch.cat([dict_batch["text_masks"].unsqueeze(1), text_masks], dim=1).to(pl_module.device)
        text_labels = torch.cat([dict_batch["text_labels"].unsqueeze(1), text_labels], dim=1).to(pl_module.device)
        images = dict_batch["image"][0].unsqueeze(1).expand(_bs,option_len,_c,_h,_w).to(pl_module.device)
        infer_input = {
            "image":[rearrange(images , "bs ol c h w -> (bs ol) c h w")],
            "text_ids":rearrange(text_ids,"bs ol tl -> (bs ol) tl"),
            "text_masks":rearrange(text_masks,"bs ol tl -> (bs ol) tl"),
            "text_labels":rearrange(text_labels,"bs ol tl -> (bs ol) tl")
        }

        if pl_module.hparams.config["use_segment_ids"] and option_len > 1:
            text_segment_ids = torch.stack(
                [dict_batch[f"false_text_{i}_segment_ids"] for i in range(option_len-1)], dim=1
            )
            text_segment_ids = torch.cat([dict_batch["text_segment_ids"].unsqueeze(1) , text_segment_ids], dim=1).to(pl_module.device)
            infer_input["text_segment_ids"] = rearrange(text_segment_ids , "bs fs tl -> (bs fs) tl")


        infer = pl_module.infer(infer_input)
        score = pl_module.rank_output(infer["cls_feats"])[:, 0]
        score = rearrange(score , "(bs ol) -> bs ol", bs=_bs, ol=option_len)
        rank_scores.extend(score.cpu().tolist())
        rank_iids.extend(dict_batch["raw_index"])

    torch.distributed.barrier()
    gather_rank_scores = all_gather(rank_scores)
    gather_rank_iids = all_gather(rank_iids)

    iids = torch.tensor(gather_rank_iids)
    iids = iids.view(-1)
    scores = torch.tensor(gather_rank_scores)
    scores = scores.view(len(iids), -1)
    topk10 = scores.topk(10, dim=1)
    topk5 = scores.topk(5, dim=1)
    topk1 = scores.topk(1, dim=1)
    gt_ids = torch.zeros(len(scores))

    #数据预处理时，已将ground truth放到options的头部，所以计算recall@k时，只需要计算topk中是否出现0即可
    tr_r10 = (gt_ids.unsqueeze(1) == topk10.indices).float().max(dim=1)[0].mean()
    tr_r5 = (gt_ids.unsqueeze(1) == topk5.indices).float().max(dim=1)[0].mean()
    tr_r1 = (gt_ids.unsqueeze(1) == topk1.indices).float().max(dim=1)[0].mean()

    rank = torch.distributed.get_rank()
    if is_test and 0 == rank:
        #由于imagechat存在少量图片的缺失，公平对比，这里使用原始对话总量
        task = pl_module.hparams.config["datasets"][0]
        if task == "imagechat":
            tr_r10 = (gt_ids.unsqueeze(1) == topk10.indices).float().max(dim=1)[0].sum()/29991
            tr_r5 = (gt_ids.unsqueeze(1) == topk5.indices).float().max(dim=1)[0].sum()/29991
            tr_r1 = (gt_ids.unsqueeze(1) == topk1.indices).float().max(dim=1)[0].sum()/29991
            
    # no need for ir metrics for text retrieval 
    return (torch.tensor(0), torch.tensor(0), torch.tensor(0) , tr_r1, tr_r5, tr_r10)

@torch.no_grad()
def compute_irtr_recall(pl_module):
    datasets = pl_module.hparams.config['datasets']
    #only calculate matching scores for targeted answer set, instead for all answers
    if 'visdial' in datasets or 'imagechat' in datasets:
        return compute_tr_recall_for_target_answer_set(pl_module)
    elif 'mmdial_caps' in datasets:
        return compute_mmdial_irtr_recall(pl_module)
    else:
        return compute_old_irtr_recall(pl_module)

def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


# def vqa_test_step(pl_module, batch, output):
#     id2answer = (
#         pl_module.trainer.datamodule.dm_dicts["vqa_trainval"].id2answer
#         if "vqa_trainval" in pl_module.trainer.datamodule.dm_dicts
#         else pl_module.trainer.datamodule.dm_dicts["vqa"].id2answer
#     )
#     vqa_logits = output["vqa_logits"]
#     vqa_preds = vqa_logits.argmax(dim=-1)
#     vqa_preds = [id2answer[pred.item()] for pred in vqa_preds]
#     questions = batch["text"]
#     qids = batch["qid"]
#     return {"qids": qids, "preds": vqa_preds}


def arc_test_step(pl_module, batch, output):
    return output


# def vqa_test_wrapup(outs, model_name):
#     rank = torch.distributed.get_rank()
#     qids, preds = list(), list()
#     for out in outs:
#         qids += out["qids"]
#         preds += out["preds"]

#     rets = list()
#     for qid, pred in zip(qids, preds):
#         rets.append({"question_id": qid, "answer": pred})
#     with open(f"vqa_submit_{rank}.json", "w") as fp:
#         json.dump(rets, fp, indent=4)

#     torch.distributed.barrier()

#     if rank == 0:
#         jsons = list()
#         paths = list(glob.glob("vqa_submit_*.json"))
#         for path in paths:
#             with open(path, "r") as fp:
#                 jsons += json.load(fp)
#         os.makedirs("result", exist_ok=True)
#         with open(f"result/vqa_submit_{model_name}.json", "w") as fp:
#             json.dump(jsons, fp, indent=4)

#     torch.distributed.barrier()
#     os.remove(f"vqa_submit_{rank}.json")


def arc_test_wrapup(outs, caplen, model_name):
    rank = torch.distributed.get_rank()
    iids, captions = list(), list()
    for out in outs:
        iids += out["iid"]
        captions += out["captions"]

    rets = list()
    for iid, caption in zip(iids, captions):
        rets.append({"image_id": iid, "caption": caption})
    with open(f"coco_cap_len{caplen}_{rank}.json", "w") as fp:
        json.dump(rets, fp, indent=4)

    torch.distributed.barrier()

    if rank == 0:
        jsons = list()
        paths = list(glob.glob(f"coco_cap_len{caplen}_*.json"))
        for path in paths:
            with open(path, "r") as fp:
                jsons += json.load(fp)
        os.makedirs("result/arc", exist_ok=True)
        jsons = sorted(jsons, key=lambda x: x["image_id"])
        with open(f"result/arc/coco_cap_{model_name}_len{caplen}.json", "w") as fp:
            json.dump(jsons, fp, indent=4)

    torch.distributed.barrier()
    os.remove(f"coco_cap_len{caplen}_{rank}.json")

def rg_test_step(pl_module, batch):
    stop_token = pl_module.hparams.config["stop_token"]
    pl_module.tokenizer.pad_token = pl_module.tokenizer.bos_token
    pl_module.tokenizer.padding_side='left'
    preds = []
    batch_prompt = pl_module.tokenizer(batch["prompt"], add_special_tokens=True, padding=True, return_tensors="pt")["input_ids"].to(pl_module.device)
    output_sequences = pl_module.transforms.generate(
            input_ids=batch_prompt,
            max_length=800,
            pad_token_id=50256,
            temperature=pl_module.hparams.config["temperature"],
            top_k=pl_module.hparams.config["top_k"],
            top_p=pl_module.hparams.config["top_p"],
            repetition_penalty=1.0,
            do_sample=True,
            num_return_sequences=1,
        )
    if len(output_sequences.shape) > 2:
        output_sequences.squeeze_()
    for gen_idx, generated_sequence in enumerate(output_sequences):
        generated_sequence = generated_sequence.tolist()
        # Decode text
        text = pl_module.tokenizer.decode(
            generated_sequence, clean_up_tokenization_spaces=True
        )
        # Remove all text after the stop token
        stop_idx = text.find(stop_token)+len(stop_token) if stop_token else None
        text = text[:stop_idx]
        total_sequence = (
            batch["prompt"][gen_idx]
            + text[
                len(
                    pl_module.tokenizer.decode(
                        batch_prompt[gen_idx], clean_up_tokenization_spaces=True
                    )
                ) :
            ]
        )
        preds.append(total_sequence)

    return {"preds" : preds, "labels" : batch["text"], "ids": batch["ids"]}

