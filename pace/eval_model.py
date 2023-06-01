'''
    @author : yzc
    @date : 2022-11-06:20:16
    evalation scripts
'''
import torch
import copy
import tqdm
import json
import os
import copy
import re
import pytorch_lightning as pl
from PIL import Image
from einops import rearrange
from pace.config import ex
from pace.modules import TransformerSS , TransformerSSDecode
from pace.transforms import pixelbert_transform
from pace.utils.format_simmc_generation import main
from pace.datamodules.multitask_datamodule import MTDataModule
from pace.modules.dist_utils import all_gather
from torch.utils.data.distributed import DistributedSampler
from pace.gadgets.my_metrics import Accuracy, VQAScore, Scalar, NDCG, BLEUScorer
from pace.utils.eval_mmconv_rg import evaluate_mmconvrg
from pace.utils.format_simmc_dst_generation import format_for_dst
from pace.utils.eval_simmc2_dst import evaluate_from_flat_list

import functools

def compute_tr_recall(model,dm_module,type,get_relevance_tensor=False):
    assert type in ['val','test']

    dms = dm_module.dms
    if type == 'val':
        text_dset = dms[0].val_dataset
    elif type == 'test':
        text_dset = dms[0].test_dataset
    
    text_dset.tokenizer = dms[0].tokenizer

    # dist_sampler = DistributedSampler(text_dset, shuffle=False)
    text_dset.draw_false_text = 99
    option_len = 100
    text_loader = torch.utils.data.DataLoader(
        text_dset,
        batch_size = 1 ,#model.hparams.config["batch_size"],
        num_workers=model.hparams.config["num_workers"],
        # sampler=dist_sampler,
        pin_memory=True,
        collate_fn=functools.partial(
            text_dset.collate,
            mlm_collator=dms[0].mlm_collator,
        ),
    )

    rank_scores = list()
    rank_iids = list()
    relevance_scores = list()
    ret = {
        "scores":None,
        "relevance_scores":None
    }
    with torch.no_grad():
        for dict_batch in tqdm.tqdm(text_loader, desc=f"{type}:rank loop"):
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

            text_ids = torch.cat([dict_batch["text_ids"].unsqueeze(1), text_ids], dim=1).to(model.device)
            text_masks = torch.cat([dict_batch["text_masks"].unsqueeze(1), text_masks], dim=1).to(model.device)
            text_labels = torch.cat([dict_batch["text_labels"].unsqueeze(1), text_labels], dim=1).to(model.device)
            images = dict_batch["image"][0].unsqueeze(1).expand(_bs,option_len,_c,_h,_w).to(model.device)
            infer_input = {
                "image":[rearrange(images , "bs ol c h w -> (bs ol) c h w")],
                "text_ids":rearrange(text_ids,"bs ol tl -> (bs ol) tl"),
                "text_masks":rearrange(text_masks,"bs ol tl -> (bs ol) tl"),
                "text_labels":rearrange(text_labels,"bs ol tl -> (bs ol) tl")
            }

            if "false_text_0_segment_ids" in dict_batch:
                text_segment_ids = torch.stack(
                    [dict_batch[f"false_text_{i}_segment_ids"] for i in range(option_len-1)], dim=1
                )
                text_segment_ids = torch.cat([dict_batch["text_segment_ids"].unsqueeze(1) , text_segment_ids], dim=1).to(model.device)
                infer_input["text_segment_ids"] = rearrange(text_segment_ids , "bs fs tl -> (bs fs) tl")


            infer = model.infer(infer_input)
            score = model.rank_output(infer["cls_feats"])[:, 0]
            score = rearrange(score , "(bs ol) -> bs ol", bs=_bs, ol=option_len)
            rank_scores.extend(score.cpu().tolist())
            rank_iids.extend(dict_batch["raw_index"])

            if get_relevance_tensor:
                curr_relevance_scores = torch.stack(
                    [dict_batch[f"false_text_{i}_relevance"] for i in range(option_len-1)], dim=1
                )
                curr_relevance_scores = torch.cat([dict_batch["text_relevance"].unsqueeze(1) , curr_relevance_scores], dim=1)
                relevance_scores.extend(curr_relevance_scores.tolist())

    # torch.distributed.barrier()
    gather_rank_scores = all_gather(rank_scores)
    gather_relevance_scores = all_gather(relevance_scores)
    gather_rank_iids = all_gather(rank_iids)

    iids = torch.tensor(gather_rank_iids)
    iids = iids.view(-1)
    scores = torch.tensor(gather_rank_scores)
    relevance_scores = torch.tensor(gather_relevance_scores)
    scores = scores.view(len(iids), -1)
    relevance_scores = relevance_scores.view(len(iids), -1)
    ret["scores"] = scores
    if get_relevance_tensor:
        ret["relevance_scores"] = relevance_scores
    return ret

def calculate_imagechat_test_rank(model,dm,tot_size):
    scores = compute_tr_recall(model,dm,'test')["scores"]
    topk10 = scores.topk(10, dim=1)
    topk5 = scores.topk(5, dim=1)
    topk1 = scores.topk(1, dim=1)

    #数据预处理时，已将ground truth放到options的头部，所以计算recall@k时，只需要计算topk中是否出现0即可
    gt_ids = torch.zeros(len(scores))
    tr_r10 = (gt_ids.unsqueeze(1) == topk10.indices).float().max(dim=1)[0].sum()
    tr_r5 = (gt_ids.unsqueeze(1) == topk5.indices).float().max(dim=1)[0].sum()
    tr_r1 = (gt_ids.unsqueeze(1) == topk1.indices).float().max(dim=1)[0].sum()

    return (tr_r1.item()/tot_size, tr_r5.item()/tot_size, tr_r10.item()/tot_size)

def detokenize(tk_list):
    r_list = []
    for tk in tk_list:
        if tk.startswith('##') and len(r_list) > 0:
            r_list[-1] = r_list[-1] + tk[2:]
        else:
            r_list.append(tk)
    tk_list = r_list
    r_list = []
    flag = False
    for i,tk in enumerate(tk_list):
        #i'll that's
        flag = False
        if len(r_list)>0 and r_list[-1] in ["'" , "-", "/", "&" , "_", "$"] :
            x = r_list[-1]
            if len(r_list)>1:
                y = r_list[-2]
                r_list = r_list[:-2]
                x = y+x+tk
            else:
                r_list = r_list[:-1]
                x = x+tk
            r_list.append(x)
            flag = True
        elif len(r_list)>0 and r_list[-1] == ".":
            x = r_list[-1]
            if len(r_list)>1:
                y = r_list[-2]
                if re.match("\d+",tk) and re.match("\d+",y):
                    r_list = r_list[:-2]
                    x = y+x+tk
                    if len(r_list)>0:
                        z = r_list[-1]
                        if z == '$':
                            r_list = r_list[:-1]
                            x = z+x
                    r_list.append(x)
                    flag = True
        elif len(r_list)>0 and (r_list[-1] in ["#", "(", "<" , "["] or tk in [")" , ">", "]"] ):
            r_list[-1] += tk
            flag = True
        if not flag:
            r_list.append(tk)
    while len(r_list)>0 and r_list[0] in [".", "?", "!", ","]:
        r_list.pop(0)
    return r_list


def generation(model,dm_module,type,decode_prompt_text=None):
    dms = dm_module.dms
    if type == 'val':
        text_dset = dms[0].make_no_false_val_dset()
    elif type == 'test':
        text_dset = dms[0].test_dataset
    
    tokenizer = text_dset.tokenizer = dms[0].tokenizer
    text_loader = torch.utils.data.DataLoader(
        text_dset,
        batch_size=16,
        num_workers=model.hparams.config["num_workers"],
        pin_memory=True,
        shuffle=False,
        collate_fn=functools.partial(
            text_dset.collate,
            mlm_collator=dms[0].mlm_collator,
        ),
    )
    outputs = list()
    decode_prompt = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(decode_prompt_text)) if decode_prompt_text != None else None
    print("decode_prompt: " , decode_prompt , decode_prompt_text)
    with torch.no_grad():
        for _b in tqdm.tqdm(text_loader, desc="generation loop"):
            output_ids  = model(_b , decode_prompt=decode_prompt)['pred_seq']
            texts = _b["text"]
            text_ids = _b["text_ids"]
            for i in range(len(texts)):
                # sent = text_dset.tokenizer.decode(output_ids[i])
                output_tokens = text_dset.tokenizer.convert_ids_to_tokens(output_ids[i])
                sent = ' '.join(detokenize(output_tokens))
                splits = sent.split("[SEP]")
                result = ""
                for split in splits:
                    split_sent = split.replace("[PAD]", "").strip()
                    if len(split_sent)>0:
                        result = split_sent
                        break
                outputs.append(result)
    return outputs


@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    device = "cuda:0" if _config["num_gpus"] > 0 else "cpu"
    pl.seed_everything(_config["seed"])
    dm = MTDataModule(_config, dist=False)
    dm.setup("test")
    tokenizer = dm.dms[0].tokenizer
    model = TransformerSS(_config)
    model.to(device)
    model.eval()

    task = _config['datasets'][0]
    name = _config['load_path'].split('/')[-1].split('.')[0]
    if task == 'imagechat':
        raw_test_json_path = f"{_config['data_root']}/imagechat_test.json"
        with open(raw_test_json_path,'r') as raw_test_json_file:
            raw_test_json = json.load(raw_test_json_file)
            tot_size = 0
            for dialog in raw_test_json:
                tot_size += len(dialog['candidates'])
            with open(f"{name}_test_result.json",'a+') as output:
                data = calculate_imagechat_test_rank(model,dm,tot_size)
                json.dump(data,output)

    elif "gen" in task:
        model = TransformerSSDecode(_config)
        model.to(device)
        model.eval()
        output = generation(model, dm , "test")

    elif task == "mmconvrg":
        prompt = _config['decode_prompt'] #"<|belief|>"
        decode_prompt = tokenizer(prompt , add_special_tokens=False)
        model = TransformerSSDecode(_config, search_beam_size=1)
        model.to(device)
        model.eval()
        output = generation(model, dm , "test" ,prompt)

        for idx in range(len(output)):
            if output[idx].find("<|belief|>") == -1:
                output[idx] = "<|belief|>" + output[idx] 
            if output[idx].find("<|endofresponse|>") == -1:
                output[idx] = output[idx] + "<|endofresponse|>"

        with open(f"results/rg/mmconv/{name}_mmconv_rg_test.json","w") as f:
            json.dump(output , f)

        
        dataset = dm.dms[0].test_dataset
        gt =  dataset.target_sents
        ret = dict()
        for idx in range(len(gt)):
            ret[idx] = [{"pred":output[idx] , "label":gt[idx]}]
        evaluate_mmconvrg(ret)

    elif task == "simmc2dst":
        prompt = _config['decode_prompt'] #"belief state : "
        decode_prompt = tokenizer(prompt, add_special_tokens=False).input_ids
        model = TransformerSSDecode(_config, search_beam_size=5)
        model.to(device)
        model.eval()
        output = generation(model, dm , "test" , prompt)
        output_file = f"results/dst/simmc/{name}_simmc2_dst_devtest_output.json"
        with open(output_file, "w") as f:
            json.dump(output , f)
        formated_output_file = f"results/dst/simmc/{name}_simmc2_dst_devtest_format_output.json"
        target_file = "../pace/utils/simmc2/rerank_simmc2.1_devtest_dst_3turns_target_special_tokens.txt"

        formated_output = format_for_dst(output)
        with open(formated_output_file ,"w") as f:
            json.dump(formated_output,f)
        with open(target_file, "r") as f:
            target_output = format_for_dst(f.readlines())
        report = evaluate_from_flat_list(formated_output , target_output)
        with open(f"results/dst/simmc/{name}_simmc2_dst_devtest_report.json", "w") as f:
            json.dump(report , f)
        print(report)

    elif task == "simmc2rg":
        model = TransformerSSDecode(_config, search_beam_size=5)
        model.to(device)
        model.eval()
        output = generation(model, dm , "test")
        dataset = dm.dms[0].test_dataset
        turn_ids = dataset.turn_ids
        formated_output = list()
        assert len(turn_ids) == len(dataset)
        for i in range(len(turn_ids)):
            formated_output.append({"turn_id":str(turn_ids[i]), "predictions":output[i]})
        with open(f"results/rg/simmc/{name}_simmc_rg_devtest_output.json","w") as f:
            json.dump(formated_output,f)
        args = dict()
        args['generation_pred_json'] = f"results/rg/simmc/{name}_simmc_rg_devtest_output.json"
        args['split_path'] = "/results/simmc2.1_dials_dstc11_devtest.json"
        args['save_path'] = f"results/rg/simmc/{name}_simmc_rg_devtest_formated_output.json"
        main(args)