"""
PretrainUnifiedTransformer
"""

import torch
import torch.nn as nn

from galaxy.args import str2bool
from galaxy.models.unified_transformer import UnifiedTransformer
from galaxy.utils.criterions import compute_kl_loss
from galaxy.utils.eval import DAEvaluation


class PretrainUnifiedTransformer(UnifiedTransformer):
    """
    Implement unified transformer for pre-training.
    """

    @classmethod
    def add_cmdline_argument(cls, group):
        """ Add cmdline argument. """
        group.add_argument("--with_filter", type=str2bool, default=False,
                           help="Whether to filter OOD distributions in pre-training data.")
        group.add_argument("--detach_filter", type=str2bool, default=True,
                           help="Whether to backpropagation through the filter scores.")
        group.add_argument("--filter_index", type=int, default=1,
                           help="The tag of dataset to be filtered, 0 for UniDA while 1 for UnDial.")
        group.add_argument("--kl_ratio", type=float, default=1.0,
                           help="The ratio of category kl loss.")
        UnifiedTransformer.add_cmdline_argument(group)
        return group

    def __init__(self, hparams, generator):
        super(PretrainUnifiedTransformer, self).__init__(hparams, generator)
        self.with_filter = hparams.with_filter
        self.detach_filter = hparams.detach_filter
        self.filter_index = hparams.filter_index
        self.kl_ratio = hparams.kl_ratio

        if self.with_joint_act:
            self.act_classifier = nn.Linear(self.hidden_dim, self.num_act)
        return

    def _compute_gates(self, logits, detach):
        """ Compute gates to filter OOD distributions in UnDial. """
        logits = logits.detach() if detach else logits
        probs = self.softmax(logits)
        max_entropy = torch.log(torch.tensor(self.num_act).float())
        max_entropy = max_entropy.to(logits.device)
        entropy = -probs * torch.log(probs + 1e-12)
        entropy = torch.sum(entropy, dim=-1)
        # return torch.clip((max_entropy - entropy - torch.log(entropy + 1e-12)) / max_entropy, 0, 1)
        return torch.clamp((max_entropy - entropy - torch.log(entropy + 1e-12)) / max_entropy, 0, 1)

    def _forward(self, inputs, is_training):
        """ Real forward process of model. """
        outputs = {}

        latent_embed, enc_embed, dec_embed = self._mask_encoder_decoder_network(
            src_token=inputs['src_token'],
            src_mask=inputs['src_mask'],
            tgt_token=inputs['tgt_token'][:, :-1],
            tgt_mask=inputs['tgt_mask'][:, :-1],
            src_pos=inputs['src_pos'],
            src_type=inputs['src_type'],
            src_turn=inputs['src_turn'],
            tgt_pos=inputs['tgt_pos'][:, :-1],
            tgt_type=inputs['tgt_type'][:, :-1],
            tgt_turn=inputs['tgt_turn'][:, :-1]
        )

        if self.with_joint_act:
            joint_act_logits = self.act_classifier(latent_embed)
            outputs["joint_act_logits"] = joint_act_logits
            joint_act_probs = self.sigmoid(joint_act_logits)
            outputs["joint_act_probs"] = joint_act_probs

            if self.with_filter:
                filter_scores = self._compute_gates(logits=joint_act_logits, detach=self.detach_filter)
                outputs['filter_scores'] = filter_scores

        outputs["dec_probs"] = self._dec_head(dec_embed=dec_embed)
        return outputs

    def _collect_metrics(self, inputs, outputs):
        """ Collect metrics for optimization and log. """
        metrics = {}
        bsz = inputs['src_token'].size(0)
        if self.with_rdrop_act:
            bsz //= 2

        tgt_len = torch.sum(torch.sum(inputs["tgt_mask"][:bsz], dim=1) - 1)
        label = inputs["tgt_token"][:bsz, 1:]
        nll = self.nll_loss(torch.log(outputs["dec_probs"][:bsz] + 1e-12).permute(0, 2, 1), label)
        nll = torch.sum(nll, dim=1)
        token_nll = torch.sum(nll) / tgt_len
        nll = torch.mean(nll)
        metrics["nll"] = nll
        metrics["token_nll"] = token_nll
        loss = token_nll if self.token_loss else nll

        if self.with_joint_act:
            output_act_probs = outputs["joint_act_probs"]
            bce = self.bce_loss(output_act_probs, inputs['act_index'].float())
            bce = torch.mean(bce, dim=-1)

            act_mask = (0 == inputs["tag"]).long()
            bce_len = torch.sum(act_mask)
            bce = bce * act_mask
            if bce_len != 0:
                bce = torch.sum(bce) / bce_len
            else:
                bce = torch.sum(bce)
            metrics['bce'] = bce
            loss = loss + bce * self.bce_ratio
            da_predictions = (output_act_probs > 0.5).int()
            da_preds = da_predictions.detach().cpu().numpy()
            da_labels = inputs['act_index'].detach().cpu().numpy()

            act_mask_np = (act_mask.detach().cpu().numpy() == 1)
            if bce_len != 0:
                da_result = DAEvaluation(preds=da_preds[act_mask_np], labels=da_labels[act_mask_np])
            else:
                da_result = {'f1_micro': 0.0}
            bce_f1 = torch.tensor(da_result['f1_micro'], device=loss.device)
            metrics['bce_f1'] = bce_f1
        else:
            bce, bce_f1, bce_len = None, None, None

        if self.with_filter:
            filter_scores = outputs['filter_scores']
            metrics['gates'] = torch.mean(filter_scores)

        if self.with_rdrop_act:
            output_act_logits = outputs["joint_act_logits"]
            if self.with_filter:
                filter_scores = outputs['filter_scores']
                kl_mask = (self.filter_index != inputs["tag"])
                filter_scores = filter_scores.masked_fill(kl_mask, 1.)
                # filter_scores = (filter_scores[:bsz] + filter_scores[bsz:]) / 2
                filter_scores = filter_scores[:bsz]
                act_kl = compute_kl_loss(p=output_act_logits[:bsz], q=output_act_logits[bsz:],
                                         filter_scores=filter_scores)
            else:
                act_kl = compute_kl_loss(p=output_act_logits[:bsz], q=output_act_logits[bsz:])
            metrics['act_kl'] = act_kl
            loss = loss + act_kl * self.kl_ratio
        else:
            act_kl = None

        metrics["loss"] = loss
        metrics["token_num"] = tgt_len
        metrics["bce_num"] = bce_len

        return metrics

    def _optimize(self, loss, optimizer=None, lr_scheduler=None):
        """ Optimize loss function and update model. """
        assert optimizer is not None
        optimizer.zero_grad()
        loss.backward()

        if self.grad_clip is not None and self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(parameters=self.parameters(), max_norm=self.grad_clip)
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        return


PretrainUnifiedTransformer.register("PretrainUnifiedTransformer")
