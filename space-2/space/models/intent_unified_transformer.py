"""
IntentUnifiedTransformer
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from space.args import str2bool
from space.models.unified_transformer import UnifiedTransformer
from space.utils.criterions import compute_kl_loss


class IntentUnifiedTransformer(UnifiedTransformer):
    """
    Implement intent unified transformer.
    """

    @classmethod
    def add_cmdline_argument(cls, group):
        """ Add cmdline argument. """
        group.add_argument("--example", type=str2bool, default=False,
                           help="Whether to use Example-Driven Training.")
        group.add_argument("--num_intent", type=int, default=0,
                           help="The category of intent for multi-classification.")
        group.add_argument("--with_rdrop", type=str2bool, default=False,
                           help="Whether to use R-Drop for intent recognition.")
        group.add_argument("--kl_ratio", type=float, default=5.0,
                           help="The ratio of kl loss of R-drop.")
        UnifiedTransformer.add_cmdline_argument(group)
        return group

    def __init__(self, hparams, reader, generator):
        super(IntentUnifiedTransformer, self).__init__(hparams, reader, generator)
        self.example = hparams.example
        self.num_intent = hparams.num_intent
        self.with_rdrop = hparams.with_rdrop
        self.kl_ratio = hparams.kl_ratio
        self.loss_fct = nn.CrossEntropyLoss()
        if self.example:
            self.loss_fct = nn.NLLLoss()
        else:
            self.intent_classifier = nn.Linear(self.hidden_dim, self.num_intent)
            self.loss_fct = nn.CrossEntropyLoss()

        if self.use_gpu:
            self.cuda()
        return

    def _forward(self, inputs, is_training, with_label):
        """ Real forward process of model in different mode(train/test). """
        def aug(v):
            assert isinstance(v, torch.Tensor)
            return torch.cat([v, v], dim=0)

        outputs = {}

        if self.with_mlm:
            mlm_embed = self._encoder_network(input_token=inputs['mlm_token'],
                                              input_mask=inputs['src_mask'],
                                              input_pos=inputs['src_pos'],
                                              input_type=inputs['src_type'],
                                              input_turn=inputs['src_turn'])
            mlm_embed = self.mlm_transform(mlm_embed)
            mlm_logits = torch.matmul(mlm_embed, self.embedder.token_embedding.weight.T) + self.mlm_bias
            mlm_probs = self.softmax(mlm_logits)
            outputs["mlm_probs"] = mlm_probs

        if self.with_cls:
            if self.with_rdrop:
                latent_embed, enc_embed = self._mask_encoder_network(input_token=aug(inputs['src_token']),
                                                                     input_mask=aug(inputs['src_mask']),
                                                                     input_pos=aug(inputs['src_pos']),
                                                                     input_type=aug(inputs['src_type']),
                                                                     input_turn=aug(inputs['src_turn']))
            else:
                latent_embed, enc_embed = self._mask_encoder_network(input_token=inputs['src_token'],
                                                                     input_mask=inputs['src_mask'],
                                                                     input_pos=inputs['src_pos'],
                                                                     input_type=inputs['src_type'],
                                                                     input_turn=inputs['src_turn'])
            features = latent_embed
        else:
            if self.with_rdrop:
                enc_embed, dec_embed = self._encoder_decoder_network(src_token=aug(inputs['src_token']),
                                                                     src_mask=aug(inputs['src_mask']),
                                                                     tgt_token=aug(inputs['tgt_token']),
                                                                     tgt_mask=aug(inputs['tgt_mask']),
                                                                     src_pos=aug(inputs['src_pos']),
                                                                     src_type=aug(inputs['src_type']),
                                                                     src_turn=aug(inputs['src_turn']))
            else:
                enc_embed, dec_embed = self._encoder_decoder_network(src_token=inputs['src_token'],
                                                                     src_mask=inputs['src_mask'],
                                                                     tgt_token=inputs['tgt_token'],
                                                                     tgt_mask=inputs['tgt_mask'],
                                                                     src_pos=inputs['src_pos'],
                                                                     src_type=inputs['src_type'],
                                                                     src_turn=inputs['src_turn'])
            features = dec_embed[:, -1]

        if self.with_project:
            features = self.subspace(features).squeeze(1)
        elif self.with_pool:
            features = self.pooler(features)

        if self.example:
            assert not self.with_rdrop
            if self.with_cls:
                ex_latent_embed, ex_enc_embed = self._mask_encoder_network(input_token=inputs['example_src_token'],
                                                                           input_mask=inputs['example_src_mask'],
                                                                           input_pos=inputs['example_src_pos'],
                                                                           input_type=inputs['example_src_type'],
                                                                           input_turn=inputs['example_src_turn'])
                ex_features = ex_latent_embed
            else:
                ex_enc_embed, ex_dec_embed = self._encoder_decoder_network(src_token=inputs['example_src_token'],
                                                                           src_mask=inputs['example_src_mask'],
                                                                           tgt_token=inputs['example_tgt_token'],
                                                                           tgt_mask=inputs['example_tgt_mask'],
                                                                           src_pos=inputs['example_src_pos'],
                                                                           src_type=inputs['example_src_type'],
                                                                           src_turn=inputs['example_src_turn'])
                ex_features = ex_dec_embed[:, -1]

            if self.with_project:
                ex_features = self.subspace(ex_features).squeeze(1)
            elif self.with_pool:
                ex_features = self.pooler(ex_features)

            probs = self.softmax(features.mm(ex_features.t()))
            example_intent = inputs['example_intent'].unsqueeze(0)
            intent_probs = torch.zeros(probs.size(0), self.num_intent)
            intent_probs = intent_probs.cuda() if self.use_gpu else intent_probs
            intent_probs = intent_probs.scatter_add(-1, example_intent.repeat(probs.size(0), 1), probs)
            outputs['intent_probs'] = intent_probs
        else:
            intent_logits = self.intent_classifier(features)
            outputs['intent_logits'] = intent_logits

        return outputs

    def _collect_metrics(self, inputs, outputs, with_label, data_file):

        metrics = {}
        batch_size = inputs['src_token'].size(0)

        intent_label = torch.cat([inputs['intent_label'], inputs['intent_label']], dim=0) \
            if self.with_rdrop else inputs['intent_label']

        if self.example:
            intent_loss = self.loss_fct(torch.log(outputs['intent_probs'] + 1e-12).view(-1, self.num_intent),
                                        intent_label.type(torch.long))
        else:
            intent_loss = self.loss_fct(outputs['intent_logits'].view(-1, self.num_intent),
                                        intent_label.type(torch.long))
        metrics["intent_loss"] = intent_loss
        loss = intent_loss

        if self.with_mlm:
            mlm_num = torch.sum(torch.sum(inputs["mlm_mask"], dim=1))
            mlm = self.nll_loss(torch.log(outputs["mlm_probs"] + 1e-12).permute(0, 2, 1), inputs["mlm_label"])
            mlm = torch.sum(mlm, dim=1)
            token_mlm = torch.sum(mlm) / mlm_num
            mlm = torch.mean(mlm)
            metrics["mlm"] = mlm
            metrics["token_mlm"] = token_mlm
            metrics["mlm_num"] = mlm_num
            loss = loss + (token_mlm if self.token_loss else mlm) * self.mlm_ratio
        else:
            mlm, token_mlm, mlm_num = None, None, None

        if self.with_rdrop:
            kl = compute_kl_loss(p=outputs['intent_logits'][:batch_size],
                                 q=outputs['intent_logits'][batch_size:])
            metrics['kl'] = kl
            loss = loss + kl * self.kl_ratio
        else:
            kl = None

        metrics["loss"] = loss

        if self.gpu > 1:
            return intent_loss, mlm, token_mlm, mlm_num, kl
        else:
            return metrics

    def _infer(self, inputs):
        """ Real inference process of model. """
        results = {}

        if self.with_cls:
            latent_embed, enc_embed = self._mask_encoder_network(input_token=inputs['src_token'],
                                                                 input_mask=inputs['src_mask'],
                                                                 input_pos=inputs['src_pos'],
                                                                 input_type=inputs['src_type'],
                                                                 input_turn=inputs['src_turn'])
            features = latent_embed
        else:
            enc_embed, dec_embed = self._encoder_decoder_network(src_token=inputs['src_token'],
                                                                 src_mask=inputs['src_mask'],
                                                                 tgt_token=inputs['tgt_token'],
                                                                 tgt_mask=inputs['tgt_mask'],
                                                                 src_pos=inputs['src_pos'],
                                                                 src_type=inputs['src_type'],
                                                                 src_turn=inputs['src_turn'])
            features = dec_embed[:, -1]

        if self.with_project:
            features = self.subspace(features).squeeze(1)
        elif self.with_pool:
            features = self.pooler(features)

        if self.example:
            results['features'] = features
        else:
            intent_logits = self.intent_classifier(features)
            intent_probs = self.softmax(intent_logits)
            results['intent_probs'] = intent_probs

        return results


IntentUnifiedTransformer.register("IntentUnifiedTransformer")
