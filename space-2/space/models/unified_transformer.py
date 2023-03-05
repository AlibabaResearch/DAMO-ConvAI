"""
UnifiedTransformer
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from space.args import str2bool
from space.modules.embedder import Embedder
from space.models.model_base import ModelBase
from space.modules.subspace import Subspace
from space.modules.transformer_block import TransformerBlock
from space.utils.criterions import SupConLoss


class UnifiedTransformer(ModelBase):
    """
    Implement unified transformer.
    """

    @classmethod
    def add_cmdline_argument(cls, group):
        """ Add cmdline argument. """
        group.add_argument("--num_token_embeddings", type=int, default=-1,
                           help="The number of tokens in vocabulary. "
                           "It will be automatically calculated after loading vocabulary.")
        group.add_argument("--num_pos_embeddings", type=int, default=512,
                           help="The maximum number of position.")
        group.add_argument("--num_type_embeddings", type=int, default=2,
                           help="The number of different type of tokens.")
        group.add_argument("--num_turn_embeddings", type=int, default=17,
                           help="The maximum number of turn.")
        group.add_argument("--temperature", type=float, default=0.07,
                           help="The temperature of contrastive loss.")
        group.add_argument("--hidden_dim", type=int, default=768,
                           help="The size of hidden vector in transformer.")
        group.add_argument("--subspace_dim", type=int, default=768,
                           help="The size of hidden vector in subspace.")
        group.add_argument("--num_heads", type=int, default=12,
                           help="The number of heads in multi head attention.")
        group.add_argument("--num_layers", type=int, default=12,
                           help="The number of layers in transformer.")
        group.add_argument("--padding_idx", type=int, default=0,
                           help="The padding index.")
        group.add_argument("--dropout", type=float, default=0.1,
                           help="The dropout ratio after multi head attention and feed forward network.")
        group.add_argument("--embed_dropout", type=float, default=0.0,
                           help="The dropout ratio of embedding layers.")
        group.add_argument("--attn_dropout", type=float, default=0.1,
                           help="The dropout ratio of multi head attention.")
        group.add_argument("--ff_dropout", type=float, default=0.1,
                           help="The dropout ratio of feed forward network.")
        group.add_argument("--mlm_ratio", type=float, default=0.1,
                           help="The mlm loss ratio of total loss.")
        group.add_argument("--pos_trainable", type=str2bool, default=True,
                           help="Whether to train position embeddings.")
        group.add_argument("--bidirectional_context", type=str2bool, default=True,
                           help="Whether to use bidirectional self-attention in context tokens.")
        group.add_argument("--with_pool", type=str2bool, default=False,
                           help="Whether to use pool op on top of final hidden representation.")
        group.add_argument("--with_project", type=str2bool, default=True,
                           help="Whether to use project op on top of final hidden representation.")
        group.add_argument("--trigger_subspaces", type=str, default="",
                           help="The name of triggered subspaces for training.")
        group.add_argument("--label_smooth", type=float, default=0.0,
                           help="Use soft label to calculate NLL loss and BoW loss.")
        group.add_argument("--initializer_range", type=float, default=0.02,
                           help="Use to initialize parameters.")
        group.add_argument("--lr", type=float, default=5e-5,
                           help="The inital learning rate for Adam.")
        group.add_argument("--weight_decay", type=float, default=0.0,
                           help="The weight decay for Adam.")
        group.add_argument("--max_grad_norm", type=float, default=5.0,
                           help="The maximum norm of gradient.")
        return group

    def __init__(self, hparams, reader, generator, dtype="float32"):
        super(UnifiedTransformer, self).__init__(hparams)
        self.reader = reader
        self.generator = generator
        self.num_token_embeddings = hparams.num_token_embeddings
        self.num_pos_embeddings = hparams.num_pos_embeddings
        self.num_type_embeddings = hparams.num_type_embeddings
        self.num_turn_embeddings = hparams.num_turn_embeddings
        self.temperature = hparams.temperature
        self.hidden_dim = hparams.hidden_dim
        self.subspace_dim = hparams.subspace_dim
        self.num_heads = hparams.num_heads
        self.num_layers = hparams.num_layers
        self.padding_idx = hparams.padding_idx
        self.dropout = hparams.dropout
        self.embed_dropout = hparams.embed_dropout
        self.attn_dropout = hparams.attn_dropout
        self.ff_dropout = hparams.ff_dropout
        self.mlm_ratio = hparams.mlm_ratio
        self.pos_trainable = hparams.pos_trainable
        self.bidirectional_context = hparams.bidirectional_context
        self.label_smooth = hparams.label_smooth
        self.initializer_range = hparams.initializer_range
        self.token_loss = hparams.token_loss
        self.learning_method = hparams.learning_method
        self.trigger_subspaces = hparams.trigger_subspaces
        self.with_contrastive = hparams.with_contrastive
        self.with_project = hparams.with_project
        self.with_pool = hparams.with_pool
        self.with_mlm = hparams.with_mlm
        self.with_cls = hparams.with_cls
        self._dtype = dtype

        self.embedder = Embedder(self.hidden_dim,
                                 self.num_token_embeddings,
                                 self.num_pos_embeddings,
                                 self.num_type_embeddings,
                                 self.num_turn_embeddings,
                                 padding_idx=self.padding_idx,
                                 dropout=self.embed_dropout,
                                 pos_trainable=self.pos_trainable)
        self.embed_layer_norm = nn.LayerNorm(normalized_shape=self.hidden_dim,
                                             eps=1e-12,
                                             elementwise_affine=True)

        self.layers = nn.ModuleList([TransformerBlock(self.hidden_dim,
                                     self.num_heads,
                                     self.dropout,
                                     self.attn_dropout,
                                     self.ff_dropout) for _ in range(hparams.num_layers)])

        if self.with_mlm:
            self.mlm_transform = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.GELU(),
                nn.LayerNorm(normalized_shape=self.hidden_dim,
                             eps=1e-12,
                             elementwise_affine=True)
            )
            self.mlm_bias = nn.Parameter(torch.zeros(self.num_token_embeddings))

        if self.with_project:
            self.subspace = Subspace(hidden_dim=self.hidden_dim,
                                     subspace_dim=self.subspace_dim,
                                     trigger_subspaces=self.trigger_subspaces)

        if self.with_pool:
            self.pooler = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.Tanh()
            )

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        self.bce_loss = nn.BCELoss(reduction='none')
        self.nll_loss = nn.NLLLoss(ignore_index=self.padding_idx, reduction='none')
        self.contrastive_loss = SupConLoss(temperature=self.temperature)
        self._create_parameters()

        self.max_grad_norm = hparams.max_grad_norm
        if self.max_grad_norm is not None:
            self.grad_clip = self.max_grad_norm
        else:
            self.grad_clip = None
        self.weight_decay = hparams.weight_decay

        if self.use_gpu:
            self.cuda()

        return

    def _create_parameters(self):
        """ Create model's paramters. """
        sequence_mask = np.tri(self.num_pos_embeddings, self.num_pos_embeddings, dtype=self._dtype)
        self.sequence_mask = torch.tensor(sequence_mask)
        return

    def _get_score(self, inputs, data_file, score_ids):
        with torch.no_grad():
            if self.reader.dynamic_score:
                labels = [inputs['labels'][label_id] for label_id in inputs['label_ids']]
                if self.with_project:
                    scores = self.reader.build_score_multiple_matrix_on_the_fly(ids=inputs['ids'],
                                                                                labels=labels,
                                                                                data_file=data_file,
                                                                                score_ids=score_ids)
                else:
                    scores = [self.reader.build_score_matrix_on_the_fly(ids=inputs['ids'],
                                                                        labels=labels,
                                                                        data_file=data_file)]
                scores = [torch.from_numpy(score) for score in scores]
            else:
                ids = inputs['ids']
                scores = torch.zeros(len(ids), len(ids))
                score_matrix = torch.from_numpy(self.reader.score_matrixs[data_file])
                for i, id in enumerate(ids):
                    scores[i] = score_matrix[id, ids]
            scores = [score.to(inputs["src_token"].device) for score in scores]
        return scores

    def _create_mask(self, input_mask, append_head=False, auto_regressive=False):
        """
        Create attention mask.
        创建从序列形式到矩阵形式的mask：[batch_size, max_seq_len， 1] -> [batch_size, max_seq_len, max_seq_len]
        mask除了要考虑attention mask（自回归），还需要考虑pad的mask（自回归和双向）
        注：
        1. 一个句子中的非<pad>词看整个句子，该句中只有<pad>词才被mask
        2. 一个句子中的<pad>词看整个句子，该句的所有词都应该被mask

        @param : input_mask
        @type : Variable(shape: [batch_size, max_seq_len])

        @param : auto_regressive
        @type : bool
        """
        seq_len = input_mask.shape[1]

        input_mask = input_mask.float()
        mask1 = input_mask.unsqueeze(-1).repeat(1, 1, seq_len)
        mask2 = mask1.permute(0, 2, 1)
        mask = mask1 * mask2

        if append_head:
            # 拼接上句首位置([M]/z)的mask
            mask = torch.cat([mask[:, :1, :], mask], dim=1)
            mask = torch.cat([mask[:, :, :1], mask], dim=2)
            seq_len += 1

        if auto_regressive:
            # 将tgt端的<pad> mask和自回归attention mask融合
            seq_mask = self.sequence_mask[:seq_len, :seq_len]
            seq_mask = seq_mask.to(mask.device)
            mask = mask * seq_mask

        mask = 1 - mask
        return mask

    def _join_mask(self, mask1, mask2):
        """
        Merge source attention mask and target attention mask.
        合并后的整个mask矩阵可以分为四个部分：左上lu/右上ru/左下lb/右下rb

        @param : mask1 : source attention mask
        @type : Variable(shape: [batch_size, max_src_len, max_src_len])

        @param : mask1 : target attention mask
        @type : Variable(shape: [batch_size, max_tgt_len, max_tgt_len])
        """
        batch_size = mask1.shape[0]
        seq_len1 = mask1.shape[1]
        seq_len2 = mask2.shape[1]
        seq_len = seq_len1 + seq_len2

        mask_lu = mask1
        mask_ru = torch.ones(batch_size, seq_len1, seq_len2)
        if self.use_gpu:
            mask_ru = mask_ru.cuda()
        mask3 = mask2[:, :, :1].repeat(1, 1, seq_len1)
        mask4 = mask1[:, :1].repeat(1, seq_len2, 1)
        mask_lb = mask3 + mask4 - mask3 * mask4
        mask_rb = mask2
        mask_u = torch.cat([mask_lu, mask_ru], dim=2)
        mask_b = torch.cat([mask_lb, mask_rb], dim=2)
        mask = torch.cat([mask_u, mask_b], dim=1)
        return mask

    def _encoder_decoder_network(self, src_token, src_mask, tgt_token, tgt_mask,
                                 src_pos=None, src_type=None, src_turn=None,
                                 tgt_pos=None, tgt_type=None, tgt_turn=None):
        src_embed = self.embedder(src_token, src_pos, src_type, src_turn)
        tgt_embed = self.embedder(tgt_token, tgt_pos, tgt_type, tgt_turn)
        embed = torch.cat([src_embed, tgt_embed], dim=1)
        embed = self.embed_layer_norm(embed)

        enc_mask = self._create_mask(src_mask, auto_regressive=not self.bidirectional_context)
        dec_mask = self._create_mask(tgt_mask, auto_regressive=True)
        mask = self._join_mask(enc_mask, dec_mask)

        for layer in self.layers:
            embed = layer(embed, mask, None)

        tgt_len = tgt_token.shape[1]
        enc_embed = embed[:, :-tgt_len]
        dec_embed = embed[:, -tgt_len:]

        return enc_embed, dec_embed

    def _mask_encoder_network(self, input_token, input_mask, input_pos=None, input_type=None, input_turn=None):
        mask_embed = self.embedder.token_embedding.weight[self.reader.cls_id]
        mask_embed = mask_embed.unsqueeze(0).unsqueeze(0)
        mask_embed = mask_embed.repeat(input_token.shape[0], 1, 1)
        input_embed = self.embedder(input_token, input_pos, input_type, input_turn)
        embed = torch.cat([mask_embed, input_embed], dim=1)
        embed = self.embed_layer_norm(embed)

        mask = self._create_mask(input_mask, auto_regressive=not self.bidirectional_context, append_head=True)

        for layer in self.layers:
            embed = layer(embed, mask, None)

        latent_embed = embed[:, 0]
        enc_embed = embed[:, 1:]
        return latent_embed, enc_embed

    def _encoder_network(self, input_token, input_mask, input_pos=None, input_type=None, input_turn=None):
        embed = self.embedder(input_token, input_pos, input_type, input_turn)
        embed = self.embed_layer_norm(embed)
        mask = self._create_mask(input_mask, auto_regressive=not self.bidirectional_context)

        for layer in self.layers:
            embed = layer(embed, mask, None)

        return embed

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

        if self.with_contrastive:
            if self.with_cls:
                latent_embed, enc_embed = self._mask_encoder_network(input_token=aug(inputs['src_token']),
                                                                     input_mask=aug(inputs['src_mask']),
                                                                     input_pos=aug(inputs['src_pos']),
                                                                     input_type=aug(inputs['src_type']),
                                                                     input_turn=aug(inputs['src_turn']))
                features = latent_embed
            else:
                enc_embed, dec_embed = self._encoder_decoder_network(src_token=aug(inputs['src_token']),
                                                                     src_mask=aug(inputs['src_mask']),
                                                                     tgt_token=aug(inputs['tgt_token']),
                                                                     tgt_mask=aug(inputs['tgt_mask']),
                                                                     src_pos=aug(inputs['src_pos']),
                                                                     src_type=aug(inputs['src_type']),
                                                                     src_turn=aug(inputs['src_turn']))
                features = dec_embed[:, -1]

            if self.with_project:
                features = self.subspace(features)
            else:
                assert self.with_pool
                features = self.pooler(features.unsqueeze(1))

            batch_size = features.size(0) // 2
            features = torch.cat([features[:batch_size].unsqueeze(1), features[batch_size:].unsqueeze(1)], dim=1)
            features = F.normalize(features, dim=-1, p=2)
            outputs['features'] = features

        return outputs

    def _collect_metrics(self, inputs, outputs, with_label, data_file):

        metrics = {}
        loss = 0.

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

        if self.with_contrastive:
            con = 0.
            num_matrix = len(self.subspace.trigger_indices) if self.with_project else 1
            if with_label:
                scores = self._get_score(inputs=inputs, data_file=data_file, score_ids=self.subspace.trigger_indices)
                for i in range(num_matrix):
                    con += self.contrastive_loss(features=outputs["features"][:, :, i, :], mask=scores[i])
            else:
                for i in range(num_matrix):
                    con += self.contrastive_loss(features=outputs["features"][:, :, i, :])
            con /= num_matrix  # mean or sum
            metrics["con"] = con
            loss = loss + con
        else:
            con = None

        metrics["loss"] = loss
        if self.gpu > 1:
            return con, mlm, token_mlm, mlm_num
        else:
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

    def _infer(self, inputs):
        """ Real inference process of model. """
        results = {}

        latent_embed, enc_embed = self._mask_encoder_network(input_token=inputs['src_token'],
                                                             input_mask=inputs['src_mask'],
                                                             input_pos=inputs['src_pos'],
                                                             input_type=inputs['src_type'],
                                                             input_turn=inputs['src_turn'])
        features = latent_embed
        if self.with_project:
            features = self.subspace(latent_embed).squeeze(1)
            features = F.normalize(features, dim=-1, p=2)
        results['features'] = features
        results['ids'] = inputs['ids']

        return results


UnifiedTransformer.register("UnifiedTransformer")
