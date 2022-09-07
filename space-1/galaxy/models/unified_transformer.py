"""
UnifiedTransformer
"""
import numpy as np
import torch
import torch.nn as nn

from galaxy.args import str2bool
from galaxy.modules.embedder import Embedder
from galaxy.models.model_base import ModelBase
from galaxy.modules.transformer_block import TransformerBlock
from galaxy.utils.eval import DAEvaluation


class UnifiedTransformer(ModelBase):
    """
    Implement unified transformer for generation.
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
        group.add_argument("--num_turn_embeddings", type=int, default=16,
                           help="The maximum number of turn.")
        group.add_argument("--num_act", type=int, default=20,
                           help="The number of unified DAs.")
        group.add_argument("--num_heads", type=int, default=12,
                           help="The number of heads in multi head attention.")
        group.add_argument("--num_layers", type=int, default=12,
                           help="The number of layers in transformer.")
        group.add_argument("--hidden_dim", type=int, default=768,
                           help="The size of hidden vector in transformer.")
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
        group.add_argument("--use_discriminator", type=str2bool, default=False,
                           help="Whether to use discriminator loss.")
        group.add_argument("--dis_ratio", type=float, default=1.0,
                           help="The ratio of discriminator loss.")
        group.add_argument("--bce_ratio", type=float, default=1.0,
                           help="The ratio of binary cross entropy loss.")
        group.add_argument("--pos_trainable", type=str2bool, default=True,
                           help="Whether to train position embeddings.")
        group.add_argument("--with_joint_act", type=str2bool, default=False,
                           help="Whether to use joint DA training or inference.")
        group.add_argument("--with_rdrop_act", type=str2bool, default=False,
                           help="Whether to use R-Drop for DA during generation.")
        group.add_argument("--initializer_range", type=float, default=0.02,
                           help="Use to initialize parameters.")
        group.add_argument("--lr", type=float, default=1e-4,
                           help="The inital learning rate for Adam.")
        group.add_argument("--weight_decay", type=float, default=0.0,
                           help="The weight decay for Adam.")
        group.add_argument("--gradient_accumulation_steps", type=int, default=1,
                           help="The number of gradient accumulation steps for one update.")
        group.add_argument("--warmup_steps", type=int, default=-1,
                           help="The number of warmup steps for lr.")
        group.add_argument("--max_grad_norm", type=float, default=5.0,
                           help="The maximum norm of gradient.")
        return group

    def __init__(self, hparams, generator, dtype="float32"):
        super(UnifiedTransformer, self).__init__(hparams)
        self.generator = generator
        self.num_token_embeddings = hparams.num_token_embeddings
        self.num_pos_embeddings = hparams.num_pos_embeddings
        self.num_type_embeddings = hparams.num_type_embeddings
        self.num_turn_embeddings = hparams.num_turn_embeddings
        self.num_act = hparams.num_act
        self.hidden_dim = hparams.hidden_dim
        self.num_heads = hparams.num_heads
        self.num_layers = hparams.num_layers
        self.padding_idx = hparams.padding_idx
        self.dropout = hparams.dropout
        self.embed_dropout = hparams.embed_dropout
        self.attn_dropout = hparams.attn_dropout
        self.ff_dropout = hparams.ff_dropout
        self.pos_trainable = hparams.pos_trainable
        self.initializer_range = hparams.initializer_range
        self.use_discriminator = hparams.use_discriminator
        self.gradient_accumulation_steps = hparams.gradient_accumulation_steps
        self.with_joint_act = hparams.with_joint_act
        self.with_rdrop_act = hparams.with_rdrop_act
        self.token_loss = hparams.token_loss
        self.dis_ratio = hparams.dis_ratio
        self.bce_ratio = hparams.bce_ratio
        self.max_len = hparams.max_len
        self.gpu = hparams.gpu
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

        if self.with_joint_act:
            self.act_classifier = nn.Sequential(
                    nn.Linear(self.hidden_dim, self.num_act),
                    nn.Sigmoid()
                )

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        self._create_parameters()

        self.nll_loss = nn.NLLLoss(ignore_index=self.padding_idx, reduction='none')
        self.bce_loss = nn.BCELoss()

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
        """ Create model's extra parameters. """
        if self.with_joint_act:
            self.mask_embed = nn.Parameter(torch.Tensor(1, 1, self.hidden_dim))
            nn.init.normal_(self.mask_embed, std=self.initializer_range)

        sequence_mask = np.tri(self.num_pos_embeddings, self.num_pos_embeddings, dtype=self._dtype)
        self.sequence_mask = torch.tensor(sequence_mask)
        return

    def _create_mask(self, input_mask, append_head=False, auto_regressive=False):
        """
        Create attention mask.

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
            mask = torch.cat([mask[:, :1, :], mask], dim=1)
            mask = torch.cat([mask[:, :, :1], mask], dim=2)
            seq_len += 1

        if auto_regressive:
            seq_mask = self.sequence_mask[:seq_len, :seq_len]
            seq_mask = seq_mask.to(mask.device)
            mask = mask * seq_mask

        mask = 1 - mask
        return mask

    def _join_mask(self, mask1, mask2):
        """
        Merge source attention mask and target attention mask.

        @param : mask1 : source attention mask
        @type : Variable(shape: [batch_size, max_src_len, max_src_len])

        @param : mask2 : target attention mask
        @type : Variable(shape: [batch_size, max_tgt_len, max_tgt_len])
        """
        batch_size = mask1.shape[0]
        seq_len1 = mask1.shape[1]
        seq_len2 = mask2.shape[1]
        seq_len = seq_len1 + seq_len2

        mask_lu = mask1
        mask_ru = torch.ones(batch_size, seq_len1, seq_len2)
        mask_ru = mask_ru.to(mask_lu.device)
        mask3 = mask2[:, :, :1].repeat(1, 1, seq_len1)
        mask4 = mask1[:, :1].repeat(1, seq_len2, 1)
        mask_lb = mask3 + mask4 - mask3 * mask4
        mask_rb = mask2
        mask_u = torch.cat([mask_lu, mask_ru], dim=2)
        mask_b = torch.cat([mask_lb, mask_rb], dim=2)
        mask = torch.cat([mask_u, mask_b], dim=1)
        return mask

    def _dec_head(self, dec_embed):
        """ Decoding head for response generation task. """
        dec_logits = torch.matmul(dec_embed, self.embedder.token_embedding.weight.T)
        dec_probs = self.softmax(dec_logits)
        return dec_probs

    def _encoder_decoder_network(self, src_token, src_mask, tgt_token, tgt_mask,
                                 src_pos=None, src_type=None, src_turn=None,
                                 tgt_pos=None, tgt_type=None, tgt_turn=None):
        """ Unified encoder-decoder network for both understanding and generation. """

        src_embed = self.embedder(src_token, src_pos, src_type, src_turn)
        tgt_embed = self.embedder(tgt_token, tgt_pos, tgt_type, tgt_turn)
        embed = torch.cat([src_embed, tgt_embed], dim=1)
        embed = self.embed_layer_norm(embed)

        enc_mask = self._create_mask(src_mask, auto_regressive=False)
        dec_mask = self._create_mask(tgt_mask, auto_regressive=True)
        mask = self._join_mask(enc_mask, dec_mask)

        for layer in self.layers:
            embed = layer(embed, mask, None)

        tgt_len = tgt_token.shape[1]
        enc_embed = embed[:, :-tgt_len]
        dec_embed = embed[:, -tgt_len:]

        return enc_embed, dec_embed

    def _mask_encoder_decoder_network(self, src_token, src_mask, tgt_token, tgt_mask,
                                     src_pos=None, src_type=None, src_turn=None,
                                     tgt_pos=None, tgt_type=None, tgt_turn=None):
        """ Unified mask-encoder-decoder network for both understanding and generation. """

        mask_embed = self.mask_embed.repeat(src_token.shape[0], 1, 1)
        mask_embed = self.embed_layer_norm(mask_embed)
        src_embed = self.embedder(src_token, src_pos, src_type, src_turn)
        tgt_embed = self.embedder(tgt_token, tgt_pos, tgt_type, tgt_turn)
        embed = torch.cat([src_embed, tgt_embed], dim=1)
        embed = self.embed_layer_norm(embed)
        embed = torch.cat([mask_embed, embed], dim=1)

        enc_mask = self._create_mask(src_mask, auto_regressive=False, append_head=True)
        dec_mask = self._create_mask(tgt_mask, auto_regressive=True, append_head=False)
        mask = self._join_mask(enc_mask, dec_mask)

        for layer in self.layers:
            embed = layer(embed, mask, None)

        tgt_len = tgt_token.shape[1]
        latent_embed = embed[:, 0]
        enc_embed = embed[:, 1: -tgt_len]
        dec_embed = embed[:, -tgt_len:]

        return latent_embed, enc_embed, dec_embed

    def _forward(self, inputs, is_training):
        """ Real forward process of model. """

        outputs = {}

        if self.with_joint_act:
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
            joint_act_probs = self.act_classifier(latent_embed)
            outputs["joint_act_probs"] = joint_act_probs
        else:
            enc_embed, dec_embed = self._encoder_decoder_network(
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

        outputs["dec_probs"] = self._dec_head(dec_embed=dec_embed)
        return outputs

    def _collect_metrics(self, inputs, outputs):
        """ Collect metrics for optimization and log. """

        metrics = {}
        loss = 0.

        label = inputs["tgt_token"][:, 1:]
        token_num = torch.sum(torch.sum(inputs["tgt_mask"], dim=1) - 1)
        nll = self.nll_loss(torch.log(outputs["dec_probs"] + 1e-12).permute(0, 2, 1), label)
        nll = torch.sum(nll, dim=1)
        token_nll = torch.sum(nll) / token_num
        nll = torch.mean(nll)
        metrics["nll"] = nll
        metrics["token_nll"] = token_nll
        metrics["token_num"] = token_num
        loss = loss + (token_nll if self.token_loss else nll)

        if self.with_joint_act:
            bce = self.bce_loss(outputs["joint_act_probs"], inputs['act_index'].float())
            metrics['bce'] = bce
            loss = loss + bce * self.bce_ratio

            da_preds = (outputs["joint_act_probs"] > 0.5).int().detach().cpu().numpy()
            da_labels = inputs['act_index'].detach().cpu().numpy()
            da_result = DAEvaluation(preds=da_preds, labels=da_labels)
            bce_f1 = torch.tensor(da_result['f1_micro'], device=loss.device)
            metrics['bce_f1'] = bce_f1
        else:
            bce, bce_f1 = None, None

        metrics["loss"] = loss

        if self.gpu > 1:
            return nll, token_nll, token_num, bce, bce_f1
        else:
            return metrics

    def _optimize(self, loss, do_update, optimizer=None):
        """ Optimize loss function and update model. """

        assert optimizer is not None

        if self.gradient_accumulation_steps > 1:
            loss = loss / self.gradient_accumulation_steps

        loss.backward()

        if self.grad_clip is not None and self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(parameters=self.parameters(), max_norm=self.grad_clip)

        if do_update:
            optimizer.step()
            optimizer.zero_grad()

        return

    def _init_state(self, src_token, src_mask, src_pos=None, src_type=None, src_turn=None):
        """ Initialize decode state. """

        state = {}
        batch_size = src_token.shape[0]

        src_embed = self.embedder(src_token, src_pos, src_type, src_turn)
        src_embed = self.embed_layer_norm(src_embed)

        mask = self._create_mask(src_mask, append_head=self.with_joint_act)

        if self.with_joint_act:
            mask_embed = self.mask_embed.repeat(batch_size, 1, 1)
            mask_embed = self.embed_layer_norm(mask_embed)
            enc_out = torch.cat([mask_embed, src_embed], dim=1)
        else:
            enc_out = src_embed

        cache = {}
        for l, layer in enumerate(self.layers):
            cache[f"layer_{l}"] = {}
            enc_out = layer(enc_out, mask, cache[f"layer_{l}"])

        state["cache"] = cache
        state["mask"] = mask[:, :1]
        state["batch_size"] = batch_size
        shape = [batch_size, 1, 1]
        state["pred_mask"] = torch.ones(shape, dtype=torch.float32)
        state["pred_pos"] = torch.zeros(shape, dtype=torch.int64)
        state["pred_type"] = torch.zeros(shape, dtype=torch.int64)
        state["pred_turn"] = torch.zeros(shape, dtype=torch.int64)
        if self.use_gpu:
            state["pred_mask"] = state["pred_mask"].cuda()
            state["pred_pos"] = state["pred_pos"].cuda()
            state["pred_type"] = state["pred_type"].cuda()
            state["pred_turn"] = state["pred_turn"].cuda()

        return state

    def _decode(self, state):
        """ Decoding one time stamp. """

        # shape: [batch_size, 1, seq_len]
        mask = state["mask"]

        # shape: [batch_size, 1, 1]
        pred_token = state["pred_token"]
        pred_mask = state["pred_mask"]
        pred_pos = state["pred_pos"]
        pred_type = state["pred_type"]
        pred_turn = state["pred_turn"]

        # list of shape(len: num_layers): [batch_size, seq_len, hidden_dim]
        cache = state["cache"]

        pred_embed = self.embedder(pred_token, pred_pos, pred_type, pred_turn).squeeze(-2)
        pred_embed = self.embed_layer_norm(pred_embed)

        # shape: [batch_size, 1, seq_len + 1]
        mask = torch.cat([mask, 1 - pred_mask], dim=2)

        # shape: [batch_size, 1, hidden_dim]
        for l, layer in enumerate(self.layers):
            pred_embed = layer(pred_embed, mask, cache[f"layer_{l}"])

        # shape: [batch_size, vocab_size]
        pred_probs = self._dec_head(dec_embed=pred_embed[:, 0])
        pred_logits = torch.log(pred_probs)

        state["mask"] = mask
        return pred_logits, state

    def _infer(self, inputs, start_id=None, eos_id=None, max_gen_len=None, prev_input=None):
        """ Real inference process of model. """

        # Initial decode state.
        state = self._init_state(src_token=inputs['src_token'],
                                 src_mask=inputs['src_mask'],
                                 src_pos=inputs['src_pos'],
                                 src_type=inputs['src_type'],
                                 src_turn=inputs['src_turn'])

        # Generation process.
        gen_results = self.generator(step_fn=self._decode,
                                     state=state,
                                     start_id=start_id,
                                     eos_id=eos_id,
                                     max_gen_len=max_gen_len,
                                     prev_input=prev_input)

        outputs = gen_results['preds']
        return outputs


UnifiedTransformer.register("UnifiedTransformer")
