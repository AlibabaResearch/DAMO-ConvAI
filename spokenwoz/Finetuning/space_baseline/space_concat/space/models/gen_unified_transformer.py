"""
GenUnifiedTransformer
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from space.args import str2bool
from space.models.unified_transformer import UnifiedTransformer
from space.utils.criterions import compute_kl_loss


class GenUnifiedTransformer(UnifiedTransformer):
    """
    Implement generation unified transformer.
    """

    @classmethod
    def add_cmdline_argument(cls, group):
        """ Add cmdline argument. """
        UnifiedTransformer.add_cmdline_argument(group)
        return group

    def __init__(self, hparams, reader, generator):
        super(GenUnifiedTransformer, self).__init__(hparams, reader, generator)
        self.understand = hparams.understand

        if self.use_gpu:
            self.cuda()
        return

    def _forward(self, inputs, audios, transcripts, tokenizer, is_training, with_label):
        """ Real forward process of model in different mode(train/test). """

        def cat(x, y, dim=1):
            return torch.cat([x, y], dim=dim)

        outputs = {}

        if self.understand or self.policy:
            if self.understand:
                prompt_token = inputs['understand_token']
                prompt_mask = inputs['understand_mask']
                if self.policy:
                    prompt_token = cat(prompt_token, inputs['policy_token'])
                    prompt_mask = cat(prompt_mask, inputs['policy_mask'])
            else:
                prompt_token = inputs['policy_token']
                prompt_mask = inputs['policy_mask']

            enc_embed, dec_embed, prompt_embed = self._encoder_prompt_decoder_network(
                src_token=inputs['src_token'],
                src_mask=inputs['src_mask'],
                tgt_token=inputs['tgt_token'][:, :-1],
                tgt_mask=inputs['tgt_mask'][:, :-1],
                prompt_token=prompt_token,
                prompt_mask=prompt_mask,
                src_pos=inputs['src_pos'],
                src_type=inputs['src_type'],
                src_turn=inputs['src_turn'],
                tgt_pos=inputs['tgt_pos'][:, :-1],
                tgt_type=inputs['tgt_type'][:, :-1],
                tgt_turn=inputs['tgt_turn'][:, :-1]
            )
        else:
            # dec_embed = self._multimodal_fusion(
            #     src_token=inputs['src_token'],
            #     src_mask=inputs['src_mask'],
            #     tgt_token=inputs['tgt_token'][:, :-1],
            #     tgt_mask=inputs['tgt_mask'][:, :-1],
            #     aud_token=audios['input_values'], 
            #     aud_mask=audios['attention_mask'],
            #     src_pos=inputs['src_pos'],
            #     src_type=inputs['src_type'],
            #     src_turn=inputs['src_turn'],
            #     tgt_pos=inputs['tgt_pos'][:, :-1],
            #     tgt_type=inputs['tgt_type'][:, :-1],
            #     tgt_turn=inputs['tgt_turn'][:, :-1]
            #         )

            enc_embed, dec_embed = self._encoder_decoder_network(
                src_token=inputs['src_token'],
                src_mask=inputs['src_mask'],
                tgt_token=inputs['tgt_token'][:, :-1],
                tgt_mask=inputs['tgt_mask'][:, :-1],
                aud_token=audios['input_values'], 
                aud_mask=audios['attention_mask'],
                src_pos=inputs['src_pos'],
                src_type=inputs['src_type'],
                src_turn=inputs['src_turn'],
                tgt_pos=inputs['tgt_pos'][:, :-1],
                tgt_type=inputs['tgt_type'][:, :-1],
                tgt_turn=inputs['tgt_turn'][:, :-1],
                tokenizer=tokenizer,
                transcripts=transcripts
            )

        outputs["dec_probs"] = self._dec_head(dec_embed=dec_embed)
        return outputs

    def _collect_metrics(self, inputs, outputs, with_label, data_file):

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

        metrics["loss"] = loss
        if self.gpu > 1:
            return nll, token_nll, token_num
        else:
            return metrics

    def _optimize(self, loss, do_update=False, optimizer=None):
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

    def _init_state(self, src_token, src_mask, aud_token=None, aud_mask=None, src_pos=None, src_type=None, src_turn=None, transcripts=None, tokenizer=None):
        """ Initialize decode state. """
        state = {}
        batch_size = src_token.shape[0]

        if self.use_gpu:
            aud_token, aud_mask = aud_token.cuda(), aud_mask.cuda()
        enc_audio_embed, aud_mask = self.audio_encoder(aud_token, aud_mask)
        cur_start = [np.argwhere(sent.cpu().numpy()==13)[-1] for sent in src_token]
        cur_end = [np.argwhere(sent.cpu().numpy()==7)[-1] for sent in src_token]
        
        src_embed = self.embedder(src_token, src_pos, src_type, src_turn)

        # print(enc_audio_embed.shape, src_embed.shape)
        
        fused_emb = self._fuse(enc_audio_embed[0], aud_mask[0], src_embed[0, int(cur_start[0])+1:int(cur_end[0])], transcripts[0], tokenizer, self.audio_encoder.config)
        src_embed = torch.cat((src_embed[0, :int(cur_end[0])], fused_emb,
                                   src_embed[0, int(cur_end[0]):sum(src_mask[0])]), dim=0).unsqueeze(0)
        # src_embed = torch.cat((src_embed[0, :int(cur_start[0])], fused_emb,
        #                            src_embed[0, int(cur_end[0]):sum(src_mask[0])]), dim=0).unsqueeze(0)
        src_mask = torch.ones(1, src_embed.shape[1]).cuda()
        src_embed = self.embed_layer_norm(src_embed)
        mask = self._create_mask(src_mask, append_head=False)

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

    def _init_prompt_state(self, src_token, src_mask, prompt_token, prompt_mask,
                           src_pos=None, src_type=None, src_turn=None,
                           prompt_pos=None, prompt_type=None, prompt_turn=None):
        """ Initialize decode state. """
        state = {}
        batch_size = src_token.shape[0]

        src_embed = self.embedder(src_token, src_pos, src_type, src_turn)
        prompt_embed = self.embedder(prompt_token, prompt_pos, prompt_type, prompt_turn)
        embed = torch.cat([src_embed, prompt_embed], dim=1)
        embed = self.embed_layer_norm(embed)
        enc_out = embed

        enc_mask = self._create_mask(src_mask, auto_regressive=False)
        dec_mask = self._create_mask(prompt_mask, auto_regressive=True)
        mask = self._join_mask(enc_mask, dec_mask)

        cache = {}
        for l, layer in enumerate(self.layers):
            cache[f"layer_{l}"] = {}
            enc_out = layer(enc_out, mask, cache[f"layer_{l}"])

        state["cache"] = cache
        state["mask"] = mask[:, -1:]  # state["mask"] = mask[:, :1]
        state["batch_size"] = batch_size
        shape = [batch_size, 1, 1]
        state['txt_out'] = enc_out
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
        mask = state["mask"].cuda()

        # shape: [batch_size, 1, 1]
        # txt_out = state["txt_out"]
        pred_token = state["pred_token"].cuda()
        pred_mask = state["pred_mask"].cuda()
        pred_pos = state["pred_pos"].cuda()
        pred_type = state["pred_type"].cuda()
        pred_turn = state["pred_turn"].cuda()

        # list of shape(len: num_layers): [batch_size, seq_len, hidden_dim]
        cache = state["cache"]

        pred_embed = self.embedder(pred_token, pred_pos, pred_type, pred_turn).squeeze(-2).cuda()
        pred_embed = self.embed_layer_norm(pred_embed.cuda()).cuda()

        # shape: [batch_size, 1, seq_len + 1]
        mask = torch.cat([mask.cuda(), 1 - pred_mask.cuda()], dim=2).cuda()

        # shape: [batch_size, 1, hidden_dim]
        for l, layer in enumerate(self.layers):
            pred_embed = layer(pred_embed.cuda(), mask.cuda(), cache[f"layer_{l}"]).cuda()
        # pred_embed = self._multimodal_infer(src_token, txt_out, None, pred_embed)
        # shape: [batch_size, vocab_size]
        pred_probs = self._dec_head(dec_embed=pred_embed[:, 0].cuda()).cuda()
        pred_logits = torch.log(pred_probs.cuda()).cuda()

        state["mask"] = mask
        return pred_logits, state

    def _infer(self, inputs, audios, transcripts, tokenizer, start_id=None, eos_id=None, max_gen_len=None, prev_input=None):
        """ Real inference process of model. """

        def cat(x, y, dim=1):
            return torch.cat([x, y], dim=dim)

        # Initial decode state.
        if self.understand or self.policy:
            if self.understand:
                prompt_token = inputs['understand_token']
                prompt_mask = inputs['understand_mask']
                if self.policy:
                    prompt_token = cat(prompt_token, inputs['policy_token'])
                    prompt_mask = cat(prompt_mask, inputs['policy_mask'])
            else:
                prompt_token = inputs['policy_token']
                prompt_mask = inputs['policy_mask']

            state = self._init_prompt_state(src_token=inputs['src_token'],
                                            src_mask=inputs['src_mask'],
                                            prompt_token=prompt_token,
                                            prompt_mask=prompt_mask,
                                            src_pos=inputs['src_pos'],
                                            src_type=inputs['src_type'],
                                            src_turn=inputs['src_turn'])
        else:
            state = self._init_state(src_token=inputs['src_token'],
                                     src_mask=inputs['src_mask'],
                                     aud_token=audios['input_values'],
                                     aud_mask=audios['attention_mask'],
                                     src_pos=inputs['src_pos'],
                                     src_type=inputs['src_type'],
                                     src_turn=inputs['src_turn'],
                                     transcripts=transcripts, 
                                     tokenizer=tokenizer)

        # Generation process.
        gen_results = self.generator(step_fn=self._decode,
                                     state=state,
                                     start_id=start_id,
                                     eos_id=eos_id,
                                     max_gen_len=max_gen_len,
                                     prev_input=prev_input)

        outputs = gen_results['preds']
        return outputs


GenUnifiedTransformer.register("GenUnifiedTransformer")
