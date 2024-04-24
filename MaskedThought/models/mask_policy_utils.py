import random
import numpy as np
from numpy import random as np_rand
import torch
import torch.nn as nn
from transformers.modeling_outputs import (

    Seq2SeqLMOutput,
 CausalLMOutputWithPast
)


class MaskPolicy():

    def __init__(self, config, bpe_prefix):
        self.config = config
        self.bpe_prefix = bpe_prefix

    def get_gpt_masked_input(self, labels, target_mask, tokenizer, mask_for='none', types=None):
        self.mask_id = tokenizer.encode('<mask>', add_special_tokens=False)[0]
        mask_this = True
        def pr(ids):
            print(tokenizer.convert_ids_to_tokens(ids[0]))
        bs = labels.shape[0]
        mask_labels = labels.detach().clone()  # bs, seq
        mask_labels_np = mask_labels.cpu().numpy()
        tmp_tks = self.get_tks(mask_labels_np, tokenizer)

        non_zero_labels = ~(
                labels.data.eq(tokenizer.pad_token_id))  # 0 pad 2 eos -100 pad
        # the count of non-special token
        non_zero_sum_tensor = non_zero_labels.sum(-1)  # bs
        non_zero_sum = non_zero_sum_tensor.detach().cpu().numpy().tolist()

        masked_pos_shift = torch.zeros_like(mask_labels)  # bs, seq
        masked_pos_non_shift = torch.zeros_like(mask_labels)  # bs, seq
        labels_numpy = mask_labels_np
        labels_numpy = labels_numpy.tolist()

        if target_mask is not None:
            target_mask_np = target_mask.cpu().numpy()
        if types is not None:
            assert bs == len(types)
        for i in range(bs):
            cand_pos = []
            all_pos = []
            k = 0
            mask_rate = self.config.mask_rate
            while k < len(target_mask_np[i]):
                if self.config.not_mask_tgt:
                    if int(target_mask_np[i][k]) == 1:
                        k += 1
                        continue
                if self.config.not_mask_source:
                    if int(target_mask_np[i][k]) == 0:
                        k += 1
                        continue
                if mask_rate == 1:
                    cand_pos.append(k)
                    k += 1
                    continue
                all_pos.append(k)

                cand_pos.append(k)
                k += 1

            sample_num = 0
            for _ in range(len(cand_pos)):
                if random.random() < mask_rate:
                    sample_num += 1
            if mask_rate == 1:
                sample_pos = cand_pos
            else:
                sample_pos = np_rand.choice(a=np.array(cand_pos), size=sample_num, replace=False).tolist()
            sample_pos = sorted(sample_pos)
            non_sample_pos = set(cand_pos) - set(sample_pos)
            non_sample_pos = sorted(list(non_sample_pos))

            for idx, j in enumerate(sample_pos):

                if self.config.mask_input and mask_this:
                    if self.config.replace:
                        r = random.random()
                        if r < self.config.replace_rate:
                            tk_id = random.randint(0, int(tokenizer.vocab_size)-1)
                            mask_labels[i][j] = tk_id
                            mask_labels[i][j] = mask_labels[i][j]
                        else:
                            mask_labels[i][j] = self.mask_id
                    else:
                        mask_labels[i][j] = self.mask_id

                masked_pos_shift[i][idx] = j + 1
                masked_pos_non_shift[i][idx] = j
        return mask_labels, masked_pos_shift, masked_pos_non_shift

    def get_tks(self, y, tokenizer):
        pre_output_ids = y.tolist()
        output_ids = []
        for i in range(len(pre_output_ids)):
            output_id = []
            for j in range(0, len(pre_output_ids[i])):
                if pre_output_ids[i][j] == -100:
                    break
                output_id.append(pre_output_ids[i][j])
            output_ids.append(output_id)
        tks = [
            tokenizer.convert_ids_to_tokens(output_ids[i]) for i in
            range(len(output_ids))]
        return tks

    def construct_gpt_return(self, lm_logits, labels, bs, masked_pos_shift, masked_pos_non_shift, outputs,
                         input_ids,  mask_labels, tokenizer, ce=False, return_dict=True, loss=None):
        device = lm_logits.device
        if masked_pos_non_shift is not None:
            probs = torch.softmax(lm_logits, dim=-1)
            log_probs = torch.log(probs)
            log_probs_all = log_probs.detach().clone()
            seq_len = labels.shape[1]
            mp = torch.zeros((bs, seq_len + 2)).to(device).scatter_(1, masked_pos_shift.to(device),
                                                                torch.ones((bs, seq_len + 1)).to(device))
            mp = mp[:, 2:]
            mp_long = mp.long()
            ori_mp = mp.clone()
            pads = torch.ones_like(labels, dtype=torch.long).to(device) * tokenizer.pad_token_id
            other_2_pads_labels = labels * ~(labels.data.eq(-100) | labels.data.eq(tokenizer.eos_token_id)) + pads * (
                        labels.data.eq(-100) | labels.data.eq(tokenizer.eos_token_id))  # -100 -> 0#
            _, max_ids = torch.max(log_probs, dim=-1)
            y_b = other_2_pads_labels * (1 - mp_long) + max_ids * mp_long
            _, s2, s3 = probs.shape
            gt_prob = torch.gather(probs.reshape(-1, s3), dim=1, index=other_2_pads_labels.reshape(-1, 1))
            gt_prob = gt_prob.reshape(-1, s2)
            gt_log_probs = torch.log(gt_prob)
            masked_ids = max_ids

        if not return_dict:
            raise NotImplementedError("not return_dict not implemented yet.")

        output = CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        if labels is None:
            return output
        else:
            return output, y_b, None, max_ids, masked_ids, input_ids, labels, log_probs, log_probs_all, \
                mask_labels.to(input_ids.device), masked_pos_shift.to(input_ids.device), \
                masked_pos_non_shift.to(input_ids.device), gt_log_probs

    def split_gpt_return(self, lm_logits, input_ids, labels,masked_pos_shift,
                     masked_pos_non_shift, bs, return_dict, outputs, mask_labels, tokenizer, loss):
        res= self.construct_gpt_return(lm_logits=lm_logits, labels=labels, bs=bs,
                                    masked_pos_shift=masked_pos_shift, masked_pos_non_shift=masked_pos_non_shift, return_dict=return_dict,
                                                outputs=outputs, input_ids=input_ids,
                                                 mask_labels=mask_labels, tokenizer=tokenizer, loss=loss)
        return res


    def format_and_padding(self, raw_src_txt, raw_tgt_txt, device, max_len=1024, pad_front=True,
                        format=False, tgt_max_len=256, cut_front=False,  instruct_type='default', cut_src=False, tokenizer=None):
        raw_src_txt = ["".join(input_text) for input_text in raw_src_txt]
        def process_format(input_text, instruct_type='default'):
            if not format:
                return input_text
            if instruct_type == 'metamath':
                def get_input(query):
                    if query.find('\n') == -1:
                        return ''
                    return '\n'.join(query.split('\n')[1:])
                PROMPT_DICT = {
                    "prompt_input": (
                        "Below is an instruction that describes a task, paired with an input that provides further context. "
                        "Write a response that appropriately completes the request.\n\n"
                        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
                    ),
                    "prompt_no_input": (
                        "Below is an instruction that describes a task. "
                        "Write a response that appropriately completes the request.\n\n"
                        "### Instruction:\n{instruction}\n\n### Response:"
                    ),
                }
                example = {'instruction': input_text.split('\n')[0], 'input': get_input(input_text)}
                prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
                input = prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(
                    example)
                return input
            if instruct_type == 'wizardlm':
                problem_prompt = (
                    "{instruction}\n\n### Response:"
                )
                input = problem_prompt.format(instruction=input_text)
                return input

            PROMPT_DICT = {
                "prompt_input": (
                    "Below is an instruction that describes a task, paired with an input that provides further context. "
                    "Write a response that appropriately completes the request.\n\n"
                    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
                ),
                "prompt_no_input": (
                    "Below is an instruction that describes a task. "
                    "Write a response that appropriately completes the request.\n\n"
                    "### Instruction:\n{instruction}\n\n### Response:"
                ),
            }
            input_text = PROMPT_DICT['prompt_no_input'].format(
                instruction=input_text)
            return input_text
        raw_src_txt = [process_format(e, instruct_type=instruct_type) for e in raw_src_txt]

        if cut_src:
            ori = tokenizer.truncation_side
            tokenizer.truncation_side = "left"
            model_inputs = tokenizer(
                raw_src_txt,
                max_length=max_len - tgt_max_len,
                truncation=True,
            )
            tokenizer.truncation_side = ori
            raw_src_txt  = tokenizer.batch_decode(model_inputs['input_ids'], skip_special_tokens=True)

        raw_tgt_txt = [input_text  for input_text in raw_tgt_txt]
        input_txt = [txt + ' ' + raw_tgt_txt[i] for i, txt in enumerate(raw_src_txt)]
        input_txt_ids = [tokenizer.encode(txt, truncation=False) for txt in input_txt]
        label_txt_ids = [tokenizer.encode(txt, truncation=False) for txt in raw_tgt_txt]

        input_txt_ids = [ids + [tokenizer.eos_token_id] for ids in input_txt_ids]
        label_txt_ids = [ids + [tokenizer.eos_token_id] for ids in label_txt_ids]

        label_txt_ids = [label_txt_id[1:] for label_txt_id in label_txt_ids]
        label_masks = []
        attention_masks = []
        for i, input_txt_id in enumerate(input_txt_ids):
            seq_len = len(input_txt_id)
            label_txt_id = label_txt_ids[i]
            label_len = len(label_txt_id)
            label_mask = [0 if i < seq_len - label_len else 1 for i in range(seq_len)]
            if np.sum(label_mask) == 0:
                print(input_txt_id)
                print(label_txt_id)
                print(seq_len, label_len)
                assert 1==0
            label_masks.append(label_mask)
            attention_mask = [1 for i in range(seq_len)]
            attention_masks.append(attention_mask)
        def add_padding(txts_ids, pad_id, max_len):
            padding_txts_ids = []
            batch_max_seq_len = max([len(txt) for txt in txts_ids])
            batch_max_seq_len = min(batch_max_seq_len, max_len)
            for txt_ids in txts_ids:
                if cut_front:
                    txt_ids = txt_ids[-batch_max_seq_len:]
                else:
                    txt_ids = txt_ids[:batch_max_seq_len]
                if pad_front:
                    padding_txts_ids.append(
                        [pad_id] * (batch_max_seq_len - len(txt_ids)) + txt_ids)
                else:
                    padding_txts_ids.append(
                         txt_ids + [pad_id] * (batch_max_seq_len - len(txt_ids)) )
            return padding_txts_ids
        padding_txts_ids = add_padding(input_txt_ids, pad_id=tokenizer.pad_token_id, max_len=max_len)
        padding_label_txt_ids = add_padding(label_txt_ids, pad_id=-100, max_len=max_len)
        padding_attention_mask = add_padding(attention_masks, pad_id=0, max_len=max_len)
        padding_label_mask = add_padding(label_masks, pad_id=0, max_len=max_len)

        return torch.tensor(padding_txts_ids, dtype=torch.long).to(device),\
            torch.tensor(padding_label_txt_ids, dtype=torch.long).to(device), \
            torch.tensor(padding_attention_mask, dtype=torch.long).to(device),  \
            torch.tensor(padding_label_mask, dtype=torch.long).to(device),