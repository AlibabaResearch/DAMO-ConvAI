import json
import os
import torch
import pickle
import random
import numpy as np
from torch.utils.data import Dataset


def pad(sequence, length, pad_token=0):
    seq_len = sequence.shape[0]
    if length > seq_len:
        padding = torch.ones(length - seq_len, dtype=sequence.dtype) * pad_token
        att = torch.cat([torch.ones_like(sequence), padding])
        sequence = torch.cat([sequence, padding])
    else:
        if sequence.dtype == torch.long:
            sequence = torch.cat([sequence[:1], sequence[1 - length:]])
        else:
            sequence = sequence[:length]
        att = torch.ones_like(sequence)
    return sequence, att


def compute_valid(transcript, offset, length):
    sv = [0 for _ in range(length)]
    ev = [0 for _ in range(length)]
    start_labels, end_labels = [], []
    for i, item in enumerate(transcript):
        sv[offset + item[-4]] = 1
        ev[offset + item[-3] - 1] = 1
        start_labels.append(float(f"{item[-2] / 160000:.3f}"))
        end_labels.append(float(f"{item[-1] / 160000:.3f}"))
    return torch.BoolTensor(sv), torch.BoolTensor(ev), start_labels, end_labels


class PretrainDataset(Dataset):
    def __init__(self, datas, num_turns, prefix):
        self.datas = datas
        self.n = len(datas)
        self.prefix = prefix
        self.num_turns = num_turns
        self.has_positive = [i for i, d in enumerate(datas) if d[-1] >= 0]

    def __len__(self):
        return len(self.has_positive)

    def __getitem__(self, idx):
        anchor_idx = self.has_positive[idx]  # 0轮
        prev_idx = self.datas[anchor_idx][-1]  # -1轮
        negative_idx_audio = random.randint(0, self.n - 3)
        if negative_idx_audio >= anchor_idx:
            negative_idx_audio += 2
        negative_idx_text = random.randint(0, self.n - 3)
        if negative_idx_text >= anchor_idx:
            negative_idx_text += 2
        history = []  # <-2轮
        curr_idx = prev_idx
        for i in range(2, self.num_turns):
            if self.datas[curr_idx][-1] == -1:
                break
            curr_idx = self.datas[curr_idx][-1]
            history = self.datas[curr_idx][1][1:] + history
        af, aw = self.datas[anchor_idx][:2]
        at = self.datas[anchor_idx][2:-1]
        pf, pw = self.datas[prev_idx][:2]
        pt = self.datas[prev_idx][2:-1]
        nf = self.datas[negative_idx_audio][0]
        nw = self.datas[negative_idx_text][1]
        af, pf, nf = map(lambda x: os.path.join(self.prefix, x), [af, pf, nf])
        return np.load(pf), pw, pt, np.load(af), aw, at, np.load(nf), nw, [0] + history


class DownstreamDataset(Dataset):
    def __init__(self, root, task, op, audio_multi_turn=False):
        if task == "iemocap":
            with open(f"{root}/{task}/{op}.pkl", "rb") as f:
                self.data_list = pickle.load(f)
        else:
            with open(f"{root}/{task}/{op}.pkl", "rb") as f:
                self.data_list = pickle.load(f)
            if audio_multi_turn:
                for i, item in enumerate(self.data_list[1]):
                    if item[3] >= 0:
                        word = item[4] + item[1][1:]
                        turn_id = [0 for _ in item[4]] + [1 for _ in range(len(word) - len(item[4]))]
                        audio = self.data_list[0][item[3]]
                    else:
                        word = item[1]
                        turn_id = [1 for _ in item[1]]
                        audio = []
                    self.data_list[1][i] = [self.data_list[0][item[0]], word, item[2], turn_id, audio]
                self.data_list = self.data_list[1]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]


class DataCollatorForPreTraining:
    def __init__(self, tokenizer, config, fp16=False, mlm_prob=0.15):
        self.tokenizer = tokenizer
        self.mlm_prob = mlm_prob
        self.config = config
        self.fp16 = fp16

    def get_mlm_instance(self, text_input):
        # text_input: tokenizer.encode之后的word indices列表。
        labels = text_input.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_prob)
        # special_tokens_mask：指定序列中哪些位置是special tokens，这些部分不能被mask。主要是[PAD][CLS][SEP]
        special_tokens_mask = self.tokenizer.get_special_tokens_mask(labels, already_has_special_tokens=True)
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # 使用labels[masked_indices]作为目标，或直接丢给RobertaForMaskedLM
        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        text_input[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        text_input[indices_random] = random_words[indices_random]
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return text_input, labels

    def __call__(self, batch):
        audios, a_mask, masked_text, text_labels, t_mask, start_valid, end_valid, token_type, starts, ends = [], [], [], [], [], [], [], [], [], []
        ml = 0
        for item in batch:
            ml = max([ml, len(item[1]) + len(item[4]) + len(item[8]) - 2, len(item[1]) + len(item[7]) + len(item[8]) - 2])
        ml = min(ml, self.config.text.max_length)
        for item in batch:
            aa, at, atr, pa, pt, ptr, na, nt, history = item
            # 文本pad之后有两个
            history, at, pt, nt = map(torch.LongTensor, [history, at, pt, nt])
            ht, h_mlm_label = self.get_mlm_instance(history)
            at, a_mlm_label = self.get_mlm_instance(at[1:])
            pt, p_mlm_label = self.get_mlm_instance(pt[1:])
            nt, _ = self.get_mlm_instance(nt[1:])
            positive = torch.cat([ht, at, pt])
            negative = torch.cat([ht, at, nt])
            if positive.shape[0] > ml:
                offset_p = ml - pt.shape[0] - 1
                offset_a = offset_p - at.shape[0]
            else:
                offset_a = history.shape[0] - 1
                offset_p = offset_a + at.shape[0]
            if negative.shape[0] > ml:
                offset_n = ml - nt.shape[0] - 1
            else:
                offset_n = offset_a + at.shape[0]
            p_text, p_tam = pad(positive, ml)
            n_text, n_tam = pad(negative, ml)
            asv, aev, asl, ael = compute_valid(atr, offset_a, offset_p)
            psv, pev, psl, pel = compute_valid(ptr, 0, ml - offset_p)
            sv = torch.cat([asv, psv])
            ev = torch.cat([aev, pev])
            start_valid.append(sv)
            end_valid.append(ev)
            starts.extend(asl + psl)
            ends.extend(ael + pel)
            p_token_type = torch.cat([torch.zeros(offset_p + 1), torch.ones(ml - offset_p - 1)]).long()
            n_token_type = torch.cat([torch.zeros(offset_n + 1), torch.ones(ml - offset_n - 1)]).long()
            mlm_label, _ = pad(torch.cat([h_mlm_label, a_mlm_label, p_mlm_label]), ml, -100)
            masked_text.extend([p_text, n_text])
            t_mask.extend([p_tam, n_tam])
            text_labels.append(mlm_label)
            token_type.extend([p_token_type, n_token_type])
            # 音频有三个
            aa, pa, na = map(torch.HalfTensor if self.fp16 else torch.FloatTensor, [aa, pa, na])
            aa, a_aam = pad(aa, self.config.audio.max_length)
            pa, p_aam = pad(pa, self.config.audio.max_length)
            na, n_aam = pad(na, self.config.audio.max_length)
            audios.extend([aa, pa, na])
            a_mask.extend([a_aam, p_aam, n_aam])
        audios, a_mask, masked_text, text_labels, t_mask, start_valid, end_valid, token_type = map(
            lambda x: torch.stack(x, dim=0),
            [audios, a_mask, masked_text, text_labels, t_mask, start_valid, end_valid, token_type]
        )
        starts, ends = map(lambda x: torch.tensor(x, dtype=audios.dtype), [starts, ends])
        return audios, a_mask, masked_text, text_labels, t_mask, start_valid, end_valid, token_type, starts, ends


class DataCollatorForDownstream:
    def __init__(self, audio_length, float_label):
        self.audio_length = audio_length
        self.float_label = float_label

    def __call__(self, batch):
        audios, a_mask, texts, labels, t_mask, turn_ids = [], [], [], [], [], []
        ml = 0
        for item in batch:
            ml = max(ml, len(item[1]))
        ml = min(ml, 512)
        for item in batch:
            audio, text, label = item[:3]
            text, tam = pad(torch.LongTensor(text), ml)
            texts.append(text)
            t_mask.append(tam)
            labels.append(label)
            if len(item) > 4:
                prev_audio, pam = pad(torch.HalfTensor(item[4]), self.audio_length)
                audios.append(prev_audio)
                a_mask.append(pam)
            audio, aam = pad(torch.HalfTensor(audio), self.audio_length)
            audios.append(audio)
            a_mask.append(aam)
            if len(item) > 3:
                token_type = pad(torch.LongTensor(item[3]), ml)[0]
                turn_ids.append(token_type)
        audios, a_mask, texts, t_mask = map(
            lambda x: torch.stack(x, dim=0),
            [audios, a_mask, texts, t_mask]
        )
        return {"audio": audios, "text": texts, "aam": a_mask, "tam": t_mask,
                "label": torch.HalfTensor(labels) if self.float_label else torch.LongTensor(labels),
                "turn_id": torch.stack(turn_ids, dim=0) if turn_ids else None}

