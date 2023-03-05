"""
Generator class.
"""

import bisect
import math

import numpy as np

import torch
from space.args import str2bool


def repeat(var, times):
    if isinstance(var, list):
        return [repeat(x, times) for x in var]
    elif isinstance(var, dict):
        return {k: repeat(v, times) for k, v in var.items()}
    elif isinstance(var, torch.Tensor):
        var = var.unsqueeze(1)
        expand_times = [1] * len(var.shape)
        expand_times[1] = times
        dtype = var.dtype
        var = var.float()
        var = var.repeat(*expand_times)
        shape = [var.shape[0] * var.shape[1]] + list(var.shape[2:])
        var = var.reshape(*shape)
        var = torch.tensor(var, dtype=dtype)
        return var
    else:
        return var


def gather(var, idx):
    if isinstance(var, list):
        return [gather(x, idx) for x in var]
    elif isinstance(var, dict):
        return {k: gather(v, idx) for k, v in var.items()}
    elif isinstance(var, torch.Tensor):
        out = var.index_select(dim=0, index=idx)
        return out
    else:
        return var


class Generator(object):
    """ Genrator class. """

    _registry = dict()

    @classmethod
    def register(cls, name):
        Generator._registry[name] = cls
        return

    @staticmethod
    def by_name(name):
        return Generator._registry[name]

    @staticmethod
    def create(hparams, *args, **kwargs):
        """ Create generator. """
        generator_cls = Generator.by_name(hparams.generator)
        return generator_cls(hparams, *args, **kwargs)

    @classmethod
    def add_cmdline_argument(cls, parser):
        group = parser.add_argument_group("Generator")
        group.add_argument("--generator", type=str, default="BeamSearch",
                           choices=["TopKSampling", "TopPSampling", "GreedySampling",
                                    "BeamSearch"])
        group.add_argument("--min_gen_len", type=int, default=1,
                           help="The minimum length of generated response.")
        group.add_argument("--max_gen_len", type=int, default=30,
                           help="The maximum length of generated response.")
        args, _ = parser.parse_known_args()
        generator_cls = cls.by_name(args.generator)
        generator_cls.add_cmdline_argument(group)
        return group

    def __init__(self, hparams, bpe):
        self.vocab_size = bpe.vocab_size
        self.bos_id = bpe.bos_id
        self.eos_id = bpe.eos_id
        self.unk_id = bpe.unk_id
        self.pad_id = bpe.pad_id
        self.min_gen_len = hparams.min_gen_len
        self.max_gen_len = hparams.max_gen_len
        self.use_gpu = hparams.use_gpu
        assert 1 <= self.min_gen_len <= self.max_gen_len
        return

    def __call__(self, step_fn, state):
        """
        Running generation.

        @param : step_fn : decoding one step
        @type : function

        @param : state : initial state
        @type : dict
        """
        raise NotImplementedError


class Sampling(Generator):
    """ Sampling Generator. """

    @classmethod
    def add_cmdline_argument(cls, group):
        group.add_argument("--ignore_unk", type=str2bool, default=True,
                           help="Whether to ignore unkown token in generation.")
        group.add_argument("--sampling_temperature", type=float, default=1.0)
        return group

    def __init__(self, hparams, bpe):
        super().__init__(hparams, bpe)
        self.ignore_unk = hparams.ignore_unk
        self.temperature = hparams.sampling_temperature
        return

    def _sampling(self, scores):
        """ Sampling function. """
        raise NotImplementedError

    def __call__(self, step_fn, state):
        """
        Running generation.

        @param : step_fn : decoding one step
        @type : function

        @param : state : initial state
        @type : dict
        """
        batch_size = state["batch_size"]
        vocab_size = self.vocab_size

        pos_index = torch.arange(0, batch_size, 1, dtype=torch.int64) * vocab_size

        # shape: [batch_size, beam_size, 1]
        predictions = torch.ones([batch_size, 1], dtype=torch.int64) * self.bos_id
        sequence_scores = torch.zeros([batch_size], dtype=torch.float32)

        unk_penalty = np.zeros(vocab_size, dtype="float32")
        unk_penalty[self.unk_id] = -1e10
        unk_penalty = torch.from_numpy(unk_penalty)

        eos_penalty = np.zeros(vocab_size, dtype="float32")
        eos_penalty[self.eos_id] = -1e10
        eos_penalty = torch.from_numpy(eos_penalty)

        scores_after_end = np.full(vocab_size, -1e10, dtype="float32")
        scores_after_end[self.pad_id] = 0
        scores_after_end = torch.from_numpy(scores_after_end)

        if self.use_gpu:
            pos_index = pos_index.cuda()
            predictions = predictions.cuda()
            sequence_scores = sequence_scores.cuda()
            unk_penalty = unk_penalty.cuda()
            eos_penalty = eos_penalty.cuda()
            scores_after_end = scores_after_end.cuda()

        # initial input
        for step in range(1, self.max_gen_len + 1):
            pre_ids = predictions[:, -1:]
            state["pred_token"] = pre_ids.unsqueeze(2)
            if step > 1:
                state["pred_mask"] = torch.not_equal(state["pred_token"], self.pad_id).float()
                state["pred_pos"] = state["pred_pos"] + 1
            scores, state = step_fn(state)

            # Generate next
            # scores shape: [batch_size, vocab_size]
            if self.ignore_unk:
                scores = scores + unk_penalty

            if step <= self.min_gen_len:
                scores = scores + eos_penalty

            # previous token is [PAD] or [EOS]
            # shape: [batch_size, 1]
            pre_eos_mask = (1 - torch.not_equal(pre_ids, self.eos_id).float()) + \
                           (1 - torch.not_equal(pre_ids, self.pad_id).float())
            scores = scores * (1 - pre_eos_mask) + \
                     pre_eos_mask.repeat(1, vocab_size) * scores_after_end

            scores = scores / self.temperature
            preds = self._sampling(scores)

            predictions = torch.cat([predictions, preds.unsqueeze(1)], dim=1)

            scores = scores.reshape(batch_size * vocab_size)
            preds = preds + pos_index
            scores = gather(scores, preds)
            sequence_scores = sequence_scores + scores

        results = {
            "preds": predictions,
            "scores": sequence_scores
        }
        return results


class GreedySampling(Sampling):
    """ Greedy sampling. """

    @classmethod
    def add_cmdline_argument(cls, group):
        return Sampling.add_cmdline_argument(group)

    def _sampling(self, logits):
        """ Implement greedy sampling. """
        preds = torch.argmax(logits, dim=1)
        return preds


class TopKSampling(Sampling):
    """ Top-k sampling. """

    @classmethod
    def add_cmdline_argument(cls, group):
        Sampling.add_cmdline_argument(group)
        group.add_argument("--top_k_ratio", type=float, default=None)
        group.add_argument("--top_k_num", type=int, default=None)
        return group

    def __init__(self, hparams, bpe):
        super().__init__(hparams, bpe)
        assert hparams.top_k_ratio is not None or hparams.top_k_num is not None
        if hparams.top_k_num is not None:
            self.top_k_num = hparams.top_k_num
        else:
            self.top_k_num = math.floor(hparams.top_k_ratio * self.vocab_size)
        assert self.top_k_num >= 1
        return

    def _sampling(self, logits):
        """ Implement top-k sampling. """
        probs = torch.softmax(logits, dim=1)
        probs, indices = torch.topk(probs, self.top_k_num)
        probs = probs / torch.sum(probs, dim=1, keepdim=True)
        preds = []
        for p, ids in zip(probs.numpy(), indices.numpy()):
            o = np.random.choice(ids, p=p)
            preds.append(o)
        preds = np.array(preds, dtype="int64")
        return torch.from_numpy(preds)


class TopPSampling(Sampling):
    """ Top-p sampling. """

    @classmethod
    def add_cmdline_argument(cls, group):
        Sampling.add_cmdline_argument(group)
        group.add_argument("--top_p_ratio", type=float, default=1.0)
        return group

    def __init__(self, hparams, bpe):
        super().__init__(hparams, bpe)
        self.top_p_ratio = hparams.top_p_ratio
        return

    def _sampling(self, logits):
        """ Implement top-k sampling. """
        probs = torch.softmax(logits, dim=1)
        preds = []
        for p in probs.numpy():
            ids = np.argsort(-p)
            p = p[ids]
            c_p = np.cumsum(p)
            i = bisect.bisect_right(c_p, self.top_p_ratio) + 1
            o = np.random.choice(ids[:i], p=p[:i]/np.sum(p[:i]))
            preds.append(o)
        preds = np.array(preds, dtype="int64")
        return torch.from_numpy(preds)


class BeamSearch(Generator):
    """ BeamSearch generator. """

    @classmethod
    def add_cmdline_argument(cls, group):
        group.add_argument("--beam_size", type=int, default=5,
                           help="The beam size in beam search.")
        group.add_argument("--length_average", type=str2bool, default=False,
                           help="Whether to use length average.")
        group.add_argument("--length_penalty", type=float, default=-1.0,
                           help="The parameter(alpha) of length penalty.")
        group.add_argument("--ignore_unk", type=str2bool, default=True,
                           help="Whether to ignore unkown token in generation.")
        return group

    def __init__(self, hparams, bpe):
        super().__init__(hparams, bpe)
        self.beam_size = hparams.beam_size
        self.length_average = hparams.length_average
        self.length_penalty = hparams.length_penalty
        self.ignore_unk = hparams.ignore_unk
        return

    def __call__(self, step_fn, state):
        """
        Running beam search.

        @param : step_fn : decoding one step
        @type : function

        @param : state : initial state
        @type : dict
        """
        batch_size = state["batch_size"]
        beam_size = self.beam_size

        # shape: [batch_size, 1]
        pos_index = torch.arange(0, batch_size, 1, dtype=torch.int64) * beam_size
        pos_index = pos_index.unsqueeze(1)

        # shape: [batch_size, beam_size, 1]
        predictions = torch.ones([batch_size, beam_size, 1], dtype=torch.int64) * self.bos_id

        if self.use_gpu:
            pos_index = pos_index.cuda()
            predictions = predictions.cuda()

        # initial input (bos_id)
        state["pred_token"] = predictions[:, :1]
        # shape: [batch_size, vocab_size]
        scores, state = step_fn(state)

        unk_penalty = np.zeros(self.vocab_size, dtype="float32")
        unk_penalty[self.unk_id] = -1e10
        unk_penalty = torch.from_numpy(unk_penalty)

        eos_penalty = np.zeros(self.vocab_size, dtype="float32")
        eos_penalty[self.eos_id] = -1e10
        eos_penalty = torch.from_numpy(eos_penalty)

        scores_after_end = np.full(self.vocab_size, -1e10, dtype="float32")
        scores_after_end[self.pad_id] = 0  # 希望<eos>之后只生成<pad>，故使词表中log(p(<pad>))最高(0)
        scores_after_end = torch.from_numpy(scores_after_end)

        if self.use_gpu:
            unk_penalty = unk_penalty.cuda()
            eos_penalty = eos_penalty.cuda()
            scores_after_end = scores_after_end.cuda()

        if self.ignore_unk:
            scores = scores + unk_penalty
        scores = scores + eos_penalty

        # shape: [batch_size, beam_size]
        sequence_scores, preds = torch.topk(scores, self.beam_size)

        predictions = torch.cat([predictions, preds.unsqueeze(2)], dim=2)
        state = repeat(state, beam_size)

        parent_idx_list = []
        pred_list = []

        for step in range(2, self.max_gen_len + 1):
            pre_ids = predictions[:, :, -1:]
            state["pred_token"] = pre_ids.reshape(batch_size * beam_size, 1, 1)
            state["pred_mask"] = torch.not_equal(state["pred_token"], self.pad_id).float()
            state["pred_pos"] = state["pred_pos"] + 1
            scores, state = step_fn(state)

            # Generate next
            # scores shape: [batch_size * beam_size, vocab_size]
            if self.ignore_unk:
                scores = scores + unk_penalty

            if step <= self.min_gen_len:
                scores = scores + eos_penalty

            # scores shape: [batch_size, beam_size, vocab_size]
            scores = scores.reshape(batch_size, beam_size, self.vocab_size)

            # previous token is [PAD] or [EOS]
            pre_eos_mask = (1 - torch.not_equal(pre_ids, self.eos_id).float()) + \
                           (1 - torch.not_equal(pre_ids, self.pad_id).float())

            scores = scores * (1 - pre_eos_mask) + \
                     pre_eos_mask.repeat(1, 1, self.vocab_size) * scores_after_end
            if self.length_average:
                scaled_value = pre_eos_mask + (1 - pre_eos_mask) * (1 - 1 / step)
                sequence_scores = sequence_scores.unsqueeze(2) * scaled_value
                scaled_value = pre_eos_mask + (1 - pre_eos_mask) * (1 / step)
                scores = scores * scaled_value
            elif self.length_penalty >= 0.0:
                scaled_value = pre_eos_mask + (1 - pre_eos_mask) * \
                    (math.pow((4 + step) / (5 + step), self.length_penalty))
                sequence_scores = scaled_value * sequence_scores
                scaled_value = pre_eos_mask + (1 - pre_eos_mask) * \
                    (math.pow(1 / (5 + step), self.length_penalty))
                scores = scores * scaled_value
            scores = scores + sequence_scores.unsqueeze(-1)
            scores = scores.reshape(batch_size, beam_size * self.vocab_size)

            topk_scores, topk_indices = torch.topk(scores, beam_size)
            # topk_indices: [batch_size, beam_size * self.vocab_size] (已reshape)
            # 判断当前时间步产生词的前一个词在哪个beam中，对vocab_size取商
            parent_idx = topk_indices.floor_divide(self.vocab_size)
            # 对vocab_size取余
            preds = topk_indices % self.vocab_size

            # Gather state / sequence_scores
            parent_idx = parent_idx + pos_index
            parent_idx = parent_idx.reshape(batch_size * beam_size)
            state = gather(state, parent_idx)
            sequence_scores = topk_scores

            predictions = predictions.reshape(batch_size * beam_size, step)
            predictions = gather(predictions, parent_idx)
            predictions = predictions.reshape(batch_size, beam_size, step)
            predictions = torch.cat([predictions, preds.unsqueeze(2)], dim=2)

        # 希望生成的整个句子已完结，所以要求最后一个token为<eos>或者<pad>(跟在<eos>之后)，否则惩罚
        pre_ids = predictions[:, :, -1]
        pre_eos_mask = (1 - torch.not_equal(pre_ids, self.eos_id).float()) + \
                       (1 - torch.not_equal(pre_ids, self.pad_id).float())
        sequence_scores = sequence_scores * pre_eos_mask + (1 - pre_eos_mask) * (-1e10)

        '''
        这里的gather方法相当于pytorch中的index_select方法
        因为gather/index_select的index参数只能是1阶的形式，不能在不同维度指定不同的index
        故需引入pos_index，先将数据都转化为1阶，再通过index+pos_index进行索引
        '''
        # 先获得ascending排序的index，便于之后对predictions和sequence_scores排序(针对beam size轴)
        indices = torch.argsort(sequence_scores, dim=1)
        indices = indices + pos_index
        indices = indices.reshape(-1)
        sequence_scores = sequence_scores.reshape(batch_size * beam_size)
        predictions = predictions.reshape(batch_size * beam_size, -1)
        sequence_scores = gather(sequence_scores, indices)
        predictions = gather(predictions, indices)
        sequence_scores = sequence_scores.reshape(batch_size, beam_size)
        predictions = predictions.reshape(batch_size, beam_size, -1)

        results = {
            "preds": predictions[:, -1],
            "scores": sequence_scores[:, -1]
        }
        return results


BeamSearch.register("BeamSearch")
GreedySampling.register("GreedySampling")
TopKSampling.register("TopKSampling")
TopPSampling.register("TopPSampling")
