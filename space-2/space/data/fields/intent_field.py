"""
Intent Field class
"""
import random
import numpy as np

from collections import defaultdict
from itertools import chain

from space.data.fields.field import BPETextField, list2np


class IntentBPETextField(BPETextField):

    @classmethod
    def add_cmdline_argument(cls, parser):
        group = parser.add_argument_group("BPETextField")
        # group.add_argument("--vocab_path", type=str, required=True,
        #                    help="The vocabulary file path.")
        BPETextField.add_cmdline_argument(parser, group=group)
        return group

    def __init__(self, hparams):
        super(IntentBPETextField, self).__init__(hparams)

    def retrieve_examples(self, dataset, labels, inds, task, num=None, cache=None):
        assert task == "intent", "Example-driven may only be used with intent prediction"
        if num is None and labels is not None:
            num = len(labels) * 2

        # Populate cache
        if cache is None:
            cache = defaultdict(list)
            for i, example in enumerate(dataset):
                assert i == example['id']
                cache[example['extra_info']['intent_label']].append(i)

        # One example for each label
        example_inds = []
        for l in set(labels.tolist()):
            if l == -1:
                continue

            ind = random.choice(cache[l])
            retries = 0
            while ind in inds.tolist() or type(ind) is not int:
                ind = random.choice(cache[l])
                retries += 1
                if retries > len(dataset):
                    break

            example_inds.append(ind)

        # Sample randomly until we hit batch size
        while len(example_inds) < min(len(dataset), num):
            ind = random.randint(0, len(dataset) - 1)
            if ind not in example_inds and ind not in inds.tolist():
                example_inds.append(ind)

        # Create examples
        example_batch = {}
        examples = [dataset[i] for i in example_inds]
        examples, _ = self.collate_fn_multi_turn(examples)
        example_batch['example_src_token'] = examples['src_token']
        example_batch['example_src_pos'] = examples['src_pos']
        example_batch['example_src_type'] = examples['src_type']
        example_batch['example_src_turn'] = examples['src_turn']
        example_batch['example_src_mask'] = examples['src_mask']
        example_batch['example_intent'] = examples['intent_label']
        if not self.with_cls:
            example_batch['example_tgt_token'] = examples['tgt_token']
            example_batch['example_tgt_mask'] = examples['tgt_mask']

        return example_batch

    def collate_fn_multi_turn(self, samples):
        batch_size = len(samples)
        batch = {}

        cur_roles = [sp["role"] for sp in samples]
        src = [sp["src"] for sp in samples]
        src_token, src_pos, src_turn, src_role = [], [], [], []
        for utts, cur_role in zip(src, cur_roles):
            utt_lens = [len(utt) for utt in utts]

            # Token ids
            src_token.append(list(chain(*utts))[-self.max_len:])

            # Position ids
            pos = [list(range(l)) for l in utt_lens]
            src_pos.append(list(chain(*pos))[-self.max_len:])

            # Turn ids
            turn = [[len(utts) - i] * l for i, l in enumerate(utt_lens)]
            src_turn.append(list(chain(*turn))[-self.max_len:])

            # Role ids
            if cur_role == 'user':
                role = [[self.bot_id if (len(utts) - i) % 2 == 0 else self.user_id] * l
                        for i, l in enumerate(utt_lens)]
            else:
                role = [[self.user_id if (len(utts) - i) % 2 == 0 else self.bot_id] * l
                        for i, l in enumerate(utt_lens)]
            src_role.append(list(chain(*role))[-self.max_len:])

        # src端序列和tgt端序列需要分开pad，以保证解码时第一个词对齐
        src_token = list2np(src_token, padding=self.pad_id)
        src_pos = list2np(src_pos, padding=self.pad_id)
        src_turn = list2np(src_turn, padding=self.pad_id)
        src_role = list2np(src_role, padding=self.pad_id)
        batch["src_token"] = src_token
        batch["src_pos"] = src_pos
        batch["src_type"] = src_role
        batch["src_turn"] = src_turn
        batch["src_mask"] = (src_token != self.pad_id).astype("int64")  # input mask

        if self.with_mlm:
            mlm_token, mlm_label = [], []
            raw_mlm_input = [sp["mlm_inputs"] for sp in samples]
            raw_mlm_label = [sp["mlm_labels"] for sp in samples]
            for inputs in raw_mlm_input:
                mlm_token.append(list(chain(*inputs))[-self.max_len:])
            for labels in raw_mlm_label:
                mlm_label.append(list(chain(*labels))[-self.max_len:])

            mlm_token = list2np(mlm_token, padding=self.pad_id)
            mlm_label = list2np(mlm_label, padding=self.pad_id)
            batch["mlm_token"] = mlm_token
            batch["mlm_label"] = mlm_label
            batch["mlm_mask"] = (mlm_label != self.pad_id).astype("int64")  # label mask

        if not self.with_cls:
            assert self.understand_ids
            tgt = [self.understand_ids for _ in samples]
            tgt_token = np.array(tgt).astype("int64")
            batch["tgt_token"] = tgt_token
            batch["tgt_mask"] = (tgt_token != self.pad_id).astype("int64")  # input mask

        if 'id' in samples[0]:
            ids = [sp["id"] for sp in samples]
            ids = np.array(ids).astype("int64")
            batch['ids'] = ids

        if 'intent_label' in samples[0]['extra_info']:
            intent_label = [sample['extra_info']['intent_label'] for sample in samples]
            intent_label = np.array(intent_label).astype("int64")
            batch['intent_label'] = intent_label

        return batch, batch_size
