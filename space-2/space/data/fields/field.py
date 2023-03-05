"""
Field class
"""
import multiprocessing
import random
from itertools import chain
import os
import glob
import json
import numpy as np
import time
import re
from tqdm import tqdm

from space.args import str2bool
from space.data.tokenizer import Tokenizer
from space.utils import ontology
from space.utils.scores import hierarchical_set_score


def max_lens(X):
    lens = [len(X)]
    while isinstance(X[0], list):
        lens.append(max(map(len, X)))
        X = [x for xs in X for x in xs]
    return lens


def list2np(X, padding=0, dtype="int64"):
    shape = max_lens(X)
    ret = np.full(shape, padding, dtype=np.int32)

    if len(shape) == 1:
        ret = np.array(X)
    elif len(shape) == 2:
        for i, x in enumerate(X):
            ret[i, :len(x)] = np.array(x)
    elif len(shape) == 3:
        for i, xs in enumerate(X):
            for j, x in enumerate(xs):
                ret[i, j, :len(x)] = np.array(x)
    return ret.astype(dtype)


class BPETextField(object):

    pad_token = "[PAD]"
    bos_token = "[BOS]"
    eos_token = "[EOS]"
    unk_token = "[UNK]"
    cls_token = "[CLS]"
    mask_token = "[MASK]"
    sos_u_token = "<sos_u>"
    eos_u_token = "<eos_u>"
    sos_b_token = "<sos_b>"
    eos_b_token = "<eos_b>"
    sos_db_token = "<sos_db>"
    eos_db_token = "<eos_db>"
    sos_a_token = "<sos_a>"
    eos_a_token = "<eos_a>"
    sos_r_token = "<sos_r>"
    eos_r_token = "<eos_r>"

    @classmethod
    def add_cmdline_argument(cls, parser, group=None):
        group = parser.add_argument_group("BPETextField") if group is None else group
        group.add_argument("--vocab_path", type=str, required=True,
                           help="The vocabulary file path.")
        group.add_argument("--filtered", type=str2bool, default=False,
                           help="Whether to filter the data with too long utterance/context. "
                           "If the data is unfiltered, it will be truncated.")
        group.add_argument("--prompt_num_for_understand", type=int, default=5,
                           help="The num of prompts for understanding.")
        group.add_argument("--with_cls", type=str2bool, default=True,
                           help="Whether to use [CLS] position to extract semantics.")
        group.add_argument("--max_len", type=int, default=256,
                           help="The maximum length of context.")
        group.add_argument("--min_utt_len", type=int, default=1,
                           help="The minimum length of utterance.")
        group.add_argument("--max_utt_len", type=int, default=50,
                           help="The maximum length of utterance.")
        group.add_argument("--min_ctx_turn", type=int, default=1,
                           help="The minimum turn of context.")
        group.add_argument("--max_ctx_turn", type=int, default=16,
                           help="The maximum turn of context.")
        group.add_argument("--tokenizer_type", type=str, default="Bert",
                           choices=["Bert", "GPT2"],
                           help="The type of tokenizer.")
        return group

    def __init__(self, hparams):
        self.score_matrixs = {}
        self.understand_tokens = ontology.get_understand_tokens(hparams.prompt_num_for_understand)
        special_tokens = [self.pad_token, self.bos_token, self.eos_token, self.unk_token]
        special_tokens.extend(self.add_sepcial_tokens())
        self.tokenizer = Tokenizer(vocab_path=hparams.vocab_path,
                                   special_tokens=special_tokens,
                                   tokenizer_type=hparams.tokenizer_type)
        self.understand_ids = self.numericalize(self.understand_tokens)
        self.tokenizer_type = hparams.tokenizer_type
        self.filtered = hparams.filtered
        self.max_len = hparams.max_len
        self.min_utt_len = hparams.min_utt_len
        self.max_utt_len = hparams.max_utt_len
        self.min_ctx_turn = hparams.min_ctx_turn
        self.max_ctx_turn = hparams.max_ctx_turn
        self.with_mlm = hparams.with_mlm
        self.with_cls = hparams.with_cls
        self.with_contrastive = hparams.with_contrastive
        self.num_process = hparams.num_process
        self.dynamic_score = hparams.dynamic_score
        self.trigger_role = hparams.trigger_role
        self.trigger_data = hparams.trigger_data.split(',') if hparams.trigger_data else []

        data_paths = list(os.path.dirname(c) for c in sorted(
            glob.glob(hparams.data_dir + '/**/' + f'train.{hparams.tokenizer_type}.jsonl', recursive=True)))
        self.data_paths = self.filter_data_path(data_paths=data_paths)
        self.labeled_data_paths = [data_path for data_path in self.data_paths if 'AnPreDial' in data_path]
        self.unlabeled_data_paths = [data_path for data_path in self.data_paths if 'UnPreDial' in data_path]

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    @property
    def num_specials(self):
        return len(self.tokenizer.special_tokens)

    @property
    def pad_id(self):
        return self.tokenizer.convert_tokens_to_ids([self.pad_token])[0]

    @property
    def bos_id(self):
        return self.tokenizer.convert_tokens_to_ids([self.bos_token])[0]

    @property
    def eos_id(self):
        return self.tokenizer.convert_tokens_to_ids([self.eos_token])[0]

    @property
    def unk_id(self):
        return self.tokenizer.convert_tokens_to_ids([self.unk_token])[0]

    @property
    def cls_id(self):
        return self.tokenizer.convert_tokens_to_ids([self.cls_token])[0]

    @property
    def mask_id(self):
        return self.tokenizer.convert_tokens_to_ids([self.mask_token])[0]

    @property
    def sos_u_id(self):
        return self.tokenizer.convert_tokens_to_ids([self.sos_u_token])[0]

    @property
    def eos_u_id(self):
        return self.tokenizer.convert_tokens_to_ids([self.eos_u_token])[0]

    @property
    def sos_b_id(self):
        return self.tokenizer.convert_tokens_to_ids([self.sos_b_token])[0]

    @property
    def eos_b_id(self):
        return self.tokenizer.convert_tokens_to_ids([self.eos_b_token])[0]

    @property
    def sos_db_id(self):
        return self.tokenizer.convert_tokens_to_ids([self.sos_db_token])[0]

    @property
    def eos_db_id(self):
        return self.tokenizer.convert_tokens_to_ids([self.eos_db_token])[0]

    @property
    def sos_a_id(self):
        return self.tokenizer.convert_tokens_to_ids([self.sos_a_token])[0]

    @property
    def eos_a_id(self):
        return self.tokenizer.convert_tokens_to_ids([self.eos_a_token])[0]

    @property
    def sos_r_id(self):
        return self.tokenizer.convert_tokens_to_ids([self.sos_r_token])[0]

    @property
    def eos_r_id(self):
        return self.tokenizer.convert_tokens_to_ids([self.eos_r_token])[0]

    @property
    def bot_id(self):
        """
        用于区分user和bot两个角色
        1和0不是词表中的index，而是专门针对role的index，大小就为2，对应超参数'num_type_embeddings'
        """
        return 0

    @property
    def user_id(self):
        """
        用于区分user和bot两个角色
        1和0不是词表中的index，而是专门针对role的index，大小就为2，对应超参数'num_type_embeddings'
        """
        return 1

    def add_sepcial_tokens(self):
        return ontology.get_special_tokens(understand_tokens=self.understand_tokens)

    def filter_data_path(self, data_paths):
        filtered_data_paths = []
        if self.trigger_data:
            for data_path in data_paths:
                for data_name in self.trigger_data:
                    if data_path.endswith(f'/{data_name}'):
                        filtered_data_paths.append(data_path)
                        break
        else:
            for data_path in data_paths:
                if data_path.endswith(f'_few'):
                    continue
                if data_path.endswith(f'_valid'):
                    continue
                filtered_data_paths.append(data_path)
        return filtered_data_paths

    def load_score_matrix(self, data_type, data_iter=None):
        """
        load score matrix for all labeled datasets
        """
        for data_path in self.labeled_data_paths:
            file_index = os.path.join(data_path, f'{data_type}.{self.tokenizer_type}.jsonl')
            file = os.path.join(data_path, f'{data_type}.Score.npy')
            if self.dynamic_score:
                score_matrix = {}
                print(f"Created 1 score cache dict for data in '{file_index}'")
            else:
                assert os.path.exists(file), f"{file} isn't exist"
                print(f"Loading 1 score matrix from '{file}' ...")
                fp = np.memmap(file, dtype='float32', mode='r')
                assert len(fp.shape) == 1
                num = int(np.sqrt(fp.shape[0]))
                score_matrix = fp.reshape(num, num)
                print(f"Loaded 1 score matrix for data in '{file_index}'")
            self.score_matrixs[file_index] = score_matrix

    def random_word(self, chars):
        output_label = []
        output_chars = []

        for i, char in enumerate(chars):
            # TODO delete this part to learn special tokens
            if char in [self.sos_u_id, self.eos_u_id, self.sos_r_id, self.eos_r_id]:
                output_chars.append(char)
                output_label.append(self.pad_id)
                continue

            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    output_chars.append(self.mask_id)

                # 10% randomly change token to random token
                elif prob < 0.9:
                    output_chars.append(random.randint(1, self.vocab_size - 1))  # start from 1, to exclude pad_id

                # 10% randomly change token to current token
                else:
                    output_chars.append(char)

                output_label.append(char)

            else:
                output_chars.append(char)
                output_label.append(self.pad_id)

        return output_chars, output_label

    def create_masked_lm_predictions(self, sample):
        src = sample['src']
        src_span_mask = sample['src_span_mask']
        mlm_inputs = []
        mlm_labels = []
        for chars, chars_span_mask in zip(src, src_span_mask):
            if sum(chars_span_mask):
                mlm_input, mlm_label = [], []
                for char, char_mask in zip(chars, chars_span_mask):
                    if char_mask:
                        mlm_input.append(self.mask_id)
                        mlm_label.append(char)
                    else:
                        mlm_input.append(char)
                        mlm_label.append(self.pad_id)
            else:
                mlm_input, mlm_label = self.random_word(chars)
            mlm_inputs.append(mlm_input)
            mlm_labels.append(mlm_label)

        sample['mlm_inputs'] = mlm_inputs
        sample['mlm_labels'] = mlm_labels
        return sample

    def create_span_masked_lm_predictions(self, sample):
        src = sample['src']
        src_span_mask = sample['src_span_mask']
        mlm_inputs = []
        mlm_labels = []
        for chars, chars_span_mask in zip(src, src_span_mask):
            mlm_input, mlm_label = [], []
            for char, char_mask in zip(chars, chars_span_mask):
                if char_mask:
                    mlm_input.append(self.mask_id)
                    mlm_label.append(char)
                else:
                    mlm_input.append(char)
                    mlm_label.append(self.pad_id)
            mlm_inputs.append(mlm_input)
            mlm_labels.append(mlm_label)

        sample['mlm_inputs'] = mlm_inputs
        sample['mlm_labels'] = mlm_labels
        return sample

    def create_token_masked_lm_predictions(self, sample):
        mlm_inputs = sample['mlm_inputs']
        mlm_labels = sample['mlm_labels']

        for i, span_mlm_label in enumerate(mlm_labels):
            if not sum(span_mlm_label):
                mlm_input, mlm_label = self.random_word(mlm_inputs[i])
                mlm_inputs[i] = mlm_input
                mlm_labels[i] = mlm_label

        return sample

    def numericalize(self, tokens):
        """
        here only "convert_tokens_to_ids",
        which need be tokenized into tokens(sub-words) by "tokenizer.tokenize" before
        """
        assert isinstance(tokens, list)
        if len(tokens) == 0:
            return []
        element = tokens[0]
        if isinstance(element, list):
            return [self.numericalize(s) for s in tokens]
        else:
            return self.tokenizer.convert_tokens_to_ids(tokens)

    def denumericalize(self, numbers):
        """
        here first "convert_ids_to_tokens", then combine sub-words into origin words
        """
        assert isinstance(numbers, list)
        if len(numbers) == 0:
            return []
        element = numbers[0]
        if isinstance(element, list):
            return [self.denumericalize(x) for x in numbers]
        else:
            return self.tokenizer.decode(
                numbers, ignore_tokens=[self.bos_token, self.eos_token, self.pad_token])

    def save_examples(self, examples, filename):
        start = time.time()
        if filename.endswith("npy"):
            print(f"Saving 1 object to '{filename}' ...")
            assert len(examples.shape) == 2 and examples.shape[0] == examples.shape[1]
            num = examples.shape[0]
            fp = np.memmap(filename, dtype='float32', mode='w+', shape=(num, num))
            fp[:] = examples[:]
            fp.flush()
            elapsed = time.time() - start
            print(f"Saved 1 object (elapsed {elapsed:.2f}s)")
        elif filename.endswith("jsonl"):
            print(f"Saving examples to '{filename}' ...")
            with open(filename, "w", encoding="utf-8") as fp:
                for ex in examples:
                    fp.write(json.dumps(ex) + "\n")
            elapsed = time.time() - start
            print(f"Saved {len(examples)} examples (elapsed {elapsed:.2f}s)")
        else:
            print(f"Saving examples to '{filename}' ...")
            raise ValueError(f"Unsport file format: {filename}")

    def load_examples(self, filename):
        start = time.time()
        if filename.endswith("npy"):
            print(f"Loading 1 object from '{filename}' ...")
            fp = np.memmap(filename, dtype='float32', mode='r')
            assert len(fp.shape) == 1
            num = int(np.sqrt(fp.shape[0]))
            examples = fp.reshape(num, num)
            elapsed = time.time() - start
            print(f"Loaded 1 object (elapsed {elapsed:.2f}s)")
        else:
            print(f"Loading examples from '{filename}' ...")
            with open(filename, "r", encoding="utf-8") as fp:
                examples = list(map(lambda s: json.loads(s.strip()), fp))
            elapsed = time.time() - start
            print(f"Loaded {len(examples)} examples (elapsed {elapsed:.2f}s)")
        return examples

    def utt_filter_pred(self, utt):
        return self.min_utt_len <= len(utt) \
            and (not self.filtered or len(utt) <= self.max_utt_len)

    def utts_filter_pred(self, utts):
        return self.min_ctx_turn <= len(utts) \
            and (not self.filtered or len(utts) <= self.max_ctx_turn)

    def get_token_pos(self, tok_list, value_label):
        find_pos = []
        found = False
        label_list = [item for item in map(str.strip, re.split("(\W+)", value_label.lower())) if len(item) > 0]
        len_label = len(label_list)
        for i in range(len(tok_list) + 1 - len_label):
            if tok_list[i:i + len_label] == label_list:
                find_pos.append((i, i + len_label))  # start, exclusive_end
                found = True
        return found, find_pos

    def build_score_matrix(self, examples):
        """
        build symmetric score matrix
        """
        assert self.num_process == 1
        print(f"Building score matrix from examples ...")
        num = len(examples)
        score_matrix = np.eye(num, num, dtype='float32')  # in case of empty label of self, resulting in score 0.

        for i in tqdm(range(num)):
            for j in range(i):
                # TODO change the score method
                score = hierarchical_set_score(frame1=examples[i]['label'], frame2=examples[j]['label'])
                score_matrix[i][j] = score
                score_matrix[j][i] = score

        print(f"Built score matrix")
        return score_matrix

    def build_score_matrix_on_the_fly(self, ids, labels, data_file):
        """
        build symmetric score matrix on the fly
        """
        num = len(labels)
        assert len(ids) == len(labels)
        score_matrix = np.eye(num, num, dtype='float32')  # in case of empty label of self, resulting in score 0.

        for i in range(num):
            for j in range(i):
                score = self.score_matrixs[data_file].get(f'{ids[i]}-{ids[j]}', None)
                if score is None:
                    score = self.score_matrixs[data_file].get(f'{ids[j]}-{ids[i]}', None)
                if score is None:
                    # TODO change the score method
                    score = hierarchical_set_score(frame1=labels[i], frame2=labels[j])
                    self.score_matrixs[data_file][f'{ids[i]}-{ids[j]}'] = score
                score_matrix[i][j] = score
                score_matrix[j][i] = score

        return score_matrix

    def build_score_multiple_matrix_on_the_fly(self, ids, labels, data_file, score_ids):
        """
        build multiple symmetric score matrix on the fly
        """
        assert len(ids) == len(labels)
        num, num_matrix = len(labels), len(score_ids)
        score_matrices = [np.eye(num, num, dtype='float32') for _ in range(num_matrix)]  # in case of empty label of self, resulting in score 0.

        for i in range(num):
            for j in range(i):
                # look up scores
                score = self.score_matrixs[data_file].get(f'{ids[i]}-{ids[j]}', None)
                if score is None:
                    score = self.score_matrixs[data_file].get(f'{ids[j]}-{ids[i]}', None)
                if score is None:
                    all_score = hierarchical_set_score(frame1=labels[i], frame2=labels[j], return_all=True)
                    score = [all_score[si] for si in score_ids]
                    self.score_matrixs[data_file][f'{ids[i]}-{ids[j]}'] = score

                # fill in score matrices
                assert len(score) == num_matrix
                for k in range(num_matrix):
                    score_matrices[k][i][j] = score[k]
                    score_matrices[k][j][i] = score[k]

        return score_matrices

    def build_score_matrix_func(self, examples, start, exclusive_end):
        """
        build sub score matrix
        """
        num = len(examples)
        process_id = os.getpid()
        description = f'PID: {process_id} Start: {start} End: {exclusive_end}'
        print(f"PID-{process_id}: Building {start} to {exclusive_end} lines score matrix from examples ...")
        score_matrix = np.zeros((exclusive_end - start, num), dtype='float32')

        for abs_i, i in enumerate(tqdm(range(start, exclusive_end), desc=description)):
            for j in range(num):
                # TODO change the score method
                score = hierarchical_set_score(frame1=examples[i]['label'], frame2=examples[j]['label'])
                score_matrix[abs_i][j] = score

        print(f"PID-{process_id}: Built {start} to {exclusive_end} lines score matrix")
        return {"start": start, "score_matrix": score_matrix}

    def build_score_matrix_multiprocessing(self, examples):
        """
        build score matrix
        """
        assert self.num_process >= 2 and multiprocessing.cpu_count() >= 2
        print(f"Building score matrix from examples ...")
        results = []
        num = len(examples)
        sub_num, res_num = num // self.num_process, num % self.num_process
        patches = [sub_num] * (self.num_process - 1) + [sub_num + res_num]

        start = 0
        pool = multiprocessing.Pool(processes=self.num_process)
        for patch in patches:
            exclusive_end = start + patch
            results.append(pool.apply_async(self.build_score_matrix_func, (examples, start, exclusive_end)))
            start = exclusive_end
        pool.close()
        pool.join()

        sub_score_matrixs = [result.get() for result in results]
        sub_score_matrixs = sorted(sub_score_matrixs, key=lambda sub: sub['start'])
        sub_score_matrixs = [sub_score_matrix['score_matrix'] for sub_score_matrix in sub_score_matrixs]
        score_matrix = np.concatenate(sub_score_matrixs, axis=0)
        assert score_matrix.shape == (num, num)
        np.fill_diagonal(score_matrix, 1.)  # in case of empty label of self, resulting in score 0.

        print(f"Built score matrix")
        return score_matrix

    def extract_span_texts(self, text, label):
        span_texts = []
        for domain, frame in label.items():
            for act, slot_values in frame.items():
                for slot, values in slot_values.items():
                    for value in values:
                        if value['span']:
                            span_texts.append(text[value['span'][0]: value['span'][1]])
                        elif str(value['value']).strip().lower() in text.strip().lower():
                            span_texts.append(str(value['value']))
        return span_texts

    def fix_label(self, label):
        for domain, frame in label.items():
            if not frame:
                return {}
            for act, slot_values in frame.items():
                if act == 'DEFAULT_INTENT' and not slot_values:
                    return {}
        return label

    def build_examples_multi_turn(self, data_file, data_type="train"):
        print(f"Reading examples from '{data_file}' ...")
        examples = []
        ignored = 0

        with open(data_file, "r", encoding="utf-8") as f:
            input_data = json.load(f)
            for dialog_id in tqdm(input_data):
                turns = input_data[dialog_id]['turns']
                history, history_role, history_span_mask = [], [], []
                for turn in turns:
                    label = turn['label']
                    role = turn['role']
                    text = turn['text']
                    utterance, span_mask = [], []

                    token_list = [tok for tok in map(str.strip, re.split("(\W+)", text.lower())) if len(tok) > 0]
                    span_list = np.zeros(len(token_list), dtype=np.int32)
                    span_texts = self.extract_span_texts(text=text, label=label)

                    for span_text in span_texts:
                        found, find_pos = self.get_token_pos(tok_list=token_list, value_label=span_text)
                        if found:
                            for start, exclusive_end in find_pos:
                                span_list[start: exclusive_end] = 1

                    token_list = [self.tokenizer.tokenize(token) for token in token_list]
                    span_list = [[tag] * len(token_list[i]) for i, tag in enumerate(span_list)]
                    for sub_tokens in token_list:
                        utterance.extend(sub_tokens)
                    for sub_spans in span_list:
                        span_mask.extend(sub_spans)
                    assert len(utterance) == len(span_mask)

                    history.append(utterance)
                    history_role.append(role)
                    history_span_mask.append(span_mask)

                    if ((self.utts_filter_pred(history) and all(map(self.utt_filter_pred, history)))
                        or data_type == "test") and role in self.trigger_role:  # TODO consider test
                        src = [s[-self.max_utt_len:] for s in history[-self.max_ctx_turn:]]
                        src_span_mask = [s[-self.max_utt_len:] for s in history_span_mask[-self.max_ctx_turn:]]
                        roles = [role for role in history_role[-self.max_ctx_turn:]]
                        src = [[self.sos_u_id] + self.numericalize(s) + [self.eos_u_id]
                               if roles[i] == 'user' else
                               [self.sos_r_id] + self.numericalize(s) + [self.eos_r_id]
                               for i, s in enumerate(src)]
                        src_span_mask = [[0] + list(map(int, s)) + [0] for s in src_span_mask]

                        ex = {"dialog_id": dialog_id,
                              "turn_id": turn['turn_id'],
                              "role": role,
                              "src": src,
                              "src_span_mask": src_span_mask,
                              "label": self.fix_label(label),
                              "extra_info": turn.get("extra_info", "")}
                        examples.append(ex)
                    else:
                        ignored += 1

        # add span mlm inputs and span mlm labels in advance
        if self.with_mlm:
            examples = [self.create_span_masked_lm_predictions(example) for example in examples]

        # add absolute id of the dataset for indexing scores in its score matrix
        for i, example in enumerate(examples):
            example['id'] = i

        print(f"Built {len(examples)} {data_type.upper()} examples ({ignored} filtered)")
        return examples

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

        if self.dynamic_score and self.with_contrastive:
            labels = [sp["label"] for sp in samples]
            batch['labels'] = labels
            batch['label_ids'] = np.arange(batch_size)

        return batch, batch_size
