from numpy import pad
import torch
import io
import random
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from .base_dataset import BaseDataset
from pace.utils.glossary import *
from pace.utils.write_mmconv_dst import *

class MMConvDSTDataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        if split == "train":
            names = ["mmconv_dst_train"]
        elif split == "val":
            names = ["mmconv_dst_dev"]
        elif split == "test":
            names = ["mmconv_dst_test"]

        super().__init__(*args, **kwargs, names=names, text_column_name="caption")
        self.update_task()

    def update_task(self):
        self.len_map = {}
        self.item_idx_map = {}
        self.sample_slot_map = {}
        self.data_var_token = '<|belief|>'
        self.split_data_var='; '
        self.split_token='<|endofcontext|>'
        total = 0
        for i, sample in enumerate(self.all_texts):
            sample = sample[0]
            last_total = total

            slots = extract(sample, self.data_var_token)[0]
            if slots:
                slots = [s for s in slots.split(self.split_data_var) if not s.startswith('img_gt')]
            else:
                slots = []
            slot_names_with_values = set()
            slot_names_without_values = set()
            for slot in slots:
                name, value, act = read_slot(slot)
                if value is not None and act is not None:
                    slot_names_with_values.add(name)
            slot_names_without_values = set(slot_values_keys).difference(slot_names_with_values)
            num_slots_with_values = len(slot_names_with_values)
            if self.split in ["train", "val"]:
                # Reducing num of slots without values
                num_slots_without_values = min(max(len(slot_names_with_values), 2), len(slot_names_without_values))
            else:
                num_slots_without_values = len(slot_names_without_values)
            total += num_slots_with_values + num_slots_without_values

            self.len_map[i] = total
            for j in range(last_total, total):
                self.item_idx_map[j] = i
            temp_idx = 0
            for name in sorted(slot_names_with_values):
                self.sample_slot_map[last_total + temp_idx] = slot_idxes[name]
                temp_idx += 1
            for name in sorted(random.sample(slot_names_without_values, num_slots_without_values)):
                self.sample_slot_map[last_total + temp_idx] = slot_idxes[name]
                temp_idx += 1


    def __getitem__(self, index):
        suite = self.get_suite(index)

        return suite

    def __len__(self):
        return self.len_map[len(self.index_mapper) - 1]

    def get_raw_image(self, index, sub_index, image_key="image"):
        index, caption_index = self.index_mapper[index]
        image_bytes = io.BytesIO(self.table[image_key][index][sub_index].as_py())
        image_bytes.seek(0)
        return Image.open(image_bytes).convert("RGB")

    def get_image(self, index, sub_index, image_key="image"):
        image = self.get_raw_image(index, sub_index, image_key=image_key)
        image_tensor = [tr(image) for tr in self.transforms]
        return {f"image_{sub_index}": image_tensor}

    def get_dst_text(self, raw_index):
        idx = self.item_idx_map[raw_index]
        index, caption_index = self.index_mapper[idx]
        raw_sample = self.all_texts[index][caption_index]
        split_idx = raw_sample.rindex(self.split_token) + len(self.split_token)
        # dst
        slots = extract(raw_sample, self.data_var_token)[0]
        if slots:
            slots = [s for s in slots.split(self.split_data_var) if not s.startswith('img_gt')]
        else:
            slots = []
        slot_idx = self.sample_slot_map[raw_index]
        slot_by_idx = slot_values_keys[slot_idx]
        has_value = False
        act2values = defaultdict(list)

        for slot in slots: # 如果slot_by_idx不在brief state上，act
            name, value, act = read_slot(slot)
            if name == slot_by_idx:
                has_value = has_value or value is not None
                if act is not None and value is not None:
                    act2values[act].append(value)
        act2values = sorted(act2values.items(), key=lambda x: act_order[x[0]]) # TODO 导致有多个action的只能学到第一个

        name = slot_by_idx
        #print("name", name) 
        sample_str = raw_sample[:split_idx]
        
        tokenized_slot = ['[CLS]'] + self.tokenizer.tokenize(name) + ['[SEP]']
        # tokenized_sample = self.tokenizer.tokenize(sample_str) + ['[SEP]']
        # tokenized_sample = tokenized_slot + tokenized_sample[-(512 - len(tokenized_slot)):]
        # tokenized_sample_tensor = torch.LongTensor(self.tokenizer.convert_tokens_to_ids(tokenized_sample))
        
        sample_context = self.tokenizer.decode(self.tokenizer.encode(sample_str, truncation=True, max_length=1024)[-(509 - len(tokenized_slot)):])
        # print(sample_context[:5])
        if sample_context[:5] == '[CLS]':
            sample_context = name + ' [SEP]' + sample_context[5:-6]
        else:
            sample_context = name + ' [SEP]' + sample_context[:-6]

        tokenized_sample_tensor = torch.LongTensor(self.tokenizer(sample_context)["input_ids"])

        encoding = self.tokenizer(
            sample_context,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )
        
        sample = {
            "text": (sample_context, encoding)
            # "input_ids" : tokenized_sample_tensor
        }
        # sample = {
        #     'input_ids': tokenized_sample_tensor,
        #     'token_type_ids': torch.cat([torch.zeros((len(tokenized_slot),), dtype=int), torch.ones((tokenized_sample_tensor.shape[0] - len(tokenized_slot),), dtype=int)]),
        #     'attention_mask': torch.ones_like(tokenized_sample_tensor)
        # }
        # span
        spans = set()
        if name in open_slots: #'telephone','name','open span','venueaddress','postcode','venuename','phone','venueneigh'
            if name == 'open span':
                if act2values:
                    for value_ in act2values[0][1]:
                        spans.add(value_)
            elif has_value:
                if name in matchable_slots:
                    for m in matchable_slots[name].finditer(sample_str.lower()):
                        spans.add(m[0])
                else:
                    if act2values:
                        for value_ in act2values[0][1]:
                            matches = match(sample_str.lower(), value_, thresh_abs=int(max(1, 0.35*len(value_))), thresh_r=0.55, text_len_delta=[int(max(-8, min(-1, -0.3*len(value_.split())))), int(min(8, max(1, 0.3*len(value_.split()))))], return_thresh=1, sorted=True)
                            if matches:
                                spans.add(matches[0][0])
                            # else:
                            #     print(sample_str)
                            #     print(value)
                            #     raise Exception()

        span_label = torch.zeros_like(tokenized_sample_tensor, dtype=int)
        tokenized_sample_str = ' '.join(self.tokenizer.tokenize('[CLS]'+sample_context+'[SEP]'))
        # open_span_count = 0
        for span in spans:
            tokenized_span = self.tokenizer.tokenize(span)
            tokenized_span_str = ' '.join(tokenized_span)
            idxs = [m.start() for m in re.finditer(re.escape(tokenized_span_str), tokenized_sample_str)]
            for j in idxs:
                # open_span_count += 1
                begin_idx = len(tokenized_sample_str[:j].split())
                span_label[begin_idx] = 2
                span_label[begin_idx + 1: begin_idx + len(tokenized_span)] = 1
        sample['span'] = span_label
        # print("span_label", span_label)
        # Action
        sample['action'] = act_ids[act2values[0][0]] if len(act2values) else -1

        # Gate
        sample['gate'] = int(has_value or (sample['action'] != -1 and (len(spans) > 0 or len(act2values) > 0)))

        # Slot
        sample['slot'] = slot_idx
        sample['slot value'] = -1 if (not act2values or slot_by_idx in open_slots) else slot_values[slot_by_idx].index(act2values[0][1][-1])

        # sample['slots'] = slot_options_lower
    
        sample['id'] = [self.item_idx_map[raw_index], slot_idx]

        sample['input_ids_len'] = tokenized_sample_tensor.shape[0]
        return sample


    def get_suite(self, index):
        result = None
        raw_index, index = index, self.item_idx_map[index]
        image_key = "image"
        while result is None:
            try:
                ret = dict()
                for sub_index in range(len(self.table["image"][index])):
                    ret.update(self.get_image(index, sub_index))
                if not self.image_only:
                    dst_sample = self.get_dst_text(raw_index)
                    ret.update(dst_sample)
                    # txt = self.get_text(index)
                    # ret.update({"replica": True if txt["cap_index"] > 0 else False})
                    # ret.update(txt)

                result = True
            except Exception as e:
                print(f"Error while read file idx {index} in {self.names[0]} -> {e}")
                index = random.randint(0, len(self.index_mapper) - 1)
        return ret

    def collate(self, batch, mlm_collator):
        batch_size = len(batch)
        keys = set([key for b in batch for key in b.keys()])
        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

        img_keys = [k for k in list(dict_batch.keys()) if "image" in k]
        img_sizes = list()

        for img_key in img_keys:
            img = dict_batch[img_key]
            img_sizes += [ii.shape for i in img if i is not None for ii in i]

        for size in img_sizes:
            assert (
                len(size) == 3
            ), f"Collate error, an image should be in shape of (3, H, W), instead of given {size}"

        if len(img_keys) != 0:
            max_height = max([i[1] for i in img_sizes])
            max_width = max([i[2] for i in img_sizes])

        for img_key in img_keys:
            img = dict_batch[img_key]
            view_size = 1

            new_images = [
                torch.zeros(batch_size, 3, max_height, max_width)
                for _ in range(view_size)
            ]

            for bi in range(batch_size):
                orig_batch = img[bi]
                for vi in range(view_size):
                    if orig_batch is not None:
                        orig = img[bi][vi]
                        new_images[vi][bi, :, : orig.shape[1], : orig.shape[2]] = orig

            dict_batch[img_key] = new_images

        txt_keys = [k for k in list(dict_batch.keys()) if "text" in k]
        # for dst
        for dst_key in ['gate', 'action', 'slot', 'slot value', 'id', 'input_ids_len']:
            dict_batch[dst_key] = torch.tensor(dict_batch[dst_key])
        dict_batch["span"] = pad_sequence(dict_batch["span"], batch_first=True, padding_value=-1)
        # print(dict_batch["span"].shape)

        if len(txt_keys) != 0:
            texts = [[d[0] for d in dict_batch[txt_key]] for txt_key in txt_keys]
            encodings = [[d[1] for d in dict_batch[txt_key]] for txt_key in txt_keys]
            draw_text_len = len(encodings)
            flatten_encodings = [e for encoding in encodings for e in encoding]
            flatten_mlms = mlm_collator(flatten_encodings)

            for i, txt_key in enumerate(txt_keys):
                texts, encodings = (
                    [d[0] for d in dict_batch[txt_key]],
                    [d[1] for d in dict_batch[txt_key]],
                )

                mlm_ids, mlm_labels = (
                    flatten_mlms["input_ids"][batch_size * (i) : batch_size * (i + 1)],
                    flatten_mlms["labels"][batch_size * (i) : batch_size * (i + 1)],
                )

                input_ids = torch.zeros_like(mlm_ids)
                attention_mask = torch.zeros_like(mlm_ids)
                for _i, encoding in enumerate(encodings):
                    _input_ids, _attention_mask = (
                        torch.tensor(encoding["input_ids"]),
                        torch.tensor(encoding["attention_mask"]),
                    )
                    input_ids[_i, : len(_input_ids)] = _input_ids
                    attention_mask[_i, : len(_attention_mask)] = _attention_mask

                dict_batch[txt_key] = texts
                dict_batch[f"{txt_key}_ids"] = input_ids
                dict_batch[f"{txt_key}_labels"] = torch.full_like(input_ids, -100)
                dict_batch[f"{txt_key}_ids_mlm"] = mlm_ids
                dict_batch[f"{txt_key}_labels_mlm"] = mlm_labels
                dict_batch[f"{txt_key}_masks"] = attention_mask

        return dict_batch
