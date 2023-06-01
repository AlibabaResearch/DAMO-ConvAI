from .base_dataset import BaseDataset
from random import shuffle , randint
from random import random as rand
import torch
import math

class BaseGenDataset(BaseDataset):
    def __init__(self, 
            *args, 
            split="", 
            use_segment_ids=False , 
            mask_prob=0.15 , 
            max_pred_len=20,
            max_source_len=200,
            whole_word_masking=True,
            mask_source_words=False, 
            source_column_name="source",
            target_column_name="target",
            **kwargs):
            
        super().__init__(*args, **kwargs)
        self.source_sents = self.table[source_column_name].to_pandas().tolist()
        self.target_sents = self.table[target_column_name].to_pandas().tolist()

        self.index_mapper = dict()
       
        self.max_image_cls_len = self.max_image_len + 1 # cls token
        self.use_segment_ids = use_segment_ids
        self.mask_prob = mask_prob
        self.max_pred_len = max_pred_len
        self.max_source_len = max_source_len
        self.whole_word_masking = whole_word_masking
        self.mask_source_words = mask_source_words
        self.max_total_len = self.max_text_len + self.max_image_cls_len
        self._tril = torch.tril(torch.ones((self.max_total_len , self.max_total_len),dtype=torch.long))
        self.vocab_words = None#list(self.tokenizer.vocab.keys())
        for i in range(len(self.source_sents)):
            self.index_mapper[i] = (i,i)

    def __len__(self):
        return len(self.source_sents)

    def get_random_word(self):
        return self.vocab_words[randint(0,len(self.vocab_words)-1)]

    def get_sep_token(self):
        raise NotImplementedError("set SEP word")
    
    def truncate_tokens(self, tokens_a , tokens_b):
        sep_len = len(self.tokenizer.tokenize(self.get_sep_token()))
        if len(tokens_a) + len(tokens_b) + 2*sep_len + 1 > self.max_text_len:
            while len(tokens_a) + len(tokens_b) + 2*sep_len + 1 > self.max_text_len:
                if len(tokens_a) + 1 + sep_len> self.max_source_len:
                    tokens_a = tokens_a[1:]
                else:
                    tokens_b = tokens_b[:-1]
        return tokens_a , tokens_b

    def _suite_for_test(self,index):
        source = self.source_sents[index]
        SEP = self.get_sep_token()
        if isinstance(source, list):
            source = ' '.join(source)

        text = source + SEP
        #self.SEP可能被子类修改
        sep_len = len(self.tokenizer.tokenize(SEP))
        tokens = self.tokenizer.tokenize(source)
        if len(tokens) + 1 + sep_len> self.max_source_len:
            tokens = tokens[-(self.max_source_len-1-sep_len):]
        tokens_a = ['[CLS]'] + tokens + ([SEP] if sep_len>0 else [])
        
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens_a)
        position_ids = []
        for i in range(len(tokens_a)):
            position_ids.append(i)

        max_len = self.max_text_len + self.max_image_cls_len
        input_mask = torch.zeros(max_len , max_len , dtype=torch.long)

        #attention to image and source
        len_image_a = self.max_image_cls_len + len(input_ids)
        input_mask[:, :len_image_a].fill_(1)

        ret =  {
            "text": text,
            "input_mlm":(input_ids, input_ids),
            "attention_masks": input_mask,
            "position_ids":position_ids,
            "img_index": index,
            "cap_index": index,
            "raw_index": index,
        }
        ret.update(self.get_image(index))
        if self.use_segment_ids:
            segment_ids = [0] * len(input_ids)
            ret["segment_ids"] = segment_ids
        return ret        

    def _suite_for_train_or_val(self,index):
        source = self.source_sents[index]
        SEP = self.get_sep_token()
        if isinstance(source , list):
            source = ' '.join(source)
        target = self.target_sents[index]
        sep_len = len(self.tokenizer.tokenize(SEP))
        text = source + SEP + target + SEP
        tokens_a = self.tokenizer.tokenize(source)
        tokens_b = self.tokenizer.tokenize(target)
        tokens_a , tokens_b = self.truncate_tokens(tokens_a , tokens_b)

        tokens =['[CLS]'] + tokens_a + ([SEP] if sep_len>0 else [])  + tokens_b + ([SEP] if sep_len>0 else [])#torch.cat((tokens_a, tokens_b), dim = 1)
        input_labels = self.tokenizer.convert_tokens_to_ids(tokens)
        segment_ids = (len(tokens_a)+ sep_len + 1) * [0] + (self.max_text_len - 1 - sep_len - len(tokens_a)) * [1]

        effective_length = (len(tokens_b) + sep_len) if not self.mask_source_words else (len(tokens_a) + len(tokens_b) + 2*sep_len)
        n_preds = min(math.ceil(effective_length * self.mask_prob),self.max_pred_len)
        
        cand_pos = list()
        for i,tk in enumerate(tokens):
            if (i >= len(tokens_a)+sep_len) and (tk != '[CLS]'):
                cand_pos.append(i)
            elif self.mask_source_words and ( i < len(tokens_a)+sep_len):
                cand_pos.append(i)
    
        # cand_pos = list(range(len(tokens_a) + 1, len(tokens)))
        shuffle(cand_pos)
        max_cand_pos = max(cand_pos)
        masked_pos = set()
        
        for pos in cand_pos:
            if pos in masked_pos:
                continue
            if len(masked_pos) >= n_preds:
                break

            def _expand_whole_word(st, end):
                new_st, new_end = st, end
                while (new_st >= 0) and tokens[new_st].startswith('##'):
                    new_st -= 1
                while (new_end < len(tokens)) and tokens[new_end].startswith('##'):
                    new_end += 1
                return new_st, new_end

            if self.whole_word_masking:
                st_pos, end_pos = _expand_whole_word(pos,pos+1)
            else:
                st_pos, end_pos = pos , pos+1
            
            for mp in range(st_pos , end_pos):
                if (0 < mp <= max_cand_pos):
                    masked_pos.add(mp)

        masked_pos = list(masked_pos)
        if len(masked_pos) > n_preds:
            shuffle(masked_pos)
            masked_pos = masked_pos[:n_preds]

        masked_weights = [1] * len(masked_pos)
        masked_tokens = [tokens[pos] for pos in masked_pos]
        for pos in masked_pos:
            if rand() < 0.8:
                tokens[pos] = '[MASK]'
            elif rand() < 0.5:
                tokens[pos] = self.get_random_word()
        
        
        max_len = self.max_text_len + self.max_image_cls_len
        input_mask = torch.zeros(max_len , max_len , dtype=torch.long)

        #attention to image and source
        len_image_a = self.max_image_cls_len + len(tokens_a) + sep_len + 1
        input_mask[:, :len_image_a].fill_(1)

        #attention of target , left only
        second_st,second_end = self.max_image_cls_len + len(tokens_a) + sep_len + 1, self.max_image_cls_len + len(tokens)
        input_mask[second_st:second_end , second_st:second_end].copy_(
            self._tril[:second_end-second_st, :second_end-second_st]
        )
        
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        n_pads = self.max_text_len - len(input_ids)
        input_ids.extend([0] * n_pads)
        input_labels.extend([0] * n_pads)
        masked_ids = self.tokenizer.convert_tokens_to_ids(masked_tokens)

        if len(masked_ids) < self.max_pred_len:
            masked_ids += [-100] * (self.max_pred_len - len(masked_ids))
        if len(masked_pos) < self.max_pred_len:
            masked_pos += [0] * (self.max_pred_len - len(masked_pos))
        if len(masked_weights) < self.max_pred_len:
            masked_weights += [0] * (self.max_pred_len - len(masked_weights))

        ret =  {
            "text": text,
            "input_mlm": (input_ids , input_labels),
            "target_mlm": (masked_ids ,masked_pos , masked_weights),
            "attention_masks": input_mask,
            "img_index": index,
            "cap_index": index,
            "raw_index": index,
        }
        ret.update(self.get_image(index))
        if self.use_segment_ids:
            ret["segment_ids"] = segment_ids
        return ret


    def __getitem__(self, index):
        if self.vocab_words is None:
            self.vocab_words = list(self.tokenizer.vocab.keys())
        suite = self.get_suite(index)
        return suite

    def get_suite(self,index):
        result = None
        while result is None:
            try:
                if self.split == 'test':
                    ret = self._suite_for_test(index)
                else :
                    ret = self._suite_for_train_or_val(index)

                result = True
            except (Exception,OSError) as e:
                print(f"Error while read file idx {index} in {self.names[0]} -> {e}")
                index = randint(0, len(self.source_sents) - 1)
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
            view_size = len(img[0])

            new_images = [
                torch.zeros(batch_size, 3, max_height, max_width)
                for _ in range(view_size)
            ]

            for bi in range(batch_size):
                # print(batch[bi])
                orig_batch = img[bi]
                for vi in range(view_size):
                    if orig_batch is None:
                        new_images[vi][bi] = None
                    else:
                        orig = img[bi][vi]
                        new_images[vi][bi, :, : orig.shape[1], : orig.shape[2]] = orig

            dict_batch[img_key] = new_images

        # txt_keys = ["text"]#[k for k in list(dict_batch.keys()) if "text" in k]
        has_segment_ids = self.use_segment_ids and "segment_ids" in dict_batch
        texts = dict_batch["text"]
        text_ids = [d[0] for d in dict_batch["input_mlm"]]
        text_labels = [d[1] for d in dict_batch["input_mlm"]]
        attention_mask = [atm for atm in dict_batch["attention_masks"]]


        do_mlm = True
        if has_segment_ids: segment_ids = dict_batch["segment_ids"]

        '''
            如果是test , 那么需要根据batch信息, 补齐text_id的长度
        '''
        if self.split == 'test':
            do_mlm = False
            position_ids = dict_batch['position_ids']
            max_source_len = max([len(text_id) for text_id in text_ids])
            for idx in range(len(text_ids)):
                raw_len = len(text_ids[idx])
                text_ids[idx].extend([0]*(max_source_len - len(text_ids[idx])))
                position_ids[idx].extend([0]*(max_source_len - len(position_ids[idx])))
                for i in range(max_source_len , self.max_text_len):
                    position_ids[idx].append(i - max_source_len + raw_len)
                if has_segment_ids : segment_ids[idx].extend([0] * (max_source_len - len(segment_ids[idx])) + [1]*(self.max_text_len - max_source_len))
                second_st , second_end = self.max_image_cls_len + max_source_len , self.max_image_cls_len + self.max_text_len
                attention_mask[idx][second_st:second_end , second_st:second_end].copy_(
                    self._tril[:second_end-second_st , :second_end-second_st]
                )
            dict_batch["position_ids"] = torch.tensor(position_ids)
        else :
            target_masked_ids = [d[0] for d in dict_batch["target_mlm"]]
            target_masked_pos = [d[1] for d in dict_batch["target_mlm"]]
            target_masked_weights = [d[2] for d in dict_batch["target_mlm"]]
            dict_batch["target_masked_ids"] = torch.tensor(target_masked_ids)
            dict_batch["target_masked_pos"] = torch.tensor(target_masked_pos)
            dict_batch["target_masked_weights"] = torch.tensor(target_masked_weights)
            
        mlm_suffix = "_mlm" if do_mlm else ""

        try:
            dict_batch["text"] = texts
            dict_batch[f"text_ids{mlm_suffix}"] = torch.tensor(text_ids)
            dict_batch[f"text_labels{mlm_suffix}"] = torch.tensor(text_labels)
            dict_batch["attention_masks"] = torch.stack(attention_mask,dim=0)
            dict_batch["text_masks"] = dict_batch["attention_masks"][:,self.max_image_cls_len:,self.max_image_cls_len:]
            if has_segment_ids:
                dict_batch["text_segment_ids"] = torch.tensor(segment_ids)
        except Exception as e:
            print(e)

        return dict_batch