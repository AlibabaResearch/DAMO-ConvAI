from .base_dataset import BaseDataset
import random
import torch

class ImageChatDataset(BaseDataset):
    def __init__(self, 
            *args, 
            split="", 
            **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        if split == "train":
            names = [f"imagechat_train_split_{i}" for i in range(25)] 
        elif split == "val":
            names = [f"imagechat_valid_split_{i}" for i in range(1)]
        elif split == "test":
            names = [f"imagechat_test_split_{i}" for i in range(2)]

        super().__init__(*args, **kwargs, names=names, text_column_name="answer")
        self.all_texts = self.answers =self.table['answer'].to_pandas().tolist()
        self.historys = self.table['history'].to_pandas().tolist()
        self.styles = self.table['style'].to_pandas().tolist()
        self.candidates = self.table['candidates'].to_pandas().tolist()
        self.img_hash = self.table['image_hash'].to_pandas().tolist()
        self.index_mapper = dict()
        
        self.SEP = '[SEP]'
        for i in range(len(self.answers)):
            self.index_mapper[i] = (i,i)


    def __len__(self):
        return len(self.answers)
    
    def concat(self,history,style,answer):
        history = history.tolist() + [style+":"+answer]
        raw_history = history
        tokens_type_indexs = [0]
        current_turn = 0
        dialog = ''
        len_tokens = 1
        history.reverse()
        for i,utter in enumerate(history):
            if len_tokens + len(self.tokenizer.tokenize(utter)) + 1 <= self.max_text_len:
                len_tokens += (len(self.tokenizer.tokenize(utter)) + 1)
            else:
                history = history[:i]
                break

        history.reverse()
        for i,utter in enumerate(history):
            curr_utter = utter + (self.SEP if i+1!=len(history) else ' ')
            tokens_type_indexs = tokens_type_indexs + (len(self.tokenizer.tokenize(utter))+1) * [current_turn] 
            dialog = dialog + curr_utter
            current_turn ^= 1
        # dialog_token_len = 0 
        # for utter in raw_history:
        #     dialog_token_len += len(self.tokenizer.tokenize(utter))
        # print(f"len of history:{len(raw_history)}, ,dialog:{dialog}, history:{history}, raw history:{raw_history},raw history token len:{dialog_token_len}")
        return dialog , tokens_type_indexs

    def pad_to_length(self, data, length):
        return data + [0]*(length-len(data))

    def get_text(self,index):
        answer = self.answers[index]
        style = self.styles[index]
        history = self.historys[index]
        text , segments = self.concat(history,style,answer)
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True
        )

        ret =  {
            "text": (text, encoding),
            "img_index": index,
            "cap_index": index,
            "raw_index": index,
        }
        if self.use_segment_ids:
            ret.update({
                'segment_ids':self.pad_to_length(segments,self.max_text_len)
            })
        
        return ret    

    def get_false_texts(self,index,draw_false_text):
        ret = dict()
        style = self.styles[index]
        history = self.historys[index]
        candidates = self.candidates[index] if len(self.candidates[index]) > draw_false_text else self.answers
        inds = list(range(1,len(candidates)))
        for i in range(draw_false_text):
            rand_index = random.choice(inds)
            answer = candidates[rand_index]
            text,segments = self.concat(history,style,answer)
            encoding = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_text_len,
                return_special_tokens_mask=True
            )

            ret.update({
                f'false_text_{i}':(text,encoding),
            })
            if self.use_segment_ids:
                ret.update({f'false_{i}_segment_ids':self.pad_to_length(segments,self.max_text_len)})
            inds.remove(rand_index)
        return ret

    def __getitem__(self, index):
        suite = self.get_suite(index)
        return suite

    def get_suite(self,index):
        result = None
        while result is None:
            try:
                ret = dict()
                ret.update(self.get_image(index))
                if not self.image_only:
                    txt = self.get_text(index)
                    # ret.update({"replica": True if txt["cap_index"] > 0 else False})
                    ret.update({"replica": True})
                    ret.update(txt)
                    ret.update(self.get_false_texts(index,self.draw_false_text))
                for i in range(self.draw_false_image):
                    ret.update(self.get_false_image(i))
                result = True
            except (Exception,OSError) as e:
                print(f"Error while read file idx {index} in {self.names[0]} -> {e}")
                index = random.randint(0, len(self.answers) - 1)
        return ret