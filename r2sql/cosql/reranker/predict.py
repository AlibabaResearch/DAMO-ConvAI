import torch

from transformers import BertTokenizer, RobertaTokenizer

from .model import ReRanker
from .utils import preprocess

class Ranker:
    def __init__(self, model_path, base_model='roberta'):
        self.model = ReRanker(base_model=base_model)
        self.model.load_state_dict(torch.load(model_path))
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base', max_len=512)
        print("load reranker model from ", model_path)
        
    def get_score(self, utterances, sql):
        tokens_tensor, attention_mask_tensor = preprocess(utterances, sql, self.tokenizer)
        tokens_tensor = tokens_tensor.unsqueeze(0)
        attention_mask_tensor = attention_mask_tensor.unsqueeze(0)
        output = self.model(input_ids=tokens_tensor, attention_mask=attention_mask_tensor)
        output = output.squeeze(dim=-1)
        return output.item()

    def get_score_batch(self, utterances, sqls):
        assert isinstance(sqls, list), "only support sql list input"
        tokens_tensor_batch, attention_mask_tensor_batch = preprocess(utterances, sqls[0], self.tokenizer)
        tokens_tensor_batch = tokens_tensor_batch.unsqueeze(0)
        attention_mask_tensor_batch = attention_mask_tensor_batch.unsqueeze(0)
        if len(sqls) > 1:
            for s in sqls[1:]:
                new_token, new_mask = preprocess(utterances, s, self.tokenizer)
                tokens_tensor_batch = torch.cat([tokens_tensor_batch, new_token.unsqueeze(0)], dim=0)
                attention_mask_tensor_batch = torch.cat([attention_mask_tensor_batch, new_mask.unsqueeze(0)])
        output = self.model(input_ids=tokens_tensor_batch, attention_mask=attention_mask_tensor_batch)
        output = output.view(-1)
        ret = []
        for i in output:
            ret.append(i.item())
        return ret

if __name__ == '__main__':
    utterances_1 = ["What are all the airlines ?", "Of these , which is Jetblue Airways ?", "What is the country corresponding it ?"]
    sql_1 = "select Country from airlines where Airline = 1 select from airlines select * from airlines where Airline = 1"
    utterances_2 = ["What are all the airlines ?", "Of these , which is Jetblue Airways ?", "What is the country corresponding it ?"]
    sql_2 = "select T2.Countryabbrev from airlines as T1 join airports as T2 where T1.Airline = 1"
    sql_lst = ["select *", "select T2.Countryabbrev from airlines as T1 join airports as T2 where T1.Airline = 1"]
    utterances_test = ['Show the document name for all documents .', 'Also show their description .']
    sql_test = ['select Document_Name , Document_Description from Documents', 'select Document_Name , Document_Description from Documents select Document_Name from Documents', 'select Document_Name , Document_Name , Document_Description from Documents', 'select Document_ID , Document_Description from Documents', 'select Document_Name from Documents']

    ranker = Ranker()
    #print("utterance : ", utterances_1)
    #print("sql : ", sql_1)
    #print(ranker.get_score(utterances_1, sql_1))
    #print("utterance : ", utterances_2)
    #print("sql : ", sql_2)
    #print(ranker.get_score(utterances_2, sql_2))
    print(ranker.get_score_batch(utterances_test, sql_test))
