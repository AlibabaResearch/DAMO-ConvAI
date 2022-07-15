import torch

def preprocess(utterances, sql, tokenizer):
    text = ""
    for u in utterances:
        for t in u.split(' '):
            text = text + ' ' + t.strip()
    sql = sql.strip()
    sql = sql.replace(".", " ")
    sql = sql.replace("_", " ")
    l = []
    for char in sql:
        if char.isupper():
            l.append(' ')
        l.append(char)
    sql = ''.join(l)
    sql = ' '.join(sql.split())
    # input:
    # [CLS] utterance [SEP] sql [SEP]
    token_encoding = tokenizer.encode_plus(text, sql, max_length=128, pad_to_max_length=True)
    tokenized_token = token_encoding['input_ids']
    attention_mask = token_encoding['attention_mask']
    tokens_tensor = torch.tensor(tokenized_token)
    attention_mask_tensor = torch.tensor(attention_mask)
    return tokens_tensor, attention_mask_tensor
