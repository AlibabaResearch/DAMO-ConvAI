import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class RatMultiHeadedAttention(nn.Module):
    """
    Each head is a self-attention operation.
    self-attention refers to https://arxiv.org/pdf/1706.03762.pdf
    """
    def __init__(self, hidden_size, heads_num, relationship_number):
        super(RatMultiHeadedAttention, self).__init__()
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        self.per_head_size = hidden_size // heads_num
        self.linear_layers = nn.ModuleList([
                nn.Linear(hidden_size, hidden_size) for _ in range(3)
            ])
        
        if relationship_number != -1:
            assert relationship_number == self.per_head_size
            self.relationship_K_parameter = nn.Parameter(torch.nn.init.uniform_(torch.Tensor(heads_num, self.per_head_size)))
            self.relationship_V_parameter = nn.Parameter(torch.nn.init.uniform_(torch.Tensor(heads_num, self.per_head_size)))
            # cross mutli-head
            #self.relationship_K_parameter = nn.Parameter(torch.nn.init.uniform_(torch.Tensor(1, self.per_head_size)))
            #self.relationship_V_parameter = nn.Parameter(torch.nn.init.uniform_(torch.Tensor(1, self.per_head_size)))
            


        self.final_linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, key, value, query, relationship_matrix, dropout):
        """
        Args:
            key: [batch_size x seq_length x hidden_size]
            value: [batch_size x seq_length x hidden_size]
            query: [batch_size x seq_length x hidden_size]
        Returns:
            output: [batch_size x seq_length x hidden_size]
        """
        batch_size, seq_length, hidden_size = key.size()
        heads_num = self.heads_num
        per_head_size = self.per_head_size

        def shape(x):
            return x. \
                   contiguous(). \
                   view(batch_size, seq_length, heads_num, per_head_size). \
                   transpose(1, 2)

        def unshape(x):
            return x. \
                   transpose(1, 2). \
                   contiguous(). \
                   view(batch_size, seq_length, hidden_size)

        # torch.Size([1, 30, 43, 10]) torch.Size([1, 30, 43, 10]) torch.Size([1, 30, 43, 10])
        query, key, value = [l(x). \
                             view(batch_size, -1, heads_num, per_head_size). \
                             transpose(1, 2) \
                             for l, x in zip(self.linear_layers, (query, key, value))
                            ]
        #self.relationship_K_parameter [30, 10]
        


        # relation ship torch.Size([1, 43, 43, 10])
        relationship_matrix = relationship_matrix.unsqueeze(3).repeat(1, 1, 1, heads_num, 1)
        # ↑ torch.Size([1, 43, 43, 30, 10])
        
        kk = self.relationship_K_parameter.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch_size, seq_length, seq_length, 1, 1)
        #kk = self.relationship_K_parameter.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch_size, seq_length, seq_length,1,1)
        # ↑ torch.Size([1, 43, 43, 30, 10])
        vv = self.relationship_V_parameter.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch_size, seq_length, seq_length, 1, 1)
        #vv = self.relationship_V_parameter.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch_size, seq_length, seq_length,1,1)
        # ↑ torch.Size([1, 43, 43, 30, 10])

        relationship_K = relationship_matrix * kk
        relationship_V = relationship_matrix * vv
        #print(relationship_K[0,17,7,:,:])
        #print(relationship_V[0,17,7,:,:])

        # [1, 43, 43 ,30 ,10] -> [1, 30, 43, 43, 10]
        relationship_K = relationship_K.transpose(1, 3)
        relationship_V = relationship_V.transpose(1, 3)

        #[1, 30, 43, 10] -> [1, 30, 43, 43, 10]
        query = query.unsqueeze(3).repeat(1, 1, 1, seq_length, 1)
        key = key.unsqueeze(2).repeat(1, 1, seq_length, 1, 1) 
        
        #↓ [1, 30, 43, 43]
        #scores = (query*(relationship_K)).sum(dim=-1)
        scores = (query*(key+relationship_K)).sum(dim=-1)
        
        scores = scores / math.sqrt(float(per_head_size))
 
        probs = nn.Softmax(dim=-1)(scores)
        
        """
        save_dct = {}
        for i in range(30):
            attention_map = probs[0, i, :, :]
            save_dct[i] = attention_map.cpu().numpy()
        import pickle as pkl
        #print(save_dct)
        pkl.dump(save_dct, open("./save_att_relation.pkl", "wb"))
        

        print("finish")
        time.sleep(5)
        """
        #probs = F.dropout(probs, p=dropout)
        
        probs = probs.unsqueeze(4).repeat(1, 1, 1, 1, per_head_size)
        value = value.unsqueeze(2).repeat(1, 1, seq_length, 1, 1)
        
        output = (probs*(value+relationship_V)).sum(dim=-2)
        
        output = unshape(output)
        output = self.final_linear(output)
        return output
