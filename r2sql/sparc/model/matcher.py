import torch
import torch.nn as nn
import torch.nn.functional as F


class Matcher(nn.Module):
    " nn Matcher for utterance with table column embedding"
    def __init__(self, input_size=950, hidden_size=128, output_size=2):
        super(Matcher, self).__init__()
        self.fc_1 = nn.Linear(input_size, hidden_size).cuda()
        self.fc_2 = nn.Linear(hidden_size, output_size).cuda()

    def load_meta(self, input_schema, schema_states):
        self.table_name = input_schema.column_names_embedder_input
        self.table_name_origin = input_schema.table_schema['table_names']
        self.table_vector = self.split_schema_states(schema_states)

    def gen_labels(self, gold_query):
        shot = set()
        for tok in gold_query:
            if '.' in tok:
                gold_table = tok.split('.')[0].strip()
                if '_' in gold_table:
                    gold_table = ' '.join(gold_table.split('_'))
                shot.add(gold_table) 
        labels = []
        for i in self.table_name_origin:
            if i in shot:
                labels.append(1)
            else:
                labels.append(0)
        labels = torch.Tensor(labels).long().cuda()
        return labels, shot, self.table_name_origin

    def split_schema_states(self, schema_states):
        table_idx_dct = {}
        for idx, name in enumerate(self.table_name):
            # skip *
            if '.' not in name:
                continue
            t_name = name.split('.')[0].strip()
            if t_name not in table_idx_dct:
                table_idx_dct[t_name] = [idx]
            else:
                table_idx_dct[t_name].append(idx)
        flag = 0
        for name in self.table_name_origin:
            idx_lst = table_idx_dct[name]
            avg_embed = torch.zeros_like(schema_states[0])
            for i in idx_lst:
                avg_embed += schema_states[i]
            avg_embed /= len(idx_lst)
            avg_embed = avg_embed.unsqueeze(0)
            if flag:
                table_vector = torch.cat([table_vector, avg_embed], dim=0)
            else:
                table_vector = avg_embed.clone()
                flag = 1
        return table_vector

    
    def forward(self, utterance_vector):
        num_header = self.table_vector.size()[0]
        # repeat utterance_vecotr for match table_vector size
        x = torch.cat([utterance_vector.repeat((num_header, 1)), self.table_vector], dim=1)
        x = self.fc_1(x)
        x = F.relu(x)
        x = self.fc_2(x)
        x = F.log_softmax(x, dim=1)
        return x
