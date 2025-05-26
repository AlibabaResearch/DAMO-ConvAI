import torch
import torch.nn as nn
import torch.distributed as torch_dist


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["avg", "avg_top2", "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        hidden_states = outputs.hidden_states

        # if self.pooler_type in ['cls_before_pooler', 'cls']:
        #     return last_hidden[:, 0]
        if self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError


class SimCSE(nn.Module):
    def __init__(self, sim_cse_config: dict):
        super(SimCSE, self).__init__()

        self.alpha_sim_cse = float(sim_cse_config.get('alpha_sim_cse', 0.5))
        self.sim_cse_temp = float(sim_cse_config.get('sim_cse_temp', 0.05))
        self.cos = nn.CosineSimilarity(dim=-1)
        self.pooler_type = sim_cse_config.get('pooler_type', 'avg')
        self.pooler = Pooler(self.pooler_type)

    def forward(self, enc_outs_1, enc_outs_2, attention_mask):
        """

        :param enc_outs_1:
        :param enc_outs_2:
        :param attention_mask: (bsz, enc_len, hsz)
        :return:
        """

        z1 = self.pooler(attention_mask, enc_outs_1)  # (bsz, hsz)
        z2 = self.pooler(attention_mask, enc_outs_2)  # (bsz, hsz)

        if torch_dist.is_initialized():
            # Dummy vectors for allgather
            z1_list = [torch.zeros_like(z1) for _ in range(torch_dist.get_world_size())]
            z2_list = [torch.zeros_like(z2) for _ in range(torch_dist.get_world_size())]
            # Allgather
            torch_dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
            torch_dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())

            # Since allgather results do not have gradients, we replace the
            # current process's corresponding embeddings with original tensors
            z1_list[torch_dist.get_rank()] = z1
            z2_list[torch_dist.get_rank()] = z2
            # Get full batch embeddings: (bsz x N, hsz)
            z1 = torch.cat(z1_list, 0)
            z2 = torch.cat(z2_list, 0)

        cos_sim = self.cos(z1.unsqueeze(1), z2.unsqueeze(0))  # (bsz x N, bsz x N)
        labels = torch.arange(cos_sim.size(0)).long().to(cos_sim.device)

        loss_fct = nn.CrossEntropyLoss()
        sim_cse_loss = loss_fct(cos_sim, labels)

        sim_cse_loss = self.alpha_sim_cse * sim_cse_loss
        return sim_cse_loss