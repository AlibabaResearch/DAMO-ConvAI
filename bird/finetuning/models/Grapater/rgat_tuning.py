import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F
from ..model_utils import FFN
from ..unified.functions import *
import pdb

class RGAT_Tuning(nn.Module):

    def __init__(self, config):
        super(RGAT_Tuning, self).__init__()
        self.rgat_layer = RGAT_Layer(config.d_model, config.d_model, num_heads=1, feat_drop=0.2)

    def forward(self, hidden_states, graph_batch, relation_embedding):
        '''
        :param hidden_states: [bsz x input_max_length x d_model]
        :return: hidden states with graph caption, while keeping the other reps
        '''
        for i, graph in enumerate(graph_batch):
            hidden_states[i] = self.graph_caption_one(hidden_states[i], graph['graph'], \
                                                      graph['edges'], relation_embedding)

        return hidden_states

    def graph_caption_one1(self, hidden_state, graph_node_idx, graph, edges, relation_emb):
        '''
        :param hidden_state: input_max_length x d_model for each bsz
        :param graph_node_idx: 1 x input_max_length: index of hidden states -> graph node index
        :param graph: graph_batch[index]['big_g']
        :param edges: graph_batch[index]['big_edges']
        :return: hidden state with rgated elements
        '''
        graph_node_idx_select = torch.tensor([idx for idx in graph_node_idx if idx >= 0], device=graph_node_idx.device)
        node_feats = hidden_state[graph_node_idx_select]
        edge_feats = relation_emb(edges)

        struct_rep, edge_feats = self.rgat_layer(node_feats, edge_feats, graph)
        hidden_state[graph_node_idx_select] = struct_rep


        return hidden_state

    def graph_caption_one(self, hidden_state, graph, edges, relation_embedding):
        
        num_nodes = graph.number_of_nodes()
        node_feats = hidden_state[:num_nodes]
        edge_feats = relation_embedding(edges)
        

        struct_rep, edge_feats = self.rgat_layer(node_feats, edge_feats, graph)
        hidden_state[:num_nodes] = struct_rep
        

        return hidden_state

class RGAT_Layer(nn.Module):

    def __init__(self, ndim, edim, num_heads=1, feat_drop=0.2):
        super(RGAT_Layer, self).__init__()
        self.ndim, self.edim = ndim, edim
        self.num_heads = num_heads
        dim = max([ndim, edim])
        self.d_k = dim // self.num_heads
        self.affine_q, self.affine_k, self.affine_v = nn.Linear(self.ndim, dim), \
            nn.Linear(self.ndim, dim, bias=False), nn.Linear(self.ndim, dim, bias=False)
        self.affine_o = nn.Linear(dim, self.ndim)
        self.layernorm = nn.LayerNorm(self.ndim)
        self.feat_dropout = nn.Dropout(p=feat_drop)
        self.ffn = FFN(self.ndim)

    def forward(self, x, lgx, graph):
        """ @Params:
                x: node feats, num_nodes x ndim
                lgx: edge feats, num_edges x edim
                g: dgl.graph
        """
        # set the same device:
        g = graph.to(x.device)
        # pre-mapping q/k/v affine
        q, k, v = self.affine_q(self.feat_dropout(x)), self.affine_k(self.feat_dropout(x)), self.affine_v(self.feat_dropout(x))
        e = lgx.view(-1, self.num_heads, self.d_k) if lgx.size(-1) == q.size(-1) else \
            lgx.unsqueeze(1).expand(-1, self.num_heads, -1)
        with g.local_scope():
            g.ndata['q'], g.ndata['k'] = q.view(-1, self.num_heads, self.d_k), k.view(-1, self.num_heads, self.d_k)
            g.ndata['v'] = v.view(-1, self.num_heads, self.d_k)
            g.edata['e'] = e
            out_x = self.propagate_attention(g)

        out_x = self.layernorm(x + self.affine_o(out_x.view(-1, self.num_heads * self.d_k)))
        out_x = self.ffn(out_x)

        return out_x, lgx

    def propagate_attention(self, g):
        # Compute attention score
        g.apply_edges(src_sum_edge_mul_dst('k', 'q', 'e', 'score'))
        g.apply_edges(scaled_exp('score', math.sqrt(self.d_k)))
        # Update node state
        g.update_all(src_sum_edge_mul_edge('v', 'e', 'score', 'v'), fn.sum('v', 'wv'))
        g.update_all(fn.copy_edge('score', 'score'), fn.sum('score', 'z'), div_by_z('wv', 'z', 'o'))
        out_x = g.ndata['o']
        return out_x


