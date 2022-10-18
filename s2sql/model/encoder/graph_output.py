#coding=utf8
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from model.model_utils import Registrable
from model.encoder.functions import scaled_exp, div_by_z, src_dot_dst

class ScoreFunction(nn.Module):

    def __init__(self, hidden_size, mlp=1, method='biaffine'):
        super(ScoreFunction, self).__init__()
        assert method in ['dot', 'bilinear', 'affine', 'biaffine']
        self.mlp = int(mlp)
        self.hidden_size = hidden_size // self.mlp
        if self.mlp > 1: # use mlp to perform dim reduction
            self.mlp_q = nn.Sequential(nn.Linear(hidden_size, self.hidden_size), nn.Tanh())
            self.mlp_s = nn.Sequential(nn.Linear(hidden_size, self.hidden_size), nn.Tanh())
        self.method = method
        if self.method == 'bilinear':
            self.W = nn.Linear(self.hidden_size, self.hidden_size)
        elif self.method == 'affine':
            self.affine = nn.Linear(self.hidden_size * 2, 1)
        elif self.method == 'biaffine':
            self.W = nn.Linear(self.hidden_size, self.hidden_size)
            self.affine = nn.Linear(self.hidden_size * 2, 1)

    def forward(self, context, node):
        """
        @args:
            context(torch.FloatTensor): num_nodes x hidden_size
            node(torch.FloatTensor): num_nodes x hidden_size
        @return:
            scores(torch.FloatTensor): num_nodes
        """
        if self.mlp > 1:
            context, node = self.mlp_q(context), self.mlp_s(node)
        if self.method == 'dot':
            scores = (context * node).sum(dim=-1)
        elif self.method == 'bilinear':
            scores = (context * self.W(node)).sum(dim=-1)
        elif self.method == 'affine':
            scores = self.affine(torch.cat([context, node], dim=-1)).squeeze(-1)
        elif self.method == 'biaffine':
            scores = (context * self.W(node)).sum(dim=-1)
            scores += self.affine(torch.cat([context, node], dim=-1)).squeeze(-1)
        else:
            raise ValueError('[Error]: Unrecognized score function method %s!' % (self.method))
        return scores

@Registrable.register('without_pruning')
class GraphOutputLayer(nn.Module):

    def __init__(self, args):
        super(GraphOutputLayer, self).__init__()
        self.hidden_size = args.gnn_hidden_size

    def forward(self, inputs, batch):
        """ Re-scatter data format:
                inputs: sum(q_len + t_len + c_len) x hidden_size
                outputs: bsize x (max_q_len + max_t_len + max_c_len) x hidden_size
        """
        outputs = inputs.new_zeros(len(batch), batch.mask.size(1), self.hidden_size)
        outputs = outputs.masked_scatter_(batch.mask.unsqueeze(-1), inputs)
        if self.training:
            return outputs, batch.mask, torch.tensor(0., dtype=torch.float).to(outputs.device)
        else:
            return outputs, batch.mask

@Registrable.register('with_pruning')
class GraphOutputLayerWithPruning(nn.Module):

    def __init__(self, args):
        super(GraphOutputLayerWithPruning, self).__init__()
        self.hidden_size = args.gnn_hidden_size
        self.graph_pruning = GraphPruning(self.hidden_size, args.num_heads, args.dropout, args.score_function)

    def forward(self, inputs, batch):
        outputs = inputs.new_zeros(len(batch), batch.mask.size(1), self.hidden_size)
        outputs = outputs.masked_scatter_(batch.mask.unsqueeze(-1), inputs)

        if self.training:
            g = batch.graph
            question = inputs.masked_select(g.question_mask.unsqueeze(-1)).view(-1, self.hidden_size)
            schema = inputs.masked_select(g.schema_mask.unsqueeze(-1)).view(-1, self.hidden_size)
            loss = self.graph_pruning(question, schema, g.gp, g.node_label)
            return outputs, batch.mask, loss
        else:
            return outputs, batch.mask

class GraphPruning(nn.Module):

    def __init__(self, hidden_size, num_heads=8, feat_drop=0.2, score_function='affine'):
        super(GraphPruning, self).__init__()
        self.hidden_size = hidden_size
        self.node_mha = DGLMHA(hidden_size, hidden_size, num_heads, feat_drop)
        self.node_score_function = ScoreFunction(self.hidden_size, mlp=2, method=score_function)
        self.loss_function = nn.BCEWithLogitsLoss(reduction='sum')

    def forward(self, question, schema, graph, node_label):
        node_context = self.node_mha(question, schema, graph)
        node_score = self.node_score_function(node_context, schema)
        loss = self.loss_function(node_score, node_label)
        return loss

class DGLMHA(nn.Module):
    """ Multi-head attention implemented with DGL lib
    """
    def __init__(self, hidden_size, output_size, num_heads=8, feat_drop=0.2):
        super(DGLMHA, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_heads = num_heads
        self.d_k = self.hidden_size // self.num_heads
        self.affine_q, self.affine_k, self.affine_v = nn.Linear(self.output_size, self.hidden_size),\
            nn.Linear(self.hidden_size, self.hidden_size, bias=False), nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.affine_o = nn.Linear(self.hidden_size, self.output_size)
        self.feat_dropout = nn.Dropout(p=feat_drop)

    def forward(self, context, node, g):
        q, k, v = self.affine_q(self.feat_dropout(node)), self.affine_k(self.feat_dropout(context)), self.affine_v(self.feat_dropout(context))
        with g.local_scope():
            g.nodes['schema'].data['q'] = q.view(-1, self.num_heads, self.d_k)
            g.nodes['question'].data['k'] = k.view(-1, self.num_heads, self.d_k)
            g.nodes['question'].data['v'] = v.view(-1, self.num_heads, self.d_k)
            out_x = self.propagate_attention(g)
        return self.affine_o(out_x.view(-1, self.hidden_size))

    def propagate_attention(self, g):
        # Compute attention score
        g.apply_edges(src_dot_dst('k', 'q', 'score'))
        g.apply_edges(scaled_exp('score', math.sqrt(self.d_k)))
        # Update node state
        g.update_all(fn.src_mul_edge('v', 'score', 'v'), fn.sum('v', 'wv'))
        g.update_all(fn.copy_edge('score', 'score'), fn.sum('score', 'z'), div_by_z('wv', 'z', 'o'))
        out_x = g.nodes['schema'].data['o']
        return out_x