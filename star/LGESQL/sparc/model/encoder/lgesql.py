#coding=utf8
import copy, math
import torch, dgl
import torch.nn as nn
import dgl.function as fn
from model.model_utils import Registrable, FFN
from model.encoder.rgatsql import RGATLayer, MultiViewRGATLayer
from model.encoder.functions import *

@Registrable.register('lgesql')
class LGESQL(nn.Module):
    """ Compared with RGAT, we utilize a line graph to explicitly model the propagation among edges:
    1. aggregate info from in-edges
    2. aggregate info from src nodes
    """
    def __init__(self, args):
        super(LGESQL, self).__init__()
        self.num_layers, self.num_heads = args.gnn_num_layers, args.num_heads
        self.relation_share_heads = args.relation_share_heads
        self.graph_view = args.local_and_nonlocal
        self.ndim = args.gnn_hidden_size # node feature dim
        self.edim = self.ndim // self.num_heads if self.relation_share_heads else \
            self.ndim // 2 if self.graph_view == 'mmc' else self.ndim
        self.relation_num = args.relation_num
        self.relation_embed = nn.Embedding(self.relation_num, self.edim)
        self.gnn_layers = nn.ModuleList([
            DualRGATLayer(self.ndim, self.edim, num_heads=args.num_heads, feat_drop=args.dropout, graph_view=self.graph_view)
            for _ in range(self.num_layers)])

    def forward(self, x, batch):
        global_lgx = self.relation_embed(batch.graph.global_edges)
        mask = batch.graph.local_mask
        local_lgx = global_lgx[mask]
        local_g, global_g, lg = batch.graph.local_g, batch.graph.global_g, batch.graph.lg
        src_ids, dst_ids = batch.graph.src_ids, batch.graph.dst_ids
        for i in range(self.num_layers):
            x, local_lgx = self.gnn_layers[i](x, local_lgx, global_lgx, local_g, global_g, lg, src_ids, dst_ids)
            if self.graph_view == 'msde':
                # update local edge embeddings in the global edge embeddings matrix
                global_lgx = global_lgx.masked_scatter_(mask.unsqueeze(-1), local_lgx)
        return x

class DualRGATLayer(nn.Module):

    def __init__(self, ndim, edim, num_heads=8, feat_drop=0.2, graph_view='mmc'):
        super(DualRGATLayer, self).__init__()
        self.ndim, self.edim = ndim, edim
        self.num_heads, self.graph_view = num_heads, graph_view
        NodeRGATLayer = MultiViewRGATLayer if self.graph_view == 'mmc' else RGATLayer
        self.node_update = NodeRGATLayer(self.ndim, self.edim, self.num_heads, feat_drop=feat_drop)
        self.edge_update = EdgeRGATLayer(self.edim, self.ndim, self.num_heads, feat_drop=feat_drop)

    def forward(self, x, local_lgx, global_lgx, local_g, global_g, lg, src_ids, dst_ids):
        if self.graph_view == 'mmc':
            out_x, _ = self.node_update(x, local_lgx, global_lgx, local_g, global_g)
        elif self.graph_view == 'msde':
            out_x, _ = self.node_update(x, global_lgx, global_g)
        else:
            out_x, _ = self.node_update(x, local_lgx, local_g)
        src_x = torch.index_select(x, dim=0, index=src_ids)
        dst_x = torch.index_select(x, dim=0, index=dst_ids)
        out_local_lgx, _ = self.edge_update(local_lgx, src_x, dst_x, lg)
        return out_x, out_local_lgx

class EdgeRGATLayer(nn.Module):

    def __init__(self, edim, ndim, num_heads=8, feat_drop=0.2):
        super(EdgeRGATLayer, self).__init__()
        self.edim, self.ndim = edim, ndim
        self.num_heads = num_heads
        self.d_k = self.ndim // self.num_heads
        self.affine_q, self.affine_k, self.affine_v = nn.Linear(self.edim, self.ndim), \
            nn.Linear(self.edim, self.ndim, bias=False), nn.Linear(self.edim, self.ndim, bias=False)
        self.affine_o = nn.Linear(self.ndim, self.edim)
        self.layernorm = nn.LayerNorm(self.edim)
        self.feat_dropout = nn.Dropout(p=feat_drop)
        self.ffn = FFN(self.edim)

    def forward(self, x, src_x, dst_x, g):
        q, k, v = self.affine_q(self.feat_dropout(x)), self.affine_k(self.feat_dropout(x)), self.affine_v(self.feat_dropout(x))
        with g.local_scope():
            g.ndata['q'] = (q + src_x).view(-1, self.num_heads, self.d_k)
            g.ndata['k'] = k.view(-1, self.num_heads, self.d_k)
            g.ndata['v'] = (v + dst_x).view(-1, self.num_heads, self.d_k)
            out_x = self.propagate_attention(g)
        out_x = self.layernorm(x + self.affine_o(out_x.view(-1, self.num_heads * self.d_k)))
        out_x = self.ffn(out_x)
        return out_x, (src_x, dst_x)

    def propagate_attention(self, g):
        # Compute attention score
        g.apply_edges(src_dot_dst('k', 'q', 'score'))
        g.apply_edges(scaled_exp('score', math.sqrt(self.d_k)))
        # Update node state
        g.update_all(fn.src_mul_edge('v', 'score', 'v'), fn.sum('v', 'wv'))
        g.update_all(fn.copy_edge('score', 'score'), fn.sum('score', 'z'), div_by_z('wv', 'z', 'o'))
        out_x = g.ndata['o']
        return out_x
