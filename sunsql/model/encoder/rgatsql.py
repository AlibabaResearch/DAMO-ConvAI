#coding=utf8
import copy, math
import torch, dgl
import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F
from model.model_utils import Registrable, FFN
from model.encoder.functions import *

@Registrable.register('rgatsql')
class RGATSQL(nn.Module):

    def __init__(self, args):
        super(RGATSQL, self).__init__()
        self.num_layers = args.gnn_num_layers
        self.relation_num = args.relation_num
        self.relation_share_layers, self.relation_share_heads = args.relation_share_layers, args.relation_share_heads
        self.graph_view = 'multiview' if args.local_and_nonlocal in ['mmc', 'msde'] else args.local_and_nonlocal
        edim = args.gnn_hidden_size // args.num_heads if self.relation_share_heads else \
            args.gnn_hidden_size // 2 if self.graph_view == 'multiview' else args.gnn_hidden_size
        if self.relation_share_layers:
            self.relation_embed = nn.Embedding(args.relation_num, edim)
        else:
            self.relation_embed = nn.ModuleList([nn.Embedding(args.relation_num, edim) for _ in range(self.num_layers)])
        gnn_layer = MultiViewRGATLayer if self.graph_view == 'multiview' else RGATLayer
        self.gnn_layers = nn.ModuleList([gnn_layer(args.gnn_hidden_size, edim, num_heads=args.num_heads, feat_drop=args.dropout)
            for _ in range(self.num_layers)])

    def forward(self, x, batch):
        if self.graph_view == 'multiview':
            # multi-view multi-head concatenation
            global_edges, mask = batch.graph.global_edges, batch.graph.local_mask
            local_g, global_g = batch.graph.local_g, batch.graph.global_g
            if self.relation_share_layers:
                global_lgx = self.relation_embed(global_edges)
                local_lgx = global_lgx[mask]
            for i in range(self.num_layers):
                global_lgx = self.relation_embed[i](global_edges) if not self.relation_share_layers else global_lgx
                local_lgx = global_lgx[mask] if not self.relation_share_layers else local_lgx
                x, (local_lgx, global_lgx) = self.gnn_layers[i](x, local_lgx, global_lgx, local_g, global_g)
        else:
            graph = batch.graph.local_g if self.graph_view == 'local' else batch.graph.global_g
            edges, mask = batch.graph.global_edges, batch.graph.local_mask
            if self.relation_share_layers:
                lgx = self.relation_embed(edges)
                lgx = lgx if self.graph_view == 'global' else lgx[mask]
            for i in range(self.num_layers):
                lgx = lgx if self.relation_share_layers else self.relation_embed[i](edges) if self.graph_view == 'global' else self.relation_embed[i](edges)[mask]
                x, lgx = self.gnn_layers[i](x, lgx, graph)
        return x

class RGATLayer(nn.Module):

    def __init__(self, ndim, edim, num_heads=8, feat_drop=0.2):
        super(RGATLayer, self).__init__()
        self.ndim, self.edim = ndim, edim
        self.num_heads = num_heads
        dim = max([ndim, edim])
        self.d_k = dim // self.num_heads
        self.affine_q, self.affine_k, self.affine_v = nn.Linear(self.ndim, dim),\
            nn.Linear(self.ndim, dim, bias=False), nn.Linear(self.ndim, dim, bias=False)
        self.affine_o = nn.Linear(dim, self.ndim)
        self.layernorm = nn.LayerNorm(self.ndim)
        self.feat_dropout = nn.Dropout(p=feat_drop)
        self.ffn = FFN(self.ndim)

    def forward(self, x, lgx, g):
        """ @Params:
                x: node feats, num_nodes x ndim
                lgx: edge feats, num_edges x edim
                g: dgl.graph
        """
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

class MultiViewRGATLayer(RGATLayer):

    def forward(self, x, local_lgx, global_lgx, local_g, global_g):
        """ @Params:
                x: node feats, num_nodes x ndim
                local_lgx: local edge feats, local_edge_num x edim
                global_lgx: all edge feats, global_edge_num x edim
                local_g: dgl.graph, a local graph for node update
                global_g: dgl.graph, a complete graph for node update
        """
        # pre-mapping q/k/v affine
        q, k, v = self.affine_q(self.feat_dropout(x)), self.affine_k(self.feat_dropout(x)), self.affine_v(self.feat_dropout(x))
        q, k, v = q.view(-1, self.num_heads, self.d_k), k.view(-1, self.num_heads, self.d_k), v.view(-1, self.num_heads, self.d_k)
        with local_g.local_scope():
            local_g.ndata['q'], local_g.ndata['k'] = q[:, :self.num_heads // 2], k[:, :self.num_heads // 2]
            local_g.ndata['v'] = v[:, :self.num_heads // 2]
            local_g.edata['e'] = local_lgx.view(-1, self.num_heads // 2, self.d_k) if local_lgx.size(-1) == self.d_k * self.num_heads // 2 else \
                local_lgx.unsqueeze(1).expand(-1, self.num_heads // 2, -1)
            out_x1 = self.propagate_attention(local_g)
        with global_g.local_scope():
            global_g.ndata['q'], global_g.ndata['k'] = q[:, self.num_heads // 2:], k[:, self.num_heads // 2:]
            global_g.ndata['v'] = v[:, self.num_heads // 2:]
            global_g.edata['e'] = global_lgx.view(-1, self.num_heads // 2, self.d_k) if global_lgx.size(-1) == self.d_k * self.num_heads // 2 else \
                global_lgx.unsqueeze(1).expand(-1, self.num_heads // 2, -1)
            out_x2 = self.propagate_attention(global_g)
        out_x = torch.cat([out_x1, out_x2], dim=1)
        out_x = self.layernorm(x + self.affine_o(out_x.view(-1, self.num_heads * self.d_k)))
        out_x = self.ffn(out_x)
        return out_x, (local_lgx, global_lgx)
