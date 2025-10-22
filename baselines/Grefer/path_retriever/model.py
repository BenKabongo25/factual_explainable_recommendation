import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn import HeteroEmbedding, EdgePredictor

'''
HeteroRGCN model adapted from the DGL official tutorial
https://docs.dgl.ai/en/0.6.x/tutorials/basics/5_hetero.html
https://docs.dgl.ai/en/0.8.x/tutorials/models/1_gnn/4_rgcn.html
'''


class HeteroRGCNLayer(nn.Module):
    def __init__(self, in_size, out_size, etypes):
        super(HeteroRGCNLayer, self).__init__()
        self.weight0 = nn.Linear(in_size, out_size)
        
        self.weight = nn.ModuleDict({
                name : nn.Linear(in_size, out_size) for name in etypes
            })

    def forward(self, g, feat_dict, eweight_dict=None):
        funcs = {}
        if eweight_dict is not None:
            g.edata['_edge_weight'] = eweight_dict
                
        for srctype, etype, dsttype in g.canonical_etypes:
            h0 = self.weight0(feat_dict[srctype])
            g.nodes[srctype].data['h0'] = h0
            
            Wh = self.weight[etype](feat_dict[srctype])
            g.nodes[srctype].data['Wh_%s' % etype] = Wh
            g.nodes[srctype].data['Wh_%s' % etype] = Wh

            if eweight_dict is not None:
                msg_fn = fn.u_mul_e('Wh_%s' % etype, '_edge_weight', 'm')
            else:
                msg_fn = fn.copy_u('Wh_%s' % etype, 'm')
                
            funcs[(srctype, etype, dsttype)] = (msg_fn, fn.mean('m', 'h'))

        def apply_func(nodes):
            h = nodes.data['h'] + nodes.data['h0']
            return {'h': h}
            
        g.multi_update_all(funcs, 'sum', apply_func)

        return {ntype : g.nodes[ntype].data['h'] for ntype in g.ntypes}


class HeteroRGCN(nn.Module):
    def __init__(self, g, emb_dim, hidden_size, out_size):
        super(HeteroRGCN, self).__init__()
        self.emb = HeteroEmbedding({ntype : g.num_nodes(ntype) for ntype in g.ntypes}, emb_dim)
        self.layer1 = HeteroRGCNLayer(emb_dim, hidden_size, g.etypes)
        self.layer2 = HeteroRGCNLayer(hidden_size, out_size, g.etypes)

    def forward(self, g, feat_nids=None, eweight_dict=None):
        if feat_nids is None:
            feat_dict = self.emb.weight
        else:
            feat_dict = self.emb(feat_nids)

        h_dict = self.layer1(g, feat_dict, eweight_dict)
        h_dict = {k : F.leaky_relu(h) for k, h in h_dict.items()}
        h_dict = self.layer2(g, h_dict, eweight_dict)
        return h_dict


class HeteroLinkPredictionModel(nn.Module):
    def __init__(self, encoder, src_ntype, tgt_ntype, link_pred_op='dot', **kwargs):
        super().__init__()
        self.encoder = encoder
        self.predictor = EdgePredictor(op=link_pred_op, **kwargs)
        self.src_ntype = src_ntype
        self.tgt_ntype = tgt_ntype

    def encode(self, g, feat_nids=None, eweight_dict=None):
        h = self.encoder(g, feat_nids, eweight_dict)
        return h

    def forward(self, src_nids, tgt_nids, g, feat_nids=None, eweight_dict=None):
        h = self.encode(g, feat_nids, eweight_dict)
        src_h = h[self.src_ntype][src_nids]
        tgt_h = h[self.tgt_ntype][tgt_nids]
        score = self.predictor(src_h, tgt_h).view(-1)
        return score


class LightGCNLayer(nn.Module):
    def forward(self, g, feat_dict):
        funcs = {}
        for srctype, etype, dsttype in g.canonical_etypes:
            funcs[(srctype, etype, dsttype)] = (fn.copy_u('h', 'm'), fn.mean('m', 'h'))

        g.multi_update_all(funcs, 'sum')
        for ntype in g.ntypes:
            g.nodes[ntype].data['h'] = g.nodes[ntype].data['h']
        return {ntype: g.nodes[ntype].data['h'] for ntype in g.ntypes}

class LightGCN(nn.Module):
    def __init__(self, g, emb_dim, num_layers):
        super(LightGCN, self).__init__()
        self.emb = HeteroEmbedding({ntype: g.num_nodes(ntype) for ntype in g.ntypes}, emb_dim)
        self.layers = nn.ModuleList([LightGCNLayer() for _ in range(num_layers)])

    def forward(self, g, feat_nids=None):
        if feat_nids is None:
            h_dict = self.emb.weight 
        else:
            h_dict = self.emb(feat_nids)
        
        for ntype in g.ntypes:
            g.nodes[ntype].data['h'] = h_dict[ntype]
        
        for layer in self.layers:
            h_dict = layer(g, h_dict)
        
        return h_dict

# class HeteroLinkPredictionModel(nn.Module):
#     def __init__(self, encoder, src_ntype, tgt_ntype, link_pred_op='dot', **kwargs):
#         super().__init__()
#         self.encoder = encoder
#         self.predictor = EdgePredictor(op=link_pred_op, **kwargs)
#         self.src_ntype = src_ntype
#         self.tgt_ntype = tgt_ntype

#     def forward(self, src_nids, tgt_nids, g, feat_nids=None, eweight_dict=None):
#         h = self.encoder(g, feat_nids)
#         src_h = h[self.src_ntype][src_nids]
#         tgt_h = h[self.tgt_ntype][tgt_nids]
#         score = self.predictor(src_h, tgt_h).view(-1)
#         return score