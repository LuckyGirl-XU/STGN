"""Hyperbolic layers."""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.module import Module
import dgl
import manifolds



def get_dim_act_curv(node_feat_dim, edge_dim, gnn_param, train_param):
    """
    Helper function to get dimension and activation at every layer.
    :param args:
    :return:
    """
    if not gnn_param['act']:
        act = lambda x: x
    else:
        act = getattr(nn, gnn_param['act'])
    acts = [act] * (gnn_param['num_layers']-1)
    dims =  [edge_dim, node_feat_dim] 
    if gnn_param['task'] in ['lp', 'rec']:
        dims += [gnn_param['dim_out']]
        acts += [gnn_param['act']]
        n_curvatures = gnn_param['num_layers']
    else:
        n_curvatures = gnn_param['num_layers'] - 1
    if gnn_param['c'] is False:
        curvatures = [nn.Parameter(torch.Tensor([1.])) for _ in range(n_curvatures)]
    else:
        # fixed curvature
        curvatures = [torch.tensor([gnn_param['c']]).to('cuda') for _ in range(n_curvatures)]
    return dims, acts, curvatures

class TimeEncode(torch.nn.Module):

    def __init__(self, dim):
        super(TimeEncode, self).__init__()
        self.dim = dim
        self.w = torch.nn.Linear(1, dim)
        self.w.weight = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, dim, dtype=np.float32))).reshape(dim, -1))
        self.w.bias = torch.nn.Parameter(torch.zeros(dim))

    def forward(self, t):
        output = torch.cos(self.w(t.reshape((-1, 1))))
        return output
    
class FermiDiracDecoder(nn.Module):
    """Fermi Dirac to compute edge probabilities based on distances."""

    def __init__(self, r, t):
        super(FermiDiracDecoder, self).__init__()
        self.r = r
        self.t = t

    def forward(self, dist):
        probs = 1. / (torch.exp((dist - self.r) / self.t) + 1.0)
        return probs
    
class HyperbolicTransformer(nn.Module):
    def __init__(self, dim_node_feat, dim_edge_feat, gnn_param, train_param, combined = False): 
        super(HyperbolicTransformer, self).__init__()
        self.gnn_param = gnn_param
        self.train_param = train_param
        self.manifold = getattr(manifolds, gnn_param['manifold'])()
        _, _, self.curvatures = get_dim_act_curv(dim_node_feat, dim_edge_feat, gnn_param, train_param)
        self.c = self.curvatures[0]
        self.c_in = self.c
        self.c_out = self.c
        self.use_bias = gnn_param['bias']
        self.num_head = gnn_param['att_head']
        self.dim_node_feat = dim_node_feat
        self.dim_edge_feat = dim_edge_feat
        self.dim_time = gnn_param['dim_time']
        self.dim_out = gnn_param['dim_out']
        self.dropout = train_param['dropout'] 
        self.att_dropout = torch.nn.Dropout(train_param['att_dropout'])
        self.att_act = torch.nn.LeakyReLU(0.2)
        self.act = torch.nn.LeakyReLU(0.2)
        self.combined = combined
        if self.dim_time > 0:
            self.time_enc = TimeEncode(self.dim_time)
        if combined:
            if self.dim_node_feat > 0:
                self.w_q_n = HypLinear(self.manifold, self.dim_node_feat, self.dim_out, self.c, self.dropout, self.use_bias)
                self.w_k_n = HypLinear(self.manifold, self.dim_node_feat, self.dim_out, self.c, self.dropout, self.use_bias)
                self.w_v_n = HypLinear(self.manifold, self.dim_node_feat, self.dim_out, self.c, self.dropout, self.use_bias)
            if dim_edge_feat > 0:
                self.w_k_e = HypLinear(self.manifold, self.dim_edge_feat, self.dim_out, self.c, self.dropout, self.use_bias)
                self.w_v_e = HypLinear(self.manifold, self.dim_edge_feat, self.dim_out, self.c, self.dropout, self.use_bias)
            if dim_time > 0:
                self.w_q_t = HypLinear(self.manifold, self.dim_time, self.dim_out, self.c, self.dropout, self.use_bias) 
                self.w_k_t = HypLinear(self.manifold, self.dim_time, self.dim_out, self.c, self.dropout, self.use_bias)
                self.w_v_t = HypLinear(self.manifold, self.dim_time, self.dim_out, self.c, self.dropout, self.use_bias)
        else:
            if self.dim_node_feat + self.dim_time > 0:
                self.w_q = HypLinear(self.manifold, self.dim_node_feat + self.dim_time, self.dim_out, self.c, self.dropout, self.use_bias)
            self.w_k = HypLinear(self.manifold, self.dim_node_feat + self.dim_edge_feat + self.dim_time, self.dim_out, self.c, self.dropout, self.use_bias)
            self.w_v = HypLinear(self.manifold, self.dim_node_feat + self.dim_edge_feat + self.dim_time, self.dim_out, self.c, self.dropout, self.use_bias)
            self.w_out = HypLinear(self.manifold, self.dim_node_feat + self.dim_out, self.dim_out, self.c, self.dropout, self.use_bias)
        self.layer_norm = torch.nn.LayerNorm(self.dim_out)
        self.hyp_act = HypAct(self.manifold, self.c_in, self.c_out, self.act)
        
    def Hyp_Encoder(self, x):
        # Features in Euclidean are mapped into Riemannian
        x_tan = self.manifold.proj_tan0(x, self.curvatures[0])
        x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0])
        x_hyp = self.manifold.proj(x_hyp, c=self.curvatures[0])
        return x_hyp
    
    def Hyp_Decoder(self, x):
        h = self.manifold.proj_tan0(self.manifold.logmap0(x, c=self.c), c=self.c)
        return h
    
    def forward(self, b):
        if self.dim_time > 0:        
            time_feat = self.time_enc(b.edata['dt'])
            time_feat = self.Hyp_Encoder(time_feat)
            zero_time_feat = self.time_enc(torch.zeros(b.num_dst_nodes(), dtype=torch.float32, device=torch.device('cuda:0')))
            zero_time_feat = self.Hyp_Encoder(zero_time_feat)
        if self.combined:
            Q = torch.zeros((b.num_edges(), self.dim_out), device=torch.device('cuda:0'))
            K = torch.zeros((b.num_edges(), self.dim_out), device=torch.device('cuda:0'))
            V = torch.zeros((b.num_edges(), self.dim_out), device=torch.device('cuda:0'))
            if self.dim_node_feat > 0:
                b.srcdata['h'] = self.Hyp_Encoder(b.srcdata['h'])
                Q += self.w_q_n(b.srcdata['h'][:b.num_dst_nodes()])[b.edges()[1]]
                K += self.w_k_n(b.srcdata['h'][b.num_dst_nodes():])[b.edges()[0] - b.num_dst_nodes()]
                V += self.w_v_n(b.srcdata['h'][b.num_dst_nodes():])[b.edges()[0] - b.num_dst_nodes()]
            if self.dim_edge_feat > 0:
                b.edata['f'] = self.Hyp_Encoder(b.edata['f'])
                K += self.w_k_e(b.edata['f'])
                V += self.w_v_e(b.edata['f'])
            if self.dim_time > 0:
                Q += self.w_q_t(zero_time_feat)[b.edges()[1]]
                K += self.w_k_t(time_feat)
                V += self.w_v_t(time_feat)
            Q = torch.reshape(Q, (Q.shape[0], self.num_head, -1))
            K = torch.reshape(K, (K.shape[0], self.num_head, -1))
            V = torch.reshape(V, (V.shape[0], self.num_head, -1))
            att = dgl.ops.edge_softmax(b, self.att_act(torch.sum(Q*K, dim=2)))
            att = self.att_dropout(att)
            V = torch.reshape(V*att[:, :, None], (V.shape[0], -1))
            b.edata['v'] = V
            b.update_all(dgl.function.copy_edge('v', 'm'), dgl.function.sum('m', 'h'))
        else:
            if self.dim_node_feat == 0:             
                b.edata['f'] = self.Hyp_Encoder(b.edata['f'])
                Q = self.w_q(zero_time_feat)[b.edges()[1]]
                K = self.w_k(torch.cat([b.edata['f'], time_feat], dim=1))
                V = self.w_v(torch.cat([b.edata['f'], time_feat], dim=1))
            elif self.dim_edge_feat == 0:
                b.srcdata['h'] = self.Hyp_Encoder(b.srcdata['h'])
                Q = self.w_q(torch.cat([b.srcdata['h'][:b.num_dst_nodes()], zero_time_feat], dim=1))[b.edges()[1]]
                K = self.w_k(torch.cat([b.srcdata['h'][b.num_dst_nodes():], time_feat], dim=1))
                V = self.w_v(torch.cat([b.srcdata['h'][b.num_dst_nodes():], time_feat], dim=1))
            else:  
                b.edata['f'] = self.Hyp_Encoder(b.edata['f'])
                Q = self.w_q.forward(torch.cat([b.srcdata['h'][:b.num_dst_nodes()], zero_time_feat], dim=1))[b.edges()[1]]
                K = self.w_k.forward(torch.cat([b.srcdata['h'][b.num_dst_nodes():], b.edata['f'], time_feat], dim=1))
                V = self.w_v.forward(torch.cat([b.srcdata['h'][b.num_dst_nodes():], b.edata['f'], time_feat], dim=1))
                Q= self.manifold.proj_tan0(self.manifold.logmap0(Q, self.c), self.c)
                K= self.manifold.proj_tan0(self.manifold.logmap0(K, self.c), self.c)
                V= self.manifold.proj_tan0(self.manifold.logmap0(V, self.c), self.c)
            Q = torch.reshape(Q, (Q.shape[0], self.num_head, -1))
            K = torch.reshape(K, (K.shape[0], self.num_head, -1))
            V = torch.reshape(V, (V.shape[0], self.num_head, -1))
            QK= self.manifold.logmap0(torch.sum(Q*K, dim=2), c=self.c)
            att = dgl.ops.edge_softmax(b, self.att_act(torch.sum(Q*K, dim=2)))
            att = self.att_dropout(att)
            V = torch.reshape(V*att[:, :, None], (V.shape[0], -1))
            V = self.manifold.proj(self.manifold.expmap0(V, c=self.c), c=self.c)
            b.srcdata['v'] = torch.cat([torch.zeros((b.num_dst_nodes(), V.shape[1]), device=torch.device('cuda:0')), V], dim=0)
            b.srcdata['v'] = self.manifold.proj(self.manifold.expmap0(b.srcdata['v'], c=self.c), c=self.c)
            b.update_all(dgl.function.copy_src('v', 'm'), dgl.function.sum('m', 'h'))# update all
        if self.dim_node_feat != 0:
            rst = torch.cat([b.dstdata['h'], b.srcdata['h'][:b.num_dst_nodes()]], dim=1)
        else:
            rst = b.dstdata['h']
        rst = self.w_out.forward(rst)
        rst = self.hyp_act.forward(rst)
        return self.layer_norm(rst)
    
           

class HypLinear(nn.Module):
    """
    Hyperbolic linear layer.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, use_bias):
        super(HypLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.dropout = dropout
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0)

    def forward(self, x):
        drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
        mv = self.manifold.mobius_matvec(drop_weight, x, self.c)
        res = self.manifold.proj(mv, self.c)
        if self.use_bias:
            bias = self.manifold.proj_tan0(self.bias.view(1, -1), self.c)
            hyp_bias = self.manifold.expmap0(bias, self.c)
            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            res = self.manifold.mobius_add(res, hyp_bias, c=self.c)
            res = self.manifold.proj(res, self.c)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features, self.c
        )




class HypAct(Module):
    """
    Hyperbolic activation layer.
    """

    def __init__(self, manifold, c_in, c_out, act):
        super(HypAct, self).__init__()
        self.manifold = manifold
        self.c_in = c_in
        self.c_out = c_out
        self.act = act

    def forward(self, x):
        xt = self.act(self.manifold.logmap0(x, c=self.c_in))
        xt = self.manifold.proj_tan0(xt, c=self.c_out)
        return self.manifold.proj(self.manifold.expmap0(xt, c=self.c_out), c=self.c_out)

    def extra_repr(self):
        return 'c_in={}, c_out={}'.format(
            self.c_in, self.c_out
        )
class EdgePredictor(torch.nn.Module):
    def __init__(self, gnn_param):
        super(EdgePredictor, self).__init__()
        self.dim_in = gnn_param['dim_out']
        self.c = gnn_param['c']
        self.manifold = getattr(manifolds, gnn_param['manifold'])()
        self.src_fc = torch.nn.Linear(self.dim_in, self.dim_in)
        self.dst_fc = torch.nn.Linear(self.dim_in, self.dim_in)
        self.out_fc = torch.nn.Linear(self.dim_in, 1)

    def forward(self, h, neg_samples=1):
        h = self.manifold.logmap0(h, c=self.c)
        h = self.manifold.proj_tan0(h, c=self.c)
        num_edge = h.shape[0] // (neg_samples + 2)
        h_src = self.src_fc(h[:num_edge])
        h_pos_dst = self.dst_fc(h[num_edge:2 * num_edge])#positive
        h_neg_dst = self.dst_fc(h[2 * num_edge:])#negative
        h_pos_edge = torch.nn.functional.relu(h_src + h_pos_dst)
        h_neg_edge = torch.nn.functional.relu(h_src.tile(neg_samples, 1) + h_neg_dst)
        return self.out_fc(h_pos_edge), self.out_fc(h_neg_edge)   