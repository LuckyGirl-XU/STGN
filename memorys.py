import torch
import dgl
from layers import TimeEncode
from torch_scatter import scatter
import manifolds
import math
import itertools
from nets import MobiusGRU
import geoopt.manifolds.stereographic.math as pmath

class MailBox():

    def __init__(self, memory_param, num_nodes, dim_edge_feat, _node_memory=None, _node_memory_ts=None,_mailbox=None, _mailbox_ts=None, _next_mail_pos=None, _update_mail_pos=None):
        self.memory_param = memory_param
        self.dim_edge_feat = dim_edge_feat
        if memory_param['type'] != 'node':
            raise NotImplementedError
        self.node_memory = torch.zeros((num_nodes, memory_param['dim_out']), dtype=torch.float32) if _node_memory is None else _node_memory
        self.node_memory_ts = torch.zeros(num_nodes, dtype=torch.float32) if _node_memory_ts is None else _node_memory_ts
        self.mailbox = torch.zeros((num_nodes, memory_param['mailbox_size'], 2 * memory_param['dim_out'] + dim_edge_feat), dtype=torch.float32) if _mailbox is None else _mailbox
        self.mailbox_ts = torch.zeros((num_nodes, memory_param['mailbox_size']), dtype=torch.float32) if _mailbox_ts is None else _mailbox_ts
        self.next_mail_pos = torch.zeros((num_nodes), dtype=torch.long) if _next_mail_pos is None else _next_mail_pos
        self.update_mail_pos = _update_mail_pos
        self.device = torch.device('cpu')
        
    def reset(self):
        self.node_memory.fill_(0)
        self.node_memory_ts.fill_(0)
        self.mailbox.fill_(0)
        self.mailbox_ts.fill_(0)
        self.next_mail_pos.fill_(0)

    def move_to_gpu(self):
        self.node_memory = self.node_memory.cuda()
        self.node_memory_ts = self.node_memory_ts.cuda()
        self.mailbox = self.mailbox.cuda()
        self.mailbox_ts = self.mailbox_ts.cuda()
        self.next_mail_pos = self.next_mail_pos.cuda()
        self.device = torch.device('cuda:0')

    def allocate_pinned_memory_buffers(self, sample_param, batch_size):
        limit = int(batch_size * 3.3)
        if 'neighbor' in sample_param:
            for i in sample_param['neighbor']:
                limit *= i + 1
        self.pinned_node_memory_buffs = list()
        self.pinned_node_memory_ts_buffs = list()
        self.pinned_mailbox_buffs = list()
        self.pinned_mailbox_ts_buffs = list()
        for _ in range(sample_param['history']):
            self.pinned_node_memory_buffs.append(torch.zeros((limit, self.node_memory.shape[1]), pin_memory=True))
            self.pinned_node_memory_ts_buffs.append(torch.zeros((limit,), pin_memory=True))
            self.pinned_mailbox_buffs.append(torch.zeros((limit, self.mailbox.shape[1], self.mailbox.shape[2]), pin_memory=True))
            self.pinned_mailbox_ts_buffs.append(torch.zeros((limit, self.mailbox_ts.shape[1]), pin_memory=True))

    def prep_input_mails(self, mfg, use_pinned_buffers=False):
        for i, b in enumerate(mfg):
            if use_pinned_buffers:
                dst_idx = idx[:b.num_dst_nodes()]
                torch.index_select(self.node_memory, 0, idx, out=self.pinned_node_memory_buffs[i][:idx.shape[0]])
                b.srcdata['mem'] = self.pinned_node_memory_buffs[i][:idx.shape[0]].cuda(non_blocking=True)
                torch.index_select(self.node_memory_ts,0, idx, out=self.pinned_node_memory_ts_buffs[i][:idx.shape[0]])
                b.srcdata['mem_ts'] = self.pinned_node_memory_ts_buffs[i][:idx.shape[0]].cuda(non_blocking=True)
                torch.index_select(self.mailbox, 0, idx, out=self.pinned_mailbox_buffs[i][:idx.shape[0]])
                b.srcdata['mem_input'] = self.pinned_mailbox_buffs[i][:idx.shape[0]].reshape(b.srcdata['ID'].shape[0], -1).cuda(non_blocking=True)
                torch.index_select(self.mailbox_ts, 0, idx, out=self.pinned_mailbox_ts_buffs[i][:idx.shape[0]])
                b.srcdata['mail_ts'] = self.pinned_mailbox_ts_buffs[i][:idx.shape[0]].cuda(non_blocking=True)
            else:
                b.srcdata['mem'] = self.node_memory[b.srcdata['ID'].long()].cuda()
                b.srcdata['mem_ts'] = self.node_memory_ts[b.srcdata['ID'].long()].cuda()
                b.srcdata['mem_input'] = self.mailbox[b.srcdata['ID'].long()].cuda().reshape(b.srcdata['ID'].shape[0], -1)
                b.srcdata['mail_ts'] = self.mailbox_ts[b.srcdata['ID'].long()].cuda()

    def update_memory(self, nid, memory, root_nodes, ts, neg_samples=1):
        if nid is None:
            return
        num_true_src_dst = root_nodes.shape[0] // (neg_samples + 2) * 2
        with torch.no_grad():
            nid = nid[:num_true_src_dst].to(self.device)
            memory = memory[:num_true_src_dst].to(self.device)
            ts = ts[:num_true_src_dst].to(self.device)
            self.node_memory[nid.long()] = memory
            self.node_memory_ts[nid.long()] = ts

    def update_mailbox(self, nid, memory, root_nodes, ts, edge_feats, block, neg_samples=1):
        with torch.no_grad():
            num_true_edges = root_nodes.shape[0] // (neg_samples + 2)
            memory = memory.to(self.device)
            if edge_feats is not None:
                edge_feats = edge_feats.to(self.device)
            if block is not None:
                block = block.to(self.device)
            # TGN/JODIE
            if self.memory_param['deliver_to'] == 'self':
                src = torch.from_numpy(root_nodes[:num_true_edges]).to(self.device)
                dst = torch.from_numpy(root_nodes[num_true_edges:num_true_edges * 2]).to(self.device)
                mem_src = memory[:num_true_edges]
                mem_dst = memory[num_true_edges:num_true_edges * 2]
                if self.dim_edge_feat > 0:
                    src_mail = torch.cat([mem_src, mem_dst, edge_feats], dim=1)
                    dst_mail = torch.cat([mem_dst, mem_src, edge_feats], dim=1)
                else:
                    src_mail = torch.cat([mem_src, mem_dst], dim=1)
                    dst_mail = torch.cat([mem_dst, mem_src], dim=1)
                mail = torch.cat([src_mail, dst_mail], dim=1).reshape(-1, src_mail.shape[1])
                nid = torch.cat([src.unsqueeze(1), dst.unsqueeze(1)], dim=1).reshape(-1)
                mail_ts = torch.from_numpy(ts[:num_true_edges * 2]).to(self.device)
                if mail_ts.dtype == torch.float64:
                    import pdb; pdb.set_trace()
                # find unique nid to update mailbox
                uni, inv = torch.unique(nid, return_inverse=True)
                perm = torch.arange(inv.size(0), dtype=inv.dtype, device=inv.device)
                perm = inv.new_empty(uni.size(0)).scatter_(0, inv, perm)
                nid = nid[perm]
                mail = mail[perm]
                mail_ts = mail_ts[perm]
                if self.memory_param['mail_combine'] == 'last':
                    self.mailbox[nid.long(), self.next_mail_pos[nid.long()]] = mail
                    self.mailbox_ts[nid.long(), self.next_mail_pos[nid.long()]] = mail_ts
                    if self.memory_param['mailbox_size'] > 1:
                        self.next_mail_pos[nid.long()] = torch.remainder(self.next_mail_pos[nid.long()] + 1, self.memory_param['mailbox_size'])
            # APAN
            elif self.memory_param['deliver_to'] == 'neighbors':
                mem_src = memory[:num_true_edges]
                mem_dst = memory[num_true_edges:num_true_edges * 2]
                if self.dim_edge_feat > 0:
                    src_mail = torch.cat([mem_src, mem_dst, edge_feats], dim=1)
                    dst_mail = torch.cat([mem_dst, mem_src, edge_feats], dim=1)
                else:
                    src_mail = torch.cat([mem_src, mem_dst], dim=1)
                    dst_mail = torch.cat([mem_dst, mem_src], dim=1)
                mail = torch.cat([src_mail, dst_mail], dim=0)
                mail = torch.cat([mail, mail[block.edges()[0].long()]], dim=0)
                mail_ts = torch.from_numpy(ts[:num_true_edges * 2]).to(self.device)
                mail_ts = torch.cat([mail_ts, mail_ts[block.edges()[0].long()]], dim=0)
                if self.memory_param['mail_combine'] == 'mean':
                    (nid, idx) = torch.unique(block.dstdata['ID'], return_inverse=True)
                    mail = scatter(mail, idx, reduce='mean', dim=0)
                    mail_ts = scatter(mail_ts, idx, reduce='mean')
                    self.mailbox[nid.long(), self.next_mail_pos[nid.long()]] = mail
                    self.mailbox_ts[nid.long(), self.next_mail_pos[nid.long()]] = mail_ts
                elif self.memory_param['mail_combine'] == 'last':
                    nid = block.dstdata['ID']
                    # find unique nid to update mailbox
                    uni, inv = torch.unique(nid, return_inverse=True)
                    perm = torch.arange(inv.size(0), dtype=inv.dtype, device=inv.device)
                    perm = inv.new_empty(uni.size(0)).scatter_(0, inv, perm)
                    nid = nid[perm]
                    mail = mail[perm]
                    mail_ts = mail_ts[perm]
                    self.mailbox[nid.long(), self.next_mail_pos[nid.long()]] = mail
                    self.mailbox_ts[nid.long(), self.next_mail_pos[nid.long()]] = mail_ts
                else:
                    raise NotImplementedError
                if self.memory_param['mailbox_size'] > 1:
                    if self.update_mail_pos is None:
                        self.next_mail_pos[nid.long()] = torch.remainder(self.next_mail_pos[nid.long()] + 1, self.memory_param['mailbox_size'])
                    else:
                        self.update_mail_pos[nid.long()] = 1
            else:
                raise NotImplementedError

    def update_next_mail_pos(self):
        if self.update_mail_pos is not None:
            nid = torch.where(self.update_mail_pos == 1)[0]
            self.next_mail_pos[nid] = torch.remainder(self.next_mail_pos[nid] + 1, self.memory_param['mailbox_size'])
            self.update_mail_pos.fill_(0)
            
class HyperGRUUpdater(torch.nn.Module):

    def __init__(self, memory_param, dim_in, dim_hid, dim_time, dim_node_feat):
        super(HyperGRUUpdater, self).__init__()
        self.c = torch.tensor(memory_param['c1'])
        self.manifold =  getattr(manifolds, memory_param['Mani'])()
        self.dim_hid = dim_hid
        self.dim_node_feat = dim_node_feat
        self.memory_param = memory_param
        self.dim_time = dim_time
        self.updater = MobiusGRU(dim_in + dim_time, dim_hid, 1, self.c).to('cuda')
        self.last_updated_memory = None
        self.last_updated_ts = None
        self.last_updated_nid = None
        if dim_time > 0:
            self.time_enc = TimeEncode(dim_time)
        if memory_param['combine_node_feature']:
            if dim_node_feat > 0 and dim_node_feat != dim_hid:
                self.node_feat_map = torch.nn.Linear(dim_node_feat, dim_hid)
    def Hyp_Encoder(self, x):
        x_tan = self.manifold.proj_tan0(x, self.c)
        x_hyp = self.manifold.expmap0(x_tan, self.c)
        x_hyp = self.manifold.proj(x_hyp, self.c)
        return x_hyp
    
    def Hyp_Decoder(self, x):
        h = self.manifold.proj_tan0(pmath.logmap0(x, self.c), self.c)
        return h
    
    def forward(self, mfg):
        for b in mfg:
            if self.dim_time > 0:
                time_feat = self.time_enc(b.srcdata['ts'] - b.srcdata['mem_ts'])
                time_feat = self.Hyp_Encoder(time_feat)
                b.srcdata['mem_input'] = torch.cat([b.srcdata['mem_input'], time_feat], dim=1)
            updated_memory = self.updater(b.srcdata['mem_input'], b.srcdata['mem'])[0]
            self.last_updated_ts = b.srcdata['ts'].detach().clone()
            self.last_updated_memory = updated_memory.detach().clone()
            self.last_updated_nid = b.srcdata['ID'].detach().clone()
            if self.memory_param['combine_node_feature']:
                if self.dim_node_feat > 0:
                    if self.dim_node_feat == self.dim_hid:
                        b.srcdata['h'] += updated_memory
                    else:
                        b.srcdata['h'] = pmath.mobius_add(updated_memory, self.node_feat_map(b.srcdata['h']), self.c)
                else:
                    b.srcdata['h'] = updated_memory            


class GRUMemeoryUpdater(torch.nn.Module):

    def __init__(self, memory_param, dim_in, dim_hid, dim_time, dim_node_feat):
        super(GRUMemeoryUpdater, self).__init__()
        self.c = memory_param['c1']
        self.manifold = getattr(manifolds, memory_param['Mani'])()
        self.dim_hid = dim_hid
        self.dim_node_feat = dim_node_feat
        self.memory_param = memory_param
        self.dim_time = dim_time
        self.updater = torch.nn.GRUCell(dim_in + dim_time, dim_hid)
        self.last_updated_memory = None
        self.last_updated_ts = None
        self.last_updated_nid = None
        if dim_time > 0:
            self.time_enc = TimeEncode(dim_time)
        if memory_param['combine_node_feature']:
            if dim_node_feat > 0 and dim_node_feat != dim_hid:
                self.node_feat_map = torch.nn.Linear(dim_node_feat, dim_hid)

    def forward(self, mfg):
        for b in mfg:
            if self.dim_time > 0:
                time_feat = self.time_enc(b.srcdata['ts'] - b.srcdata['mem_ts'])
                time_feat = self.manifold.proj(self.manifold.expmap0(time_feat, self.c), self.c)
                b.srcdata['mem_input'] = torch.cat([b.srcdata['mem_input'], time_feat], dim=1)
            b.srcdata['mem_input'] = self.manifold.proj_tan0(self.manifold.logmap0(b.srcdata['mem_input'], c=self.c), self.c)
            b.srcdata['mem'] = self.manifold.proj_tan0(self.manifold.logmap0(b.srcdata['mem'], c=self.c), c=self.c)
            updated_memory = self.updater(b.srcdata['mem_input'], b.srcdata['mem'])
            updated_memory = self.manifold.proj(self.manifold.expmap0(updated_memory, self.c), self.c)
            self.last_updated_ts = b.srcdata['ts'].detach().clone()
            self.last_updated_memory = updated_memory.detach().clone()
            self.last_updated_nid = b.srcdata['ID'].detach().clone()
            if self.memory_param['combine_node_feature']:
                if self.dim_node_feat > 0:
                    if self.dim_node_feat == self.dim_hid:
                        b.srcdata['h'] += updated_memory
                    else:
                        b.srcdata['h'] = updated_memory + self.node_feat_map(b.srcdata['h'])
                else:
                    b.srcdata['h'] = updated_memory
                    #print('h', b.srcdata['h']) 

class HypGRU(torch.nn.Module):
    def __init__(self, memory_param, dim_in, dim_hid, dim_time, dim_node_feat):
        super(HypGRU, self).__init__()
        self.nhid = dim_hid
        self.c = memory_param['c1']
        self.manifold =  getattr(manifolds, memory_param['Mani'])()
        self.dim_hid = dim_hid
        self.dim_node_feat = dim_node_feat
        self.memory_param = memory_param
        self.dim_time = dim_time
        #self.updater = torch.nn.GRUCell(dim_in + dim_time, dim_hid)
        self.last_updated_memory = None
        self.last_updated_ts = None
        self.last_updated_nid = None
        if dim_time > 0:
            self.time_enc = TimeEncode(dim_time)
        if memory_param['combine_node_feature']:
            if dim_node_feat > 0 and dim_node_feat != dim_hid:
                self.node_feat_map = torch.nn.Linear(dim_node_feat, dim_hid)
        self.weight_ih = torch.nn.parameter.Parameter(torch.Tensor(3*dim_hid, dim_in + dim_time), requires_grad=True).to('cuda:0')
        self.weight_hh = torch.nn.parameter.Parameter(torch.Tensor(3*dim_hid, dim_hid), requires_grad=True).to('cuda:0')
        #self.bias = True
        if memory_param['bias']:
            bias = torch.nn.parameter.Parameter(torch.zeros(3, dim_hid) * 1e-5, requires_grad=False)
            self.bias = self.manifold.expmap0(bias, self.c).to('cuda:0')
        else:
            self.register_buffer("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.nhid)
        for weight in itertools.chain.from_iterable([self.weight_ih, self.weight_hh]):
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, mfg):
        for b in mfg:
            if self.dim_time > 0:
                time_feat = self.time_enc(b.srcdata['ts'] - b.srcdata['mem_ts'])
                time_feat = self.Hyp_Encoder(time_feat)
                b.srcdata['mem_input'] = torch.cat([b.srcdata['mem_input'], time_feat], dim=1)
            updated_memory = self.mobius_gru_cell(b.srcdata['mem_input'], b.srcdata['mem'], self.weight_ih, self.weight_hh, self.bias, self.c)
            #self.updater(b.srcdata['mem_input'], b.srcdata['mem'])
            self.last_updated_ts = b.srcdata['ts'].detach().clone()
            self.last_updated_memory = updated_memory.detach().clone()
            self.last_updated_nid = b.srcdata['ID'].detach().clone()
            if self.memory_param['combine_node_feature']:
                if self.dim_node_feat > 0:
                    if self.dim_node_feat == self.dim_hid:
                        b.srcdata['h'] = self.manifold.mobius_add(b.srcdata['h'], updated_memory, c= self.c)
                    else:
                        b.srcdata['h'] = self.manifold.mobius_add(updated_memory, self.node_feat_map(b.srcdata['h']), c= self.c)
                else:
                    b.srcdata['h'] = updated_memory
        

    def mobius_gru_cell(input, hx, weight_ih, weight_hh, bias, k, nonlin=None):
        W_ir, W_ih, W_iz = weight_ih.chunk(3)
        b_r, b_h, b_z = bias
        W_hr, W_hh, W_hz = weight_hh.chunk(3)

        z_t = self.manifold.logmap0(one_hyperb_rnn_transform(W_hz, hx, W_iz, input, b_z, k), k=k).sigmoid()
        r_t = self.manifold.logmap0(one_hyperb_rnn_transform(W_hr, hx, W_ir, input, b_r, k), k=k).sigmoid()

        rh_t = self.manifold.mobius_pointwise_mul(r_t, hx, k=k)
        h_tilde = one_hyperb_rnn_transform(W_hh, rh_t, W_ih, input, b_h, k)

        if nonlin is not None:
            h_tilde = self.manifold.mobius_fn_apply(nonlin, h_tilde, k=k)
        delta_h = self.manifold.mobius_add(-hx, h_tilde, k=k)
        h_out = self.manifold.mobius_add(hx, pmath.mobius_pointwise_mul(z_t, delta_h, k=k), k=k)
        return h_out

    def one_hyperb_rnn_transform(W, h, U, x, b, k):
        W_otimes_h = self.manifold.mobius_matvec(W, h, k)
        U_otimes_x = self.manifold.mobius_matvec(U, x, k)
        Wh_plus_Ux = self.manifold.mobius_add(W_otimes_h, U_otimes_x, k)
        return self.manifold.mobius_add(Wh_plus_Ux, b, k)
    
    def Hyp_Encoder(self, x):
        x_tan = self.manifold.proj_tan0(x, self.c)
        x_hyp = self.manifold.expmap0(x_tan, c=self.c)
        x_hyp = self.manifold.proj(x_hyp, c=self.c)
        return x_hyp
    
    def mobius_linear(self, input, weight, bias=None, hyperbolic_input=True, hyperbolic_bias=True, nonlin=None):
        if hyperbolic_input:
            output = self.manifold.mobius_matvec(weight, input)
        else:
            output = torch.nn.functional.linear(input, weight)
            output = self.manifold.expmap0(output)
        if bias is not None:
            if not hyperbolic_bias:
                bias = self.manifold.expmap0(bias)
            output = self.manifold.mobius_add(output, bias)
        if nonlin is not None:
            output = self.manifold.mobius_fn_apply(nonlin, output)
        output = self.manifold.project(output)
        return output
