import torch
import dgl
from memorys import *
from hyper_layers import *
import manifolds

class GeneralModel(torch.nn.Module):

    def __init__(self, dim_node, dim_edge, sample_param, memory_param, gnn_param, train_param, combined=False):
        super(GeneralModel, self).__init__()
        self.dim_node = dim_node
        self.dim_node_input = dim_node
        self.dim_edge = dim_edge
        self.sample_param = sample_param
        self.memory_param = memory_param
        if not 'dim_out' in gnn_param:
            gnn_param['dim_out'] = memory_param['dim_out']
        self.gnn_param = gnn_param
        self.train_param = train_param
        if memory_param['type'] == 'node':
            if memory_param['memory_update'] == 'gru':
                self.memory_updater = GRUMemeoryUpdater(memory_param, 2 * memory_param['dim_out'] + dim_edge, memory_param['dim_out'], memory_param['dim_time'], dim_node)
            elif memory_param['memory_update'] == 'hygru':
                self.memory_updater = HyperGRUUpdater(memory_param, 2 * memory_param['dim_out'] + dim_edge, memory_param['dim_out'], memory_param['dim_time'], dim_node)
            else:
                raise NotImplementedError
            self.dim_node_input = memory_param['dim_out']
        self.layers = torch.nn.ModuleDict()
        #  HyperbolicTransformer
        if gnn_param['arch'] == 'HyperbolicTransformer':
            for h in range(sample_param['history']):
                self.layers['l0h' + str(h)] = HyperbolicTransformer(self.dim_node_input, dim_edge, gnn_param, train_param, combined=combined)
            for l in range(1, gnn_param['layer']):
                for h in range(sample_param['history']):
                    self.layers['l' + str(l) + 'h' + str(h)] = HyperbolicTransformer(self.dim_node_input, dim_edge, gnn_param, train_param, combined = False)
        else:
            raise NotImplementedError
        self.edge_predictor = EdgePredictor(gnn_param)
        if 'combine' in gnn_param and gnn_param['combine'] == 'rnn':
            self.combiner = torch.nn.RNN(gnn_param['dim_out'], gnn_param['dim_out'])
    
    def forward(self, mfgs, neg_samples=1):
        if self.memory_param['type'] == 'node':
            self.memory_updater(mfgs[0])
        out = list()
        for l in range(self.gnn_param['layer']):
            for h in range(self.sample_param['history']):
                rst = self.layers['l' + str(l) + 'h' + str(h)](mfgs[l][h])
                if 'time_transform' in self.gnn_param and self.gnn_param['time_transform'] == 'JODIE':
                    rst = self.layers['l0h' + str(h) + 't'](rst, mfgs[l][h].srcdata['mem_ts'], mfgs[l][h].srcdata['ts'])
                if l != self.gnn_param['layer'] - 1:
                    mfgs[l + 1][h].srcdata['h'] = rst
                else:
                    out.append(rst)
        if self.sample_param['history'] == 1:
            out = out[0]
        else:
            out = torch.stack(out, dim=0)
            out = self.combiner(out)[0][-1, :, :]
        return self.edge_predictor(out, neg_samples=neg_samples)

    def get_emb(self, mfgs):
        if self.memory_param['type'] == 'node':
            self.memory_updater(mfgs[0])
        out = list()
        for l in range(self.gnn_param['layer']):
            for h in range(self.sample_param['history']):
                rst = self.layers['l' + str(l) + 'h' + str(h)](mfgs[l][h])
                if 'time_transform' in self.gnn_param and self.gnn_param['time_transform'] == 'JODIE':
                    rst = self.layers['l0h' + str(h) + 't'](rst, mfgs[l][h].srcdata['mem_ts'], mfgs[l][h].srcdata['ts'])
                if l != self.gnn_param['layer'] - 1:
                    mfgs[l + 1][h].srcdata['h'] = rst
                else:
                    out.append(rst)
        if self.sample_param['history'] == 1:
            out = out[0]
        else:
            out = torch.stack(out, dim=0)
            out = self.combiner(out)[0][-1, :, :]
        return out

class NodeClassificationModel(torch.nn.Module):

    def __init__(self, dim_in, dim_hid, num_class, gnn_param):
        super(NodeClassificationModel, self).__init__()
        self.dropout = torch.nn.Dropout(0.2)
        self.fc1 = torch.nn.Linear(dim_in, dim_hid)
        self.fc2 = torch.nn.Linear(dim_hid, dim_hid)
        self.fc3 = torch.nn.Linear(dim_hid, num_class)
        self.c = gnn_param['c']
        self.manifold = getattr(manifolds, gnn_param['manifold'])()
        self.layer_norm = torch.nn.LayerNorm(num_class)
        self.act = torch.nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.manifold.proj_tan0(self.manifold.logmap0(x, c=self.c), c=self.c)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        x = torch.nn.functional.relu(x)
        x = self.fc3(self.dropout(x))
        return self.layer_norm(x)