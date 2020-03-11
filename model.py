import torch as th
from torch.nn import init
import torch.nn as nn
from dgl.nn.pytorch import GraphConv
import dgl

class Readout(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(Readout, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.weight = nn.Parameter(th.Tensor(in_feats, out_feats))
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight)

    def forward(self, graph, feat):
        graph.ndata['h'] = feat
        hg = dgl.mean_nodes(graph, 'h')
        result = th.matmul(hg, self.weight)
        return result

class GraphClassifier(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GraphClassifier, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
        # output layers
        self.layers.append(Readout(n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, graph):
        h = graph.ndata['feat']
        for i, layer in enumerate(self.layers):
            if i != 0 and i != (len(self.layers) - 1):
                h = self.dropout(h)
            h = layer(graph, h)
        return h