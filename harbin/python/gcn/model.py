import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.nn.pytorch import GraphConv

gcn_msg = fn.copy_src(src='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')

class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, feature):
        # Creating a local scope so that all the stored ndata and edata
        # (such as the `'h'` ndata below) are automatically popped out
        # when the scope exits.
        with g.local_scope():
            g.ndata['h'] = feature
            g.update_all(gcn_msg, gcn_reduce)
            h = g.ndata['h']
            return self.linear(h)
        
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = GCNLayer(1, 32)
        self.layer2 = GCNLayer(32, 64)
        self.layer3 = GCNLayer(64, 128)
    
    def forward(self, g, features):
        x = F.relu(self.layer1(g, features))
        x = F.relu(self.layer2(g, x))
        x = self.layer3(g, x)
        return x
    
class GCN_RES(nn.Module):
    def __init__(self):
        super(GCN_RES, self).__init__()
        self.layer1 = GCNLayer(1, 64)
        self.layer2 = GCNLayer(64, 64)
#         self.layer3 = GCNLayer(64, 64)
        self.layer3 = GCNLayer(64, 128)
        self.dropout1 = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.4)
    
    def forward(self, g, features):
        x = self.dropout1(features)
        x = F.relu(self.layer1(g, x))
        newx = self.layer2(g, x)
        x = x + newx;
        x = F.relu(x)
#         newx = F.relu(self.layer3(g, x))
#         x = x + newx;
        x = self.dropout2(x)
        x = self.layer3(g, x)
        return x
    
# net = Net()
# print(net)

class GCN(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.g, h)
        return h
    
    
