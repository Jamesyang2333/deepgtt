import torch
import torch.nn as nn
import torch.nn.functional as F
from gat.layers import GraphAttentionLayer, SpGraphAttentionLayer


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions_1 = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions_1):
            self.add_module('attention_1_{}'.format(i), attention)
        self.attentions_2 = [GraphAttentionLayer(nhid * nheads, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions_2):
            self.add_module('attention_2_{}'.format(i), attention)
        self.attentions_3 = [GraphAttentionLayer(nhid * nheads, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions_3):
            self.add_module('attention_3_{}'.format(i), attention)
        self.attentions_4 = [GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions_4):
            self.add_module('attention_4_{}'.format(i), attention)

#         self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions_1], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions_2], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions_3], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions_4], dim=1)
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = F.elu(self.out_att(x, adj))
        return x


class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions_1 = [SpGraphAttentionLayer(nfeat, 
                                                 nhid, 
                                                 dropout=dropout, 
                                                 alpha=alpha, 
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions_1):
            self.add_module('attention_1_{}'.format(i), attention)
        
        self.attentions_2 = [SpGraphAttentionLayer(nhid * nheads, 
                                                 nclass // nheads, 
                                                 dropout=dropout, 
                                                 alpha=alpha, 
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions_2):
            self.add_module('attention_2_{}'.format(i), attention)
        
#         self.attentions_3 = [SpGraphAttentionLayer(nhid * nheads, 
#                                                  nhid, 
#                                                  dropout=dropout, 
#                                                  alpha=alpha, 
#                                                  concat=True) for _ in range(nheads)]
#         for i, attention in enumerate(self.attentions_3):
#             self.add_module('attention_3_{}'.format(i), attention)
        
#         self.attentions_4 = [SpGraphAttentionLayer(nhid * nheads, 
#                                                  nclass, 
#                                                  dropout=dropout, 
#                                                  alpha=alpha, 
#                                                  concat=True) for _ in range(nheads)]
#         for i, attention in enumerate(self.attentions_4):
#             self.add_module('attention_4_{}'.format(i), attention)

#         self.out_att = SpGraphAttentionLayer(nhid * nheads, 
#                                              nclass, 
#                                              dropout=dropout, 
#                                              alpha=alpha, 
#                                              concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions_1], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions_2], dim=1)
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = torch.cat([att(x, adj) for att in self.attentions_3], dim=1)
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = torch.cat([att(x, adj) for att in self.attentions_4], dim=1)
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = F.elu(self.out_att(x, adj))
        return x

