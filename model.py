import torch
import torch.nn as nn
import torch.nn.functional as F
import os, argparse, math, numpy as np, pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from scipy.spatial.distance import cdist, pdist

class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_ft, out_ft))
        self.bias = nn.Parameter(torch.Tensor(out_ft)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        x = x.matmul(self.weight) # node feature transformation
        if self.bias is not None:
            x = x + self.bias
        x = torch.matmul(G, x) # hypergraph convolution
        return x


class HGEMMD(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_class, dropout):
        super().__init__()
        self.views = len(in_dim)
        self.num_class = num_class
        self.dropout = dropout
        self.FeatureInfoEncoder = nn.ModuleList([nn.Linear(in_dim[view], in_dim[view]) for view in range(self.views)])
        self.FeatureEncoder = nn.ModuleList([nn.Linear(in_dim[view], hidden_dim[0]) for view in range(self.views)])
        self.TCPConfidenceLayer = nn.ModuleList([nn.Linear(hidden_dim[0], 1) for _ in range(self.views)])
        self.TCPClassifierLayer = nn.ModuleList([nn.Linear(hidden_dim[0], num_class) for _ in range(self.views)])
        self.HGNN = HGNN_conv(self.views*hidden_dim[0], hidden_dim[0])
        self.MMClasifier = []
        assert len(hidden_dim) >= 1, "The length of hidden dim need to be greater than or equal to 1."
        if len(hidden_dim) == 1:
            self.MMClasifier.append(nn.Linear((self.views+1)*hidden_dim[0], num_class))
        else:
            self.MMClasifier.append(nn.Linear((self.views+1)*hidden_dim[0], hidden_dim[1]))
            self.MMClasifier.append(nn.ReLU())
            self.MMClasifier.append(nn.Dropout(p=dropout))
            for layer in range(1, len(hidden_dim) -1):
                self.MMClasifier.append(nn.Linear(hidden_dim[layer], hidden_dim[layer+1]))
                self.MMClasifier.append(nn.ReLU())
                self.MMClasifier.append(nn.Dropout(p=dropout))
            self.MMClasifier.append(nn.Linear(hidden_dim[-1], num_class))
        self.MMClasifier = nn.Sequential(*self.MMClasifier)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, data_list, G):
        FeatureInfo, TCPLogit, TCPConfidence, ModalityEmbedding = dict(), dict(), dict(), dict()
        for view in range(self.views):
            featureinfo = torch.sigmoid(self.FeatureInfoEncoder[view](data_list[view]))
            feature = self.FeatureEncoder[view](data_list[view] * featureinfo)
            feature = F.dropout(F.relu(feature), self.dropout, training=self.training)
            tcp_logit = self.TCPClassifierLayer[view](feature)
            tcp_confidence = self.TCPConfidenceLayer[view](feature)
            feature = feature * tcp_confidence
            FeatureInfo[view] = featureinfo; ModalityEmbedding[view] = feature
            TCPLogit[view] = tcp_logit; TCPConfidence[view] = tcp_confidence
        MMfeature_mmdynamics = torch.cat([i for i in ModalityEmbedding.values()], dim=1) # shape: N X (num_view*hidden_dim[0])
        MMfeature_hypergraph = self.HGNN(MMfeature_mmdynamics, G) # shape: N X hidden_dim[0]
        MMlogit = self.MMClasifier(torch.cat([MMfeature_mmdynamics, MMfeature_hypergraph], dim=1)) # shape: N X num_class
        return MMlogit, FeatureInfo, TCPLogit, TCPConfidence, ModalityEmbedding, MMfeature_mmdynamics, MMfeature_hypergraph
    
    def forward_criterion(self, data_list, labeled_indices, unlabeled_indices, label, pre_calc_G, lambda_1=0.1, lambda_2=0.1):
        MMlogit, FeatureInfo, TCPLogit, TCPConfidence, ModalityEmbedding, MMfeature, MMfeature_hypergraph = self.forward(data_list, pre_calc_G)
        criterion = torch.nn.CrossEntropyLoss()
        MMloss = criterion(MMlogit[labeled_indices], label[labeled_indices])
        intra_sample_loss = 0
        for view in range(self.views):
            view_pred = F.softmax(TCPLogit[view], dim=1)
            view_conf = torch.gather(input=view_pred, dim=1, index=label.unsqueeze(dim=1)).view(-1)
            confidence_loss = F.mse_loss(TCPConfidence[view].view(-1)[labeled_indices], view_conf[labeled_indices]) + criterion(TCPLogit[view][labeled_indices], label[labeled_indices])
            sparsity_loss = torch.mean(FeatureInfo[view][labeled_indices])
            intra_sample_loss = intra_sample_loss + confidence_loss + sparsity_loss
        tau = 0.1
        anchor_embeddings_original = MMfeature[labeled_indices] # shape: (num_labeled, num_view*hidden_dim[0])
        unlabeled_embeddings_original = MMfeature[unlabeled_indices] # shape: (num_unlabeled, num_view*hidden_dim[0])
        anchor_embeddings_hypergraph = MMfeature_hypergraph[labeled_indices] # shape: (num_labeled, hidden_dim[0])
        unlabeled_embeddings_hypergraph = MMfeature_hypergraph[unlabeled_indices] # shape: (num_unlabeled, hidden_dim[0])
        cos_sim_original = F.cosine_similarity(unlabeled_embeddings_original.unsqueeze(1), anchor_embeddings_original.unsqueeze(0), dim=-1) # shape: (num_unlabeled, num_labeled)
        P_u = F.softmax(cos_sim_original / tau, dim=-1)  # Similarity distribution for original view, shape: (num_unlabeled, num_labeled)
        cos_sim_hypergraph = F.cosine_similarity(unlabeled_embeddings_hypergraph.unsqueeze(1), anchor_embeddings_hypergraph.unsqueeze(0), dim=-1) # shape: (num_unlabeled, num_labeled)
        Q_u = F.softmax(cos_sim_hypergraph / tau, dim=-1)  # Similarity distribution for hypergraph view, shape: (num_unlabeled, num_labeled)
        inter_sample_loss = torch.mean(torch.mean(P_u * torch.log(P_u / (Q_u + 1e-8)), dim=-1) + torch.mean(Q_u * torch.log(Q_u / (P_u + 1e-8)), dim=-1))
        MMloss = MMloss + lambda_1*intra_sample_loss + lambda_2*inter_sample_loss
        return MMloss
