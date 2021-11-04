import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from .gcn_layers import GCNLayer, SAGELayer, HetLayer
import math

class GAU_I(nn.Module):
    """ LSTM with masks """
    def __init__(self, dim_feats, dim_h, idx2feats, p2idx, out_sz = 300, rnn_type='lstm', dropout=0.2):
        super(GAU_I, self).__init__()

        self.dim_feats = dim_feats
        self.dim_h = dim_h
        self.out_sz = out_sz
        self.idx2feats = idx2feats
        self.p2idx = p2idx
        # Transform hidden space to feature space
        self.fc1 = nn.Linear(dim_h, 1)
        self.fc2 = nn.Linear(dim_h, out_sz)

        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size=dim_feats, hidden_size=dim_h, num_layers=2, bias=True, batch_first=True, dropout=dropout)
        if rnn_type == 'rnn':
            self.rnn = nn.RNN(input_size=dim_feats, hidden_size=dim_h, num_layers=2, bias=True, batch_first=True, dropout=dropout)
        if rnn_type == 'gru':
            self.rnn = nn.GRU(input_size=dim_feats, hidden_size=dim_h, num_layers=2, bias=True, batch_first=True, dropout=dropout)
        self.rnn_type = rnn_type

    def forward(self, feats, feats_length):
        """
            pids: Not used in this verison
            feats: (batch_size, max_len_in_batch, (feat_sz))
            return: (batch_size, max_len_in_batch, (feat_sz))
        """
        sort = np.argsort(-feats_length)
        length_sort = feats_length[sort]
        reversed_sort = np.argsort(sort)

        # packing operation
        x = torch.nn.utils.rnn.pack_padded_sequence(feats[sort], length_sort, batch_first=True)
        if self.rnn_type == 'lstm':
            output, (h, c) = self.rnn(x)
        else:
            output, h = self.rnn(x)
        # unpacking operation
        output, out_len = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        # Now we have output of shape (batch_size, len_seq, hidden_size); and we get the last hidden layer
        # output = output[np.arange(len(out_len)), out_len-1, None]
        output = output[reversed_sort]
        out_len = out_len[reversed_sort]

        # Flags and Features
        flag = torch.sigmoid(self.fc1(output))
        # Map it to feature space
        output = self.fc2(output)

        return output, out_len, flag

    @staticmethod
    def min_max(arr):
        """
            arr: Tensor
        """
        return (arr - torch.min(arr)) / (torch.max(arr)-torch.min(arr))

class GAU_E(nn.Module):
    def __init__(self, dim_feats, dim_h, idx2feats, p2idx, out_sz = 300, rnn_type='lstm', dropout=0.2):
        super(GAU_E, self).__init__()

        self.dim_feats = dim_feats
        self.dim_h = dim_h
        self.out_sz = out_sz
        self.idx2feats = idx2feats
        self.p2idx = p2idx
        # Transform hidden space to feature space
        self.fc = nn.Linear(dim_h, out_sz)
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size=dim_feats, hidden_size=dim_h, num_layers=2, bias=True, batch_first=True, dropout=dropout)
        if rnn_type == 'rnn':
            self.rnn = nn.RNN(input_size=dim_feats, hidden_size=dim_h, num_layers=2, bias=True, batch_first=True, dropout=dropout)
        if rnn_type == 'gru':
            self.rnn = nn.GRU(input_size=dim_feats, hidden_size=dim_h, num_layers=2, bias=True, batch_first=True, dropout=dropout)
        self.rnn_type = rnn_type

    def forward(self, feats, feats_length):
        """
            pids: Not used in this verison
            feats: (batch_size, max_len_in_batch, (feat_sz))
            return: (batch_size, max_len_in_batch, (feat_sz))
        """
        sort = np.argsort(-feats_length)
        length_sort = feats_length[sort]
        reversed_sort = np.argsort(sort)

        # packing operation
        x = torch.nn.utils.rnn.pack_padded_sequence(feats[sort], length_sort, batch_first=True)
        if self.rnn_type == 'lstm':
            output, (h, c) = self.rnn(x)
        else:
            output, h = self.rnn(x)
        # unpacking operation
        output, out_len = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        # Now we have output of shape (batch_size, len_seq, hidden_size); and we get the last hidden layer
        # output = output[np.arange(len(out_len)), out_len-1, None]
        output = output[reversed_sort]
        out_len = out_len[reversed_sort]

        # Map it to feature space
        output = self.fc(output)

        return output, out_len

    @staticmethod
    def min_max(arr):
        """
            arr: Tensor
        """
        return (arr - torch.min(arr)) / (torch.max(arr)-torch.min(arr))

class GNN(nn.Module):
    """
        GCN and GSage
    """
    def __init__(self, dim_feats, dim_h, n_classes, n_layers,
                activation, dropout, gnnlayer_type='gcn'):
        super(GNN, self).__init__()
        heads = [1] * (n_layers + 1)
        if gnnlayer_type == 'gcn':
            gnnlayer = GCNLayer
        elif gnnlayer_type == 'gsage':
            gnnlayer = SAGELayer
        elif gnnlayer_type == 'hetgcn':
            gnnlayer = HetLayer
        self.gnnlayer_type = gnnlayer_type
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(gnnlayer(dim_feats, dim_h, heads[0], activation, 0))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(gnnlayer(dim_h * heads[i], dim_h, heads[i+1], activation, dropout))
        # output layer
        self.layers.append(gnnlayer(dim_h * heads[-2], n_classes, heads[-1], None, dropout))

    def forward(self, adj, features_u, features_v):
        h_u, h_v = features_u, features_v
        if self.gnnlayer_type == 'gcn':
            d_u, d_v = self.get_normed_d(adj)
            for i, layer in enumerate(self.layers):
                h_u, h_v = layer(adj, h_u, h_v, d_u, d_v)
                if i == len(self.layers) - 2:
                    emb = h_u
        if self.gnnlayer_type == 'gsage' or self.gnnlayer_type == 'hetgcn':
            for i, layer in enumerate(self.layers):
                h_u, h_v = layer(adj, h_u, h_v)
                if i == len(self.layers) - 2:
                    emb = h_u
        # We only need user predictions in the end
        # return h_u
        return F.log_softmax(h_u, dim=1), emb

    @staticmethod
    def get_normed_d(A):
        """ Get normalized degree matrix of A"""
        d_u = A.sum(1) + 1
        # Self Loop
        d_v = A.sum(0) + 1

        d_u = torch.pow(d_u, -0.5)
        d_v = torch.pow(d_v, -0.5)

        return d_u, d_v


class DeepAE(nn.Module):
    def __init__(self, dim_feats, dim_h=64, dropout=0., alpha=0.025):
        super(DeepAE, self).__init__()
        self.alpha = alpha
        # shared encoder
        self.enc1 = GCNLayer(dim_feats, dim_h, 1, F.relu, 0)
        self.enc2 = GCNLayer(dim_h, dim_h, 1, F.relu, dropout)
        # attribute decoder
        self.attr_dec1 = GCNLayer(dim_h, dim_h, 1, F.relu, dropout)
        self.attr_dec2 = GCNLayer(dim_h, dim_feats, 1, F.relu, dropout)
        # # structure decoder
        # self.struct_dec1 = GCNLayer(dim_h, dim_h, 1, F.relu, dropout)

    def forward(self, adj, features_u, features_v):
        h_u, h_v = features_u, features_v
        d_u, d_v, d_u_orig = self.get_normed_d(adj)
        # encoding
        h_u, h_v = self.enc1(adj, h_u, h_v, d_u, d_v)
        z_u, z_v = self.enc2(adj, h_u, h_v, d_u, d_v)
        # attribute decoding
        x_u, x_v = self.attr_dec1(adj, z_u, z_v, d_u, d_v)
        x_u, x_v = self.attr_dec2(adj, x_u, x_v, d_u, d_v)
        # structure decoding
        # s_u, s_v = self.struct_dec1(adj, z_u, z_v, d_u, d_v)
        adj_pred = torch.sigmoid(x_u @ x_v.T)

        loss, err = self.get_loss(adj, adj_pred, features_u, x_u, x_v, d_u_orig)
        # print(loss)
        # print('done---')
        return loss, err, z_u

    def get_loss(self, adj, adj_pred, h, h_pred, x_v, d):
        # attribute reconstruction loss
        # diff_attr = torch.square(h_pred - h)
        diff_attr = torch.pow(h_pred - h, 2)
        err_attr = torch.sqrt(diff_attr.sum(1))
        loss_attr = torch.mean(err_attr)
        # structure reconstruction loss
        # diff_stru = torch.square(adj_pred - adj)
        diff_stru = torch.pow(adj_pred - adj, 2)
        err_stru = torch.sqrt(diff_stru.sum(1))
        loss_stru = torch.mean(err_stru)
        # 1st order proximity
        loss_f = 2 * torch.trace((h_pred.T * d) @ h_pred)
        # semantic proximity
        h_u_norm = F.normalize(h_pred, p=1, dim=1)
        h_v_norm = F.normalize(x_v, p=1, dim=1)
        loss_s = torch.sum(adj * (h_u_norm @ h_v_norm.T) * torch.log(adj_pred))
        # eighted loss
        loss = self.alpha * loss_attr + (1-self.alpha) * loss_stru + 0.1 * loss_f - 0.1 * loss_s
        # anomaly rank
        err = self.alpha * err_attr + (1-self.alpha) * err_stru
        err = (err - err.min()) / (err.max() - err.min() + 1e-8)
        return loss, err

    @staticmethod
    def get_normed_d(A):
        """ Get normalized degree matrix of A"""
        # add self Loop
        d_u = A.sum(1) + 1
        d_v = A.sum(0) + 1
        d_u_normed = torch.pow(d_u, -0.5)
        d_v_normed = torch.pow(d_v, -0.5)
        return d_u_normed, d_v_normed, d_u

class Dominant(nn.Module):
    def __init__(self, dim_feats, dim_h=64, dropout=0., alpha=0.5):
        super(Dominant, self).__init__()
        self.alpha = alpha
        # shared encoder
        self.enc1 = GCNLayer(dim_feats, dim_h, 1, F.relu, 0)
        self.enc2 = GCNLayer(dim_h, dim_h, 1, F.relu, dropout)
        # attribute decoder
        self.attr_dec1 = GCNLayer(dim_h, dim_h, 1, F.relu, dropout)
        self.attr_dec2 = GCNLayer(dim_h, dim_feats, 1, F.relu, dropout)
        # structure decoder
        self.struct_dec1 = GCNLayer(dim_h, dim_h, 1, F.relu, dropout)

    def forward(self, adj, features_u, features_v):
        h_u, h_v = features_u, features_v
        d_u, d_v = self.get_normed_d(adj)
        # encoding
        h_u, h_v = self.enc1(adj, h_u, h_v, d_u, d_v)
        z_u, z_v = self.enc2(adj, h_u, h_v, d_u, d_v)
        # attribute decoding
        x_u, x_v = self.attr_dec1(adj, z_u, z_v, d_u, d_v)
        x_u, _ = self.attr_dec2(adj, x_u, x_v, d_u, d_v)
        # structure decoding
        s_u, s_v = self.struct_dec1(adj, z_u, z_v, d_u, d_v)
        adj_pred = torch.sigmoid(s_u @ s_v.T)
        # get loss
        # print(torch.isnan(features_u).sum(), torch.isnan(x_u).sum())
        # x = x_u - features_u
        # print(torch.isnan(x).sum())
        # print(torch.isnan(torch.square(x)).sum())
        loss, err = self.get_loss(adj, adj_pred, features_u, x_u)
        # print(loss)
        # print('done---')
        return loss, err, z_u

    def get_loss(self, adj, adj_pred, h, h_pred):
        # attribute reconstruction loss
        # x = h_pred - h
        # diff_attr = torch.square(x)
        diff_attr = torch.pow(h_pred - h, 2)
        # diff_attr = torch.square(h_pred - h)
        err_attr = torch.sqrt(diff_attr.sum(1))
        loss_attr = torch.mean(err_attr)
        # structure reconstruction loss
        diff_stru = torch.pow(adj_pred - adj, 2)
        # diff_stru = torch.square(adj_pred - adj)
        err_stru = torch.sqrt(diff_stru.sum(1))
        loss_stru = torch.mean(err_stru)
        # get weighted loss and err
        loss = self.alpha * loss_attr + (1-self.alpha) * loss_stru
        err = self.alpha * err_attr + (1-self.alpha) * err_stru
        err = (err - err.min()) / (err.max() - err.min() + 1e-8)
        return loss, err

    @staticmethod
    def get_normed_d(A):
        """ Get normalized degree matrix of A"""
        # add self Loop
        d_u = A.sum(1) + 1
        d_v = A.sum(0) + 1
        d_u = torch.pow(d_u, -0.5)
        d_v = torch.pow(d_v, -0.5)
        return d_u, d_v
