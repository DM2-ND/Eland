import logging
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from torch.nn import MSELoss, BCELoss, CosineEmbeddingLoss
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score, roc_auc_score, f1_score
from tqdm import tqdm
from utils.utils import MultipleOptimizer
import matplotlib.pyplot as plt
import math
from .model_zoo import GAU_I, GNN, GAU_E, Dominant, DeepAE
from .baselines import lockinfer
from .fraudar import *

class Eland_itr_uns:
    def __init__(self, adj_matrix, lstm_dataloader, user_features, item_features,
            labels, tvt_nids, u2index, p2index, idx2feats, dim_feats=300, cuda=0, hidden_size=64, n_layers=4,
            epochs=300, seed=-1, lr=0.001, weight_decay=5e-4, dropout=0.4,
            log=True, name='debug', gnnlayer_type='gsage', rnn_type='lstm', bmloss_type='mse', device='cuda', base_pred=150, method='dominant'):
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_epochs = epochs
        self.n_classes = len(np.unique(labels))
        self.labels = labels
        self.train_nid, self.val_nid, self.test_nid = tvt_nids
        self.lstm_dataloader = lstm_dataloader
        self.tvt_nids = tvt_nids
        self.u2index = u2index
        self.p2index = p2index
        self.dim_feats = dim_feats
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.gnnlayer_type = gnnlayer_type
        self.rnn_type = rnn_type
        self.dropout = dropout
        self.seed = seed
        self.bmloss_type = bmloss_type
        self.base_pred = base_pred
        self.dom_alpha = 0.5
        self.method = method
        if log:
            self.logger = self.get_logger(name)
        else:
            self.logger = logging.getLogger()
        # Fix random seed if needed
        if seed > 0:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        # if not torch.cuda.is_available():
        # 	cuda = -1
        # self.device = torch.device(f'cuda: {cuda}' if cuda >= 0 else 'cpu')
        self.device = device
        # Log parameters for reference
        all_vars = locals()
        self.log_parameters(all_vars)
        # load data
        self.load_data(adj_matrix, user_features, item_features, self.labels, tvt_nids, gnnlayer_type, idx2feats)
        self.idx2feats = torch.cuda.FloatTensor(idx2feats)
        # idx2feats = idx2feats.to(self.device)

    def load_data(self, adj_matrix, user_features, item_features, labels, tvt_nids, gnnlayer_type, idx2feats):
        """Process data"""
        if isinstance(user_features, torch.FloatTensor):
            self.user_features = user_features
        else:
            self.user_features = torch.FloatTensor(user_features)

        if isinstance(item_features, torch.FloatTensor):
            self.item_features = item_features
        else:
            self.item_features = torch.FloatTensor(item_features)
        # Normalize
        self.user_features = F.normalize(self.user_features, p=1, dim=1)
        self.item_features = F.normalize(self.item_features, p=1, dim=1)
        if isinstance(labels, torch.LongTensor):
            self.labels = labels
        else:
            self.labels = torch.LongTensor(labels)
        assert sp.issparse(adj_matrix)
        if not isinstance(adj_matrix, sp.coo_matrix):
            adj_matrix = sp.coo_matrix(adj_matrix)
        self.adj = scipysp_to_pytorchsp(adj_matrix).to_dense()

    def train(self, total_epochs=8):
        # torch.manual_seed(12345)
        ad_model = Dominant_trainer(self.dim_feats, self.hidden_size, self.user_features, self.item_features, self.tvt_nids,
                                    self.labels, self.device, self.logger, self.lr, self.weight_decay, self.dropout, self.dom_alpha, self.method)
        nc_logits, embs = ad_model.fit(self.adj, n_epochs=300)
        res = self.eval_node_cls(nc_logits[self.test_nid], self.labels[self.test_nid])
        self.logger.info('Baseline: auc {:.5f}, ap {:.5f}'.format(res['auc'], res['ap']))
        auc_list, ap_list = [], []
        for epoch in range(total_epochs):
            self.gau_model = GAU_Model(dim_feats=self.dim_feats, dim_h=self.hidden_size, u2idx=self.u2index, idx2feats=self.idx2feats,
                                    p2idx=self.p2index, rnn_type=self.rnn_type, lstm_dataloader=self.lstm_dataloader, device=self.device,
                                     logger=self.logger, lr=self.lr*5, embs=embs, bmloss_type=self.bmloss_type)
            ad_model = Dominant_trainer(self.dim_feats, self.hidden_size, self.user_features, self.item_features, self.tvt_nids,
                                        self.labels, self.device, self.logger, self.lr, self.weight_decay, self.dropout, self.dom_alpha, self.method)
            # self.gnn_model = GNN_Model(dim_feats=self.dim_feats, dim_h=self.hidden_size, user_features=self.user_features,
            #                 item_features=self.item_features, tvt_nids=self.tvt_nids, labels=self.labels, n_classes=self.n_classes,
            #                 n_layers=self.n_layers, activation=F.relu, dropout=self.dropout, gnnlayer_type=self.gnnlayer_type,
            #                 logger=self.logger, device=self.device, lr=self.lr, weight_decay=self.weight_decay)
            input_adj = self.adj.clone()
            self.gau_model.train(n_epochs=25)
            input_adj = self.gau_model.inference(input_adj, nc_logits, self.base_pred)
            nc_logits, embs = ad_model.fit(input_adj, n_epochs=300)
            res = self.eval_node_cls(nc_logits[self.test_nid], self.labels[self.test_nid])
            self.logger.info('Epoch {}/{}: auc {:.5f}, ap {:.5f}'.format(epoch+1, total_epochs, res['auc'], res['ap']))
            auc_list.append(res['auc'])
            ap_list.append(res['ap'])
        return np.max(auc_list), np.max(ap_list)

    @staticmethod
    def get_logger(name):
        logger = logging.getLogger(name)
        if (logger.hasHandlers()):
            logger.handlers.clear()
        logger.setLevel(logging.INFO)
        # Foramtter
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        # console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        # create file handler
        if name is not None:
            fh = logging.FileHandler(f'logs/ELANDitr-{name}.log')
            fh.setFormatter(formatter)
            fh.setLevel(logging.INFO)
            logger.addHandler(fh)
        return logger

    @staticmethod
    def eval_node_cls(logits, labels):
        logits = logits.cpu().numpy()
        # logits = logits.T[1]
        logits = (logits - np.min(logits)) / (np.max(logits) - np.min(logits) + 1e-8)
        labels = labels.cpu().numpy()

        roc_auc = roc_auc_score(labels, logits)
        ap = average_precision_score(labels, logits, pos_label = 1)
        results = {
            'ap': ap,
            'auc': roc_auc
        }
        return results

    # @staticmethod
    # def eval_node_cls(logits, labels, n_classes):
    #     logits = logits.cpu().numpy()
    #     y_pred = np.argmax(logits, axis=1)
    #     logits = logits.T[1]
    #     labels = labels.cpu().numpy()

    #     fpr, tpr, _ = roc_curve(labels, logits, pos_label=1)
    #     roc_auc = roc_auc_score(labels, logits)
    #     precisions, recalls, _ = precision_recall_curve(labels, logits, pos_label=1)
    #     # pr_auc = metrics.auc(recalls, precisions)
    #     ap = average_precision_score(labels, logits, pos_label = 1)
    #     f1s = np.nan_to_num(2*precisions*recalls/(precisions+recalls))
    #     best_comb = np.argmax(f1s)
    #     f1 = f1_score(labels, y_pred)
    #     pre = precisions[best_comb]
    #     rec = recalls[best_comb]
    #     # calc confusion matrix
    #     conf_mat = np.zeros((n_classes, n_classes))
    #     # for i in range(len(preds)):
    #     # 	conf_mat[labels[i], preds[i]] += 1
    #     results = {
    #         'pre': pre,
    #         'rec': rec,
    #         'f1': f1,
    #         'ap': ap,
    #         'conf': conf_mat,
    #         'auc': roc_auc
    #     }
    #     return results

    def log_parameters(self, all_vars):
        del all_vars['self']
        del all_vars['adj_matrix']
        del all_vars['user_features']
        del all_vars['item_features']
        del all_vars['labels']
        del all_vars['tvt_nids']
        del all_vars['lstm_dataloader']
        del all_vars['u2index']
        del all_vars['p2index']
        del all_vars['idx2feats']
        self.logger.info(f'Parameters: {all_vars}')

class Dominant_trainer:
    def __init__(self, dim_feats, dim_h, user_features, item_features,
                tvt_nids, labels, device, logger, lr, weight_decay, dropout, alpha, method):
        if method == 'dominant':
            self.model = Dominant(dim_feats, dim_h, dropout, alpha)
        elif method == 'deepae':
            self.model = DeepAE(dim_feats, dim_h, 0, 0.025)
        self.user_features = user_features
        self.item_features = item_features
        self.train_nid, self.val_nid, self.test_nid = tvt_nids
        self.device = device
        self.logger = logger
        self.labels = labels
        self.lr = lr
        self.weight_decay = weight_decay

    def fit(self, adj, n_epochs=300):
        user_features = self.user_features.to(self.device)
        item_features = self.item_features.to(self.device)
        model = self.model.to(self.device)
        labels = self.labels.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best_val_auc = 0.
        best_test_auc = 0.
        best_embs = None
        cnt_wait = 0
        patience = 50
        for epoch in range(n_epochs):
            model.train()
            model.zero_grad()
            input_adj = adj.clone()
            input_adj = input_adj.to(self.device)
            loss, logits, embs = model(input_adj, user_features, item_features)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # Validation
            model.eval()
            adj = adj.detach()
            with torch.no_grad():
                input_adj = adj.clone()
                input_adj = input_adj.to(self.device)
                _, logits_eval, embs_eval = model(input_adj, user_features, item_features)
            res_training = self.eval_node_cls(logits[self.train_nid].detach(), labels[self.train_nid])
            res = self.eval_node_cls(logits_eval[self.val_nid], labels[self.val_nid])

            if res['auc'] > best_val_auc:
                cnt_wait = 0
                best_val_auc = res['auc']
                res_test = self.eval_node_cls(logits_eval[self.test_nid], labels[self.test_nid])
                best_logits = logits_eval
                best_embs = embs_eval
            else:
                cnt_wait += 1
            print('epoch {}/{}, loss {:.5f}, training auc {:.5f}, val auc {:.5f}, best test auc {:.5f}'.format(epoch+1, n_epochs, loss.item(), res_training['auc'], res['auc'], res_test['auc']))
            if cnt_wait >= patience:
                print('Early stop!')
                break
        return best_logits, best_embs

    @staticmethod
    def eval_node_cls(logits, labels):
        logits = logits.cpu().numpy()
        # logits = logits.T[1]
        logits = (logits - np.min(logits)) / (np.max(logits) - np.min(logits) + 1e-8)
        labels = labels.cpu().numpy()

        roc_auc = roc_auc_score(labels, logits)
        ap = average_precision_score(labels, logits, pos_label = 1)
        results = {
            'ap': ap,
            'auc': roc_auc
        }
        return results

class GNN_Model:
    def __init__(self, dim_feats, dim_h, user_features, item_features, n_classes,
                n_layers, activation, dropout, gnnlayer_type, tvt_nids, labels, device, logger, lr, weight_decay):
        self.gnn = GNN(dim_feats=dim_feats, dim_h=dim_h, n_classes=n_classes,
                       n_layers=n_layers, activation=F.relu, dropout=dropout, gnnlayer_type=gnnlayer_type)
        self.user_features = user_features
        self.item_features = item_features
        self.n_classes = n_classes
        self.train_nid, self.val_nid, self.test_nid = tvt_nids
        self.device = device
        self.logger = logger
        self.labels = labels
        self.lr = lr
        self.weight_decay = weight_decay

    def fit(self, adj, n_epochs=400):
        self.user_features = self.user_features.to(self.device)
        self.item_features = self.item_features.to(self.device)
        self.gnn = self.gnn.to(self.device)
        self.labels = self.labels.to(self.device)
        criterion = F.nll_loss
        optimizer = torch.optim.Adam(self.gnn.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best_val_auc = 0.
        best_test_auc = 0.
        best_embs = None
        cnt_wait = 0
        patience = 50
        for epoch in range(n_epochs):
            self.gnn.train()
            self.gnn.zero_grad()
            input_adj = adj.clone()
            input_adj = input_adj.to(self.device)
            nc_logits, _ = self.gnn(input_adj, self.user_features, self.item_features)
            loss = criterion(nc_logits[self.train_nid], self.labels[self.train_nid])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            adj = adj.detach()
            # Validation
            self.gnn.eval()
            with torch.no_grad():
                input_adj = adj.clone()
                input_adj = input_adj.to(self.device)
                nc_logits_eval, embs_eval = self.gnn(input_adj, self.user_features, self.item_features)
            res_training = self.eval_node_cls(nc_logits[self.train_nid].detach(), self.labels[self.train_nid], self.n_classes)
            res = self.eval_node_cls(nc_logits_eval[self.val_nid], self.labels[self.val_nid], self.n_classes)

            if res['auc'] > best_val_auc:
                cnt_wait = 0
                best_val_auc = res['auc']
                test_auc = self.eval_node_cls(nc_logits_eval[self.test_nid], self.labels[self.test_nid], self.n_classes)['auc']
                # if test_auc > best_test_auc:
                best_test_auc = test_auc
                best_logits = nc_logits_eval
                best_embs = embs_eval
                best_res = self.eval_node_cls(nc_logits_eval[self.test_nid], self.labels[self.test_nid], self.n_classes)
            else:
                cnt_wait += 1
            print('epoch {}/{}, loss {:.5f}, training auc {:.5f}, val auc {:.5f}, best test auc {:.5f}'.format(epoch+1, n_epochs, loss.item(), res_training['auc'], res['auc'], best_res['auc']))
            if cnt_wait >= patience:
                self.logger.info('Early stop!')
                break
        return best_logits, best_embs

    @staticmethod
    def eval_node_cls(logits, labels, n_classes):
        logits = logits.cpu().numpy()
        y_pred = np.argmax(logits, axis=1)
        logits = logits.T[1]
        labels = labels.cpu().numpy()

        # fpr, tpr, _ = roc_curve(labels, logits, pos_label=1)
        roc_auc = roc_auc_score(labels, logits)
        # precisions, recalls, _ = precision_recall_curve(labels, logits, pos_label=1)
        # pr_auc = metrics.auc(recalls, precisions)
        ap = average_precision_score(labels, logits, pos_label = 1)
        # f1s = np.nan_to_num(2*precisions*recalls/(precisions+recalls))
        # best_comb = np.argmax(f1s)
        f1 = f1_score(labels, y_pred)
        # pre = precisions[best_comb]
        # rec = recalls[best_comb]
        # calc confusion matrix
        conf_mat = np.zeros((n_classes, n_classes))
        # for i in range(len(preds)):
        # 	conf_mat[labels[i], preds[i]] += 1
        results = {
            # 'pre': pre,
            # 'rec': rec,
            'f1': f1,
            'ap': ap,
            'conf': conf_mat,
            'auc': roc_auc
        }
        return results

class GAU_Model:
    def __init__(self, dim_feats, dim_h, u2idx, idx2feats, p2idx, rnn_type, lstm_dataloader, logger, device, lr, embs, bmloss_type, flag_weight=0.01):
        self.gau = GAU_E(dim_feats=dim_feats+dim_h, dim_h=dim_h, idx2feats=idx2feats, p2idx=p2idx, rnn_type=rnn_type)
        self.u2idx, self.p2idx, self.idx2feats = u2idx, p2idx, idx2feats
        self.logger = logger
        self.lstm_dataloader = lstm_dataloader
        self.device = device
        self.lr = lr
        # For multi-task two-step decoder
        self.flag_weight = flag_weight
        # For user embedding; dim (len(users), hid_size)
        self.embs = embs
        self.bmloss_type = bmloss_type

    def train(self, n_epochs=15):
        """ Train the behavioral modelling network """
        optimizer = torch.optim.Adam(self.gau.parameters(), lr = self.lr)
        if self.bmloss_type == 'mse':
            criterion1 = MSELoss()
        else:
            criterion1 = CosineEmbeddingLoss()
        criterion2 = BCELoss()
        self.gau.train()
        self.gau.to(self.device)
        for epoch in range(n_epochs):
            print('epoch {}/{}'.format(epoch, n_epochs))
            self.gau.zero_grad()
            optimizer.zero_grad()
            cur_loss = []
            for batch_idx, (uids, feats, repeat_flags, feats_len) in enumerate(self.lstm_dataloader):
                # feats: (batch_size, max_len, dim_feats)
                # We need feats + embs: (batch_idx, max_len, dim_feats + dim_h)
                feats = feats.to(self.device).float()

                # Select embeddings:
                # batch_embs: (batch_size, hidden_sz)
                batch_embs = self.embs[[self.u2idx[uid] for uid in uids]]
                # batch_embs = self.embs[[self.u2idx[uid.item()] for uid in uids]]
                batch_embs = batch_embs.unsqueeze(1).repeat(1, 30, 1)

                input_feats = torch.cat((feats, batch_embs), dim=2)
                # repeat_flags = repeat_flags.to(self.device).float()
                out, out_len = self.gau(input_feats, feats_len)
                loss = 0
                for idx in range(len(out_len)):
                    if self.bmloss_type == 'cos':
                        loss += criterion1(out[idx, :out_len[idx]-1, :], feats[idx, 1:out_len[idx], :], torch.cuda.LongTensor([1]))
                    else:
                        loss += criterion1(out[idx, :out_len[idx]-1, :], feats[idx, 1:out_len[idx], :])
                    # loss += 0.1 * criterion2(flag[idx, :out_len[idx]-1], repeat_flags[idx, :out_len[idx]-1])
                loss.backward()
                cur_loss.append(loss.item())
                nn.utils.clip_grad_norm_(self.gau.parameters(), 5)
                optimizer.step()
                optimizer.zero_grad()
                self.gau.zero_grad()

    def inference(self, adj, logits, base_pred):
        """
            logits: output from nc_net, times base_pred to get the final prediction number for each user
        """
        self.gau.eval()
        # logits = F.softmax(logits, dim=1)
        # logits = logits.T[1] # Probability of being malicious
        pred_num = base_pred * logits	# pred_num: (n_users)
        adj = adj.to(self.device)

        with torch.no_grad():
            for batch_idx, (uids, feats, _, feats_length) in enumerate(self.lstm_dataloader):
                feats = feats.to(self.device).float()

                # repeat_flags = repeat_flags
                # match: delta: (batch_size, feature_size) --> (batch_size, dict_size)
                # 		 pred_features: (batch_size, feature_size) --> (batch_size, feature_size)
                for i in range(1, int(max(pred_num).item()+1)):
                    # Select embeddings:
                    # batch_embs: (batch_size, hidden_sz)
                    batch_embs = self.embs[[self.u2idx[uid] for uid in uids]]
                    # batch_embs = self.embs[[self.u2idx[uid.item()] for uid in uids]]
                    batch_embs = batch_embs.unsqueeze(1).repeat(1, feats.shape[1], 1)

                    input_feats = torch.cat((feats, batch_embs), dim=2)

                    out, out_len = self.gau(input_feats, feats_length)
                    delta = out[np.arange(len(out_len)), out_len-1, None]
                    # last_flag = flag[np.arange(len(out_len)), out_len-1, None]

                    # delta: (batch_size, 1, feature_size)
                    # last_flag: (batch_size, 1, 1)
                    delta = delta.squeeze()
                    # flag = flag.squeeze(1)
                    indices = [self.u2idx[uid] for uid in uids]
                    # indices = [self.u2idx[uid.item()] for uid in uids]
                    # u_delta, pred_features = self.match(delta, last_flag, adj[indices])
                    u_delta, pred_features = self.match(delta, adj[indices])
                    # We apply mask based on pred_num
                    for idx, uid in enumerate(uids):
                        if pred_num[self.u2idx[uid]] < i:
                        # if pred_num[self.u2idx[uid.item()]] < i:
                            u_delta[idx] = torch.cuda.FloatTensor(self.idx2feats.size(0)).fill_(0.)
                    adj[indices] += u_delta
                    if max(feats_length) >= feats.size(1):
                        feats = torch.cat((feats, torch.cuda.FloatTensor(feats.size(0), 1, feats.size(2)).fill_(0.)), dim=1)
                        # feats2[:, feats.size(1), :] = torch.zeros(feats.size(2))
                    for idx in range(len(feats_length)):
                        feats[idx][feats_length[idx]] = pred_features.detach()[idx]

        return adj

    def match(self, x, adj): # (self, x, flag, adj)
        """
            x: (batch_size, features_size)
            flag: (batch_size, 1)
        """
        similarity_matrix = self.cosine_similarity(x, self.idx2feats)  # idx2feats: (dict_sz, feat_sz)
        # similarity_matrix: (batch_idx, dict_sz)
        # adj: (batch_idx, dict_sz)
        # for i, f in enumerate(flag): # Applying mask...
        #	if f < 0.5:
        #		similarity_matrix[i] *= (adj[i] == 0)
        #	else:
        #		similarity_matrix[i] *= (adj[i] == 1)

        max_idx = torch.argmax(similarity_matrix, dim=1)
        similarity_matrix = similarity_matrix.zero_()
        similarity_matrix = similarity_matrix.scatter_(1, max_idx.view(-1, 1), 1)
        # similarity_matrix = F.gumbel_softmax(similarity_matrix, tau=0.01, hard=True, dim=1)
        pred_features = self.idx2feats[torch.argmax(similarity_matrix, dim=1)]

        return similarity_matrix, pred_features

    @staticmethod
    def cosine_similarity(x1, x2):
        """
            x1: (batch_size, feature_size); x2: (dict_size, feature_size)
        """
        x2 = x2.T
        return (x1@x2) / ((torch.norm(x1, p=2, dim=1).reshape(-1, 1) @ torch.norm(x2, p=2, dim=0).reshape(1, -1)) + 1e-8)

# Global Function
def scipysp_to_pytorchsp(sp_mx):
    """ converts scipy sparse matrix to pytorch sparse matrix """
    if not sp.isspmatrix_coo(sp_mx):
        sp_mx = sp_mx.tocoo()
    coords = np.vstack((sp_mx.row, sp_mx.col)).transpose()
    values = sp_mx.data
    shape = sp_mx.shape
    pyt_sp_mx = torch.sparse.FloatTensor(torch.LongTensor(coords.T),
                                         torch.FloatTensor(values),
                                         torch.Size(shape))
    return pyt_sp_mx
