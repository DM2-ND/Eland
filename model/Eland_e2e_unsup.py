import logging
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from torch.nn import MSELoss, CosineEmbeddingLoss
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score, roc_auc_score, f1_score
from tqdm import tqdm
from .utils import MultipleOptimizer
from .gcn_layers import GCNLayer, SAGELayer, HetLayer
from .model_zoo import GAU_E, GNN, Dominant, DeepAE
import matplotlib.pyplot as plt
# from torch.utils.tensorboard import SummaryWriter
import math

class Eland_e2e_uns(object):
    def __init__(self, adj_matrix, lstm_dataloader, user_features, item_features,
            labels, tvt_nids, u2index, p2index, idx2feats, dim_feats=300, cuda=0, hidden_size=128, n_layers=2,
            epochs=400, seed=-1, lr=0.0001, weight_decay=1e-5, dropout=0.4, tensorboard=False,
            log=True, name='debug', method='dominant', rnn_type='lstm', pretrain_bm=25, pretrain_nc=300, alpha=0.05, bmloss_type='mse', device='cuda', base_pred=20):
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_epochs = epochs
        self.pretrain_bm = pretrain_bm
        self.pretrain_nc = pretrain_nc
        self.n_classes = len(np.unique(labels))
        self.alpha = alpha
        self.labels = labels
        self.train_nid, self.val_nid, self.test_nid = tvt_nids
        self.bmloss_type = bmloss_type
        self.base_pred = base_pred
        if log:
            self.logger = self.get_logger(name)
        else:
            self.logger = logging.getLogger()
        # if not torch.cuda.is_available():
        # 	cuda = -1
        # self.device = torch.device(f'cuda:{cuda}' if cuda >= 0 else 'cpu')
        self.device = device
        # Log parameters for reference
        all_vars = locals()
        self.log_parameters(all_vars)
        # Fix random seed if needed
        if seed > 0:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        # load data
        self.load_data(adj_matrix, user_features, item_features, self.labels, tvt_nids, idx2feats)
        idx2feats = torch.cuda.FloatTensor(idx2feats)
        # idx2feats = idx2feats.to(self.device)
        self.model = Eland_Model(dim_feats, hidden_size, lstm_dataloader, self.n_classes, n_layers,
                            u2idx = u2index, p2idx = p2index, idx2feats = idx2feats, dropout=dropout,
                            device=self.device, rnn_type=rnn_type, method=method, activation=F.relu, bmloss_type=bmloss_type)

    def load_data(self, adj_matrix, user_features, item_features, labels, tvt_nids, idx2feats):
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

    def pretrain_bm_net(self, n_epochs=25):
        """ pretrain the behavioral modelling network """
        optimizer = torch.optim.Adam(self.model.bm_net.parameters(), lr = self.lr*5)
        if self.bmloss_type == 'mse':
            criterion = MSELoss()
        elif self.bmloss_type == 'cos':
            criterion = CosineEmbeddingLoss()
        self.model.bm_net.train()
        self.model.bm_net.to(self.device)
        for epoch in range(n_epochs):
            self.model.bm_net.zero_grad()
            optimizer.zero_grad()
            cur_loss = []
            for batch_idx, (uids, feats, _, feats_len) in enumerate(self.model.loader):
                feats = feats.to(self.device).float()
                loss = 0
                out, out_len = self.model.bm_net(feats, feats_len)
                for idx in np.arange(len(out_len)):
                    if self.bmloss_type == 'cos':
                        # loss += criterion(out[idx, :out_len[idx]-1, :], feats[idx, 1:out_len[idx], :], torch.cuda.LongTensor([1]))
                        loss += criterion(out[idx, :out_len[idx]-1, :], feats[idx, 1:out_len[idx], :], torch.LongTensor([1]).to(self.device))
                    else:
                        loss += criterion(out[idx, :out_len[idx]-1, :], feats[idx, 1:out_len[idx], :])
                # print('--------')
                # print(torch.isnan(out[idx, :out_len[idx]-1, :]).sum(), torch.isnan(feats[idx, :out_len[idx]-1, :]).sum())
                # print(torch.isnan(out).sum(), torch.isnan(feats).sum())
                # print(loss)
                loss.backward()
                cur_loss.append(loss.item())
                nn.utils.clip_grad_norm_(self.model.bm_net.parameters(), 5)
                optimizer.step()
                optimizer.zero_grad()
                self.model.bm_net.zero_grad()
            self.logger.info(f'BM Module pretrain, Epoch {epoch+1}/{n_epochs}: loss {round(np.mean(cur_loss), 8)}')

    def pretrain_nc_net(self, n_epochs=300):
        """ pretrain the node classification network """
        optimizer = torch.optim.Adam(self.model.nc_net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best_val_auc = 0.
        best_test_auc = 0.
        best_res = None
        self.model.nc_net.to(self.device)
        user_features = self.user_features.to(self.device)
        item_features = self.item_features.to(self.device)
        self.labels = self.labels.to(self.device)

        cnt_wait = 0
        patience = 50
        for epoch in range(n_epochs):
            self.model.nc_net.train()
            self.model.nc_net.zero_grad()
            input_adj = self.adj.clone()
            input_adj = input_adj.to(self.device)
            loss, nc_logits, _ = self.model.nc_net(input_adj, user_features, item_features)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # Detach from computation graph
            self.adj = self.adj.detach()
            # Validation
            self.model.nc_net.eval()
            with torch.no_grad():
                input_adj = self.adj.clone()
                input_adj = input_adj.to(self.device)
                _, nc_logits_eval, _ = self.model.nc_net(input_adj, user_features, item_features)
            res_training = self.eval_node_cls(nc_logits[self.train_nid].detach(), self.labels[self.train_nid], self.n_classes)
            res = self.eval_node_cls(nc_logits_eval[self.val_nid], self.labels[self.val_nid], self.n_classes)

            if res['auc'] > best_val_auc:
                cnt_wait = 0
                best_val_auc = res['auc']
                test_auc = self.eval_node_cls(nc_logits_eval[self.test_nid], self.labels[self.test_nid], self.n_classes)['auc']
                if test_auc > best_test_auc:
                    best_test_auc = test_auc
                    best_res = self.eval_node_cls(nc_logits_eval[self.test_nid], self.labels[self.test_nid], self.n_classes)
                self.logger.info('NCNet pretrain, Epoch [{} / {}]: loss {:.4f}, train_auc {:.4f}, val_auc {:.4f}, test_auc {:.4f}, test_ap {:.4f}'
                        .format(epoch+1, n_epochs, loss.item(), res_training['auc'], res['auc'], test_auc, best_res['ap']))
            else:
                cnt_wait += 1
                self.logger.info('NCNet pretrain, Epoch [{} / {}]: loss {:.4f}, train_auc {:.4f}, val_auc: {:.4f}'
                        .format(epoch+1, n_epochs, loss.item(), res_training['auc'], res['auc']))

            if cnt_wait >= patience:
                self.logger.info('Early stop!')
                break

        self.logger.info('Best Test Results: auc {:.4f}, ap {:.4f}'.format(best_res['auc'], best_res['ap']))
        return best_res['auc'], best_res['ap']

    def train(self):
        """ End-to-end training for bm_net and nc_net """
        # For debugging
        torch.autograd.set_detect_anomaly(True)
        # Move variables to device if haven't done so
        self.user_features = self.move_to_cuda(self.user_features, self.device)
        self.item_features = self.move_to_cuda(self.item_features, self.device)
        self.labels = self.move_to_cuda(self.labels, self.device)
        self.model = self.model.to(self.device)
        # Pretrain
        if self.pretrain_bm > 0:
            self.pretrain_bm_net(self.pretrain_bm)
        if self.pretrain_nc > 0:
            self.pretrain_nc_net(self.pretrain_nc)
        # optimizers
        optims = MultipleOptimizer(torch.optim.Adam(self.model.bm_net.parameters(), lr=self.lr),
                                torch.optim.Adam(self.model.nc_net.parameters(), lr=self.lr, weight_decay=self.weight_decay))
        # optims = torch.optim.Adam(self.model.parameters(), lr = self.lr)

        # criterion = nn.CrossEntropyLoss()
        criterion = F.nll_loss
        best_test_auc = 0.
        best_val_auc = 0.
        best_res = None
        cnt_wait = 0
        patience = 50
        # Training...
        for epoch in range(self.n_epochs):
            self.model.train()
            self.model.zero_grad()
            input_adj = self.adj.clone()
            input_adj = input_adj.to(self.device)
            nc_logits, modified_adj, bm_loss, nc_loss = self.model(input_adj, self.user_features, self.item_features, self.n_epochs, epoch)
            loss = nc_loss + bm_loss * self.alpha
            optims.zero_grad()
            loss.backward()
            # for name, params in self.model.named_parameters():
            # 	if params.requires_grad:
            # 		print(f'{name}: requires grad')
            # 		print(torch.sum(params.grad))
            optims.step()
            # Computation Graph
            # Validation
            self.model.eval()
            with torch.no_grad():
                # input_adj = self.adj.clone()
                # input_adj = input_adj.to(self.device)
                # nc_logits_eval_original, _ = self.model.nc_net(input_adj, self.user_features, self.item_features)
                # input_adj = self.adj.clone()
                # input_adj = input_adj.to(self.device)
                nc_logits_eval_modified, _, _, _ = self.model(input_adj, self.user_features, self.item_features, self.n_epochs, epoch)
            training_res = self.eval_node_cls(nc_logits[self.train_nid].detach(), self.labels[self.train_nid], self.n_classes)
            # res = self.eval_node_cls(nc_logits_eval_original[self.val_nid], self.labels[self.val_nid], self.n_classes)
            res_modified = self.eval_node_cls(nc_logits_eval_modified[self.val_nid], self.labels[self.val_nid], self.n_classes)
            if res_modified['auc'] > best_val_auc:
                cnt_wait = 0
                best_val_auc = res_modified['auc']
                # res_test = self.eval_node_cls(nc_logits_eval_original[self.test_nid], self.labels[self.test_nid], self.n_classes)
                res_test_modified = self.eval_node_cls(nc_logits_eval_modified[self.test_nid], self.labels[self.test_nid], self.n_classes)
                if res_test_modified['auc'] > best_test_auc:
                    best_test_auc = res_test_modified['auc']
                    best_res = res_test_modified
                self.logger.info('Eland Training, Epoch [{}/{}]: loss {:.4f}, train_auc: {:.4f}, val_auc {:.4f}, test_auc {:.4f}, test_ap {:.4f}'
                        .format(epoch+1, self.n_epochs, loss.item(), training_res['auc'], res_modified['auc'], res_test_modified['auc'], res_test_modified['ap']))
            else:
                cnt_wait += 1
                self.logger.info('Eland Training, Epoch [{}/{}]: loss {:.4f}, train_auc: {:.4f}, val_auc {:.4f}'
                        .format(epoch+1, self.n_epochs, loss.item(), training_res['auc'], res_modified['auc']))

            if cnt_wait >= patience:
                self.logger.info('Early stop!')
                break
        self.logger.info('Best Test Results: auc {:.4f}, ap {:.4f}'.format(best_res['auc'], best_res['ap']))

        return best_res['auc'], best_res['ap']

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

    @staticmethod
    def transform_mat(matrix):
        """
            Since in the original matrix, there are items that have zero degree, we add a small delta in order to calculate the norm properly
        """
        delta = 1e-5
        matrix = matrix + delta
        return matrix

    @staticmethod
    def move_to_cuda(var, device):
        if not var.is_cuda:
            return var.to(device)
        else:
            return var

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
            fh = logging.FileHandler(f'logs/ELANDe2e-{name}.log')
            fh.setFormatter(formatter)
            fh.setLevel(logging.INFO)
            logger.addHandler(fh)
        return logger

    @staticmethod
    def eval_node_cls(logits, labels, n_classes):
        logits = logits.cpu().numpy()
        # y_pred = np.argmax(logits, axis=1)
        # logits = logits.T[1]
        labels = labels.cpu().numpy()

        # fpr, tpr, _ = roc_curve(labels, logits, pos_label=1)
        roc_auc = roc_auc_score(labels, logits)
        # precisions, recalls, _ = precision_recall_curve(labels, logits, pos_label=1)
        ap = average_precision_score(labels, logits, pos_label = 1)
        # f1 = f1_score(labels, y_pred)
        # conf_mat = np.zeros((n_classes, n_classes))
        results = {
            # 'f1': f1,
            'ap': ap,
            # 'conf': conf_mat,
            'auc': roc_auc
        }

        return results

class Eland_Model(nn.Module):
    def __init__(self, dim_feats, dim_h, lstm_dataloader, n_classes, n_layers, activation,
                dropout, device, method, u2idx, p2idx, idx2feats, bmloss_type, rnn_type):
        super(Eland_Model, self).__init__()

        self.device = device
        self.loader = lstm_dataloader
        self.u2idx, self.p2idx = u2idx, p2idx
        self.idx2feats = idx2feats.to(self.device)

        # Behavior Modelling
        self.bm_net = GAU_E(dim_feats, dim_h, idx2feats, p2idx, rnn_type=rnn_type, out_sz = 300)

        # Node Classification
        if method == 'dominant':
            self.nc_net = Dominant(dim_feats, dim_h, dropout=0, alpha=0.5)
        elif method == 'deepae':
            self.nc_net = DeepAE(dim_feats, dim_h, dropout=0, alpha=0.025)
        # self.nc_net = GNN(dim_feats, dim_h, n_classes, n_layers, activation, dropout, gnnlayer_type=gnnlayer_type)

        # bm_loss
        self.bmloss_type = bmloss_type
        if bmloss_type == 'mse':
            self.criterion = MSELoss()
        elif bmloss_type == 'cos':
            self.criterion = CosineEmbeddingLoss()

    def forward(self, original_adj, user_features, item_features, total_epochs=100, cur_epoch=100):
        # num_pred = self.base_pred
        # Behavior Modelling as Graph Augmentation through Delta
        # TODO: Potential improvement
        bm_loss = 0 # init loss for future backward
        for batch_idx, (uids, feats, _, feats_length) in enumerate(self.loader):
            feats = feats.to(self.device).float()
            num_pred = torch.mul(torch.true_divide(feats_length, self.loader.dataset.total_edges) ,4000)
            # num_pred = torch.mul(torch.div(feats_length, self.loader.dataset.total_edges), 4000.0)
            num_pred = torch.floor(num_pred)
            num_pred = num_pred.to(self.device)

            # Using next element as label
            out, out_len = self.bm_net(feats, feats_length)
            for idx in np.arange(len(out_len)):
                if self.bmloss_type == 'cos':
                    bm_loss += self.criterion(out[idx, :out_len[idx]-1, :], feats[idx, 1:out_len[idx], :], torch.cuda.LongTensor([1]))
                else:
                    bm_loss += self.criterion(out[idx, :out_len[idx]-1, :], feats[idx, 1:out_len[idx], :])

            #delta: (batch_size, 1, feature_size)
            delta = out[np.arange(len(out_len)), out_len-1, None]
            delta = delta.squeeze()

            # Store intermediary vars
            feats2 = feats.clone()

            for i in range(1, torch.max(num_pred).int()+1):
                u_delta, pred_features = self.match(delta, 0.5 + 4.5 * (total_epochs-cur_epoch)/total_epochs)
                tmp = i <= num_pred
                # indices = [self.u2idx[uid.item()] for uid in uids]
                indices = [self.u2idx[uid] for uid in uids]
                # Update Graph
                original_adj[indices] += torch.mul(tmp.unsqueeze(1).repeat(1, u_delta.size(1)), u_delta)
                if max(feats_length) >= feats.size(1):
                    feats2 = torch.cat((feats, torch.cuda.FloatTensor(feats.size(0), 1, feats.size(2)).fill_(0.).to(self.device)), dim=1)
                for idx in range(len(feats_length)):
                    feats2[idx][feats_length[idx]] = pred_features.detach()[idx]
                out, out_len = self.bm_net(feats2, feats_length)
                delta = out[np.arange(len(out_len)), out_len-1, None]
                #delta: (batch_size, 1, feature_size)
                delta = delta.squeeze()
        # Update features
        user_features = original_adj @ item_features
        nc_loss, nc_logits, _ = self.nc_net(original_adj, user_features, item_features)

        return nc_logits, original_adj, bm_loss, nc_loss

    def match(self, x, tau):
        """
            x: (batch_size, features_size)
        """
        # match: delta: (batch_size, feature_size) --> (batch_size, dict_size)
        # pred_features: (batch_size, feature_size) --> (batch_size, feature_size)
        similarity_matrix = self.cosine_similarity(x, self.idx2feats)  # idx2feats: (dict_sz, feat_sz)
        similarity_matrix = F.gumbel_softmax(similarity_matrix, tau=tau, hard=True, dim=1)
        pred_features = self.idx2feats[torch.argmax(similarity_matrix, dim=1)]

        return similarity_matrix, pred_features

    @staticmethod
    def min_max(mat):
        return (mat - torch.min(mat, dim=1)[0].reshape(-1, 1)) / (torch.max(mat, dim=1)[0] - torch.min(mat, dim=1)[0]).reshape(-1, 1)

    @staticmethod
    def cosine_similarity(x1, x2):
        """
            x1: (batch_size, feature_size); x2: (dict_size, feature_size)
        """
        x2 = x2.T
        return (x1@x2) / ((torch.norm(x1, p=2, dim=1).reshape(-1, 1) @ torch.norm(x2, p=2, dim=0).reshape(1, -1)) + 1e-8)


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

# tmp functions for debugging
def get_tensor_size(tensor):
    return tensor.element_size() * tensor.nelement()
def getBack(var_grad_fn):
    print(var_grad_fn)
    print(len(var_grad_fn.next_functions))
def get_sum_weights(model):
    s = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            s+=torch.sum(parameter.data)
    return s
