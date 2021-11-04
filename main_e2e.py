import os
import sys
import json
import argparse
import pickle as pk
from collections import Counter
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix, coo_matrix
import torch
from torch.utils.data import DataLoader

from model.Eland_e2e import Eland_e2e
from model.Eland_e2e_unsup import Eland_e2e_uns
from Datasets import MyDataSet

def parseArgs():
    arg_parser = argparse.ArgumentParser(description='Helper')
    arg_parser.add_argument('--log_name', default='debug', type=str)
    arg_parser.add_argument('--dataset', default='reddit', type=str)
    arg_parser.add_argument('--graph_num', default=1, type=int)
    arg_parser.add_argument('--method', default='gcn', type=str)
    arg_parser.add_argument('--rnn', default='gru', type=str)
    arg_parser.add_argument('--gpu', type=int, default=-1)
    arg_parser.add_argument('--baseline', action='store_true')

    args = arg_parser.parse_args()
    args.argv = sys.argv
    if args.gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    args.device = torch.device('cuda:0')

    return args

def init_vars(ds):
    """ Initialize u2index, labels, train/validation/test indices """
    u_all = set()
    pos_uids = set()
    labeled_uids = set()
    with open(f'../data/{ds}/userlabels', 'r') as f:
        for line in f:
            arr = line.strip('\r\n').split(',')
            u_all.add(arr[0])
            if arr[1] == 'anomaly':
                pos_uids.add(arr[0])
                labeled_uids.add(arr[0])
            elif arr[1] == 'benign':
                labeled_uids.add(arr[0])
    print(f'loaded labels, total of {len(pos_uids)} positive users and {len(labeled_uids)} labeled users')

    # get users' features
    u2index = pk.load(open(f'../data/{ds}/u2index.pkl', 'rb'))
    user_feats = np.load(open(f'../data/{ds}/user2vec.npy', 'rb'), allow_pickle=True)
    # Get prod features
    p2index = pk.load(open(f'../data/{ds}/p2index.pkl', 'rb'))
    item_feats = np.load(open(f'../data/{ds}/prod2vec.npy', 'rb'), allow_pickle=True)

    labels = np.zeros(len(u2index))
    for u in u2index:
        if u in pos_uids:
            labels[u2index[u]] = 1
    labels = labels.astype(int)

    tvt_file = f'../data/{ds}/tvt_idx.pkl'
    if os.path.isfile(tvt_file):
        tvt_idx = pk.load(open(tvt_file, 'rb'))
        idx_train, idx_val, idx_test = tvt_idx
    else:
        n_train = int(len(u2index) * 0.2)
        n_val = n_train
        n_test = len(u2index) - n_train - n_val
        idx_labeled = np.arange(len(u2index))
        np.random.shuffle(idx_labeled)
        idx_train = idx_labeled[:n_train]
        idx_val = idx_labeled[n_train: n_train+n_val]
        idx_test = idx_labeled[n_train+n_val: n_train + n_val + n_test]
        tvt_idx = (idx_train, idx_val, idx_test)
        pk.dump(tvt_idx, open(tvt_file, 'wb'))
    print('Train: total of {:5} users with {:5} pos users and {:5} neg users'.format(len(idx_train), np.sum(labels[idx_train]), len(idx_train)-np.sum(labels[idx_train])))
    print('Val:   total of {:5} users with {:5} pos users and {:5} neg users'.format(len(idx_val), np.sum(labels[idx_val]), len(idx_val)-np.sum(labels[idx_val])))
    print('Test:  total of {:5} users with {:5} pos users and {:5} neg users'.format(len(idx_test), np.sum(labels[idx_test]), len(idx_test)-np.sum(labels[idx_test])))

    return u2index, labels, tvt_idx, user_feats, p2index, item_feats

def load_graph(ds, graph_num, u2index, p2index):
    """ Get graph, graph features, and initialize u2index, p2index """
    edges = Counter()
    n = int(graph_num * 10)
    edgelist_file = f'../data/{ds}/splitted_edgelist_{n}' if n < 10 else f'../data/{ds}/edgelist'
    with open(edgelist_file, 'r') as f:
        for line in f:
            arr = line.strip('\r\n').split(',')
            u = arr[0]
            p = arr[1]
            t = int(arr[2])
            # if p not in p2index:
            # 	p2index[p] = len(p2index)
            edges[(u2index[u], p2index[p])] += 1
    # Construct the graph
    row = []
    col = []
    entry = []
    for edge, w in edges.items():
        i, j = edge
        row.append(i)
        col.append(j)
        entry.append(w)
    graph = csr_matrix((entry, (row, col)), shape=(len(u2index), len(p2index)))
    return graph

def load_data_weibo(graph_num):
    """ Initialize u2index, labels, train/validation/test indices """
    u_all = set()
    pos_uids = set()
    with open('../data/weibo/userlabels', 'r') as f:
        for line in f:
            arr = line.strip('\r\n').split(',')
            uid = arr[0]
            u_all.add(uid)
            if arr[1] == 'anomaly':
                pos_uids.add(uid)
    print(f'loaded labels, total of {len(u_all)} users with {len(pos_uids)} positive users')
    u_all = list(u_all)
    np.random.shuffle(u_all)
    u2index = {}
    labels = []
    for u in u_all:
        u2index[u] = len(u2index)
        if u in pos_uids:
            labels.append(1)
        else:
            labels.append(0)
    labels = np.array(labels)
    n_train = 5000
    n_val = 5000
    n_test = len(u_all) - 10000
    idx_train = np.arange(n_train)
    idx_val = np.arange(n_train, n_train+n_val)
    idx_test = np.arange(n_train+n_val, n_train + n_val + n_test)
    tvt_idx = (idx_train, idx_val, idx_test)
    print('Train: total of {:5} users with {:5} pos users and {:5} neg users'.format(len(idx_train), np.sum(labels[idx_train]), len(idx_train)-np.sum(labels[idx_train])))
    print('Val:   total of {:5} users with {:5} pos users and {:5} neg users'.format(len(idx_val), np.sum(labels[idx_val]), len(idx_val)-np.sum(labels[idx_val])))
    print('Test:  total of {:5} users with {:5} pos users and {:5} neg users'.format(len(idx_test), np.sum(labels[idx_test]), len(idx_test)-np.sum(labels[idx_test])))
    # Get Features
    item_features = np.load(open('../data/weibo/prod2vec.npy', 'rb'), allow_pickle=True)
    p2index_i = pk.load(open('../data/weibo/p2index.pkl', 'rb'))
    p2index = {}
    for p, i in p2index_i.items():
        p2index[str(p)] = i

    """ Get graph, graph features, and initialize u2index, p2index """
    edges = Counter()
    n = int(graph_num * 10)
    edgelist_file = f'../data/weibo/splitted_edgelist_{n}' if n < 10 else '../data/weibo/edgelist'
    with open(edgelist_file, 'r') as f:
        for line in f:
            arr = line.strip('\r\n').split(',')
            u = arr[0]
            p = arr[1]
            t = int(arr[2])
            assert p in p2index
            edges[(u2index[u], p2index[p])] += 1
    # Construct the graph
    row = []
    col = []
    entry = []
    for edge, w in edges.items():
        i, j = edge
        row.append(i)
        col.append(j)
        entry.append(w)
    graph = csr_matrix((entry, (row, col)), shape=(len(u2index), len(p2index)))
    # Construct features
    user_features = np.zeros((len(u2index), 300))
    for u, index in u2index.items():
        cur_row = graph.getrow(index)
        user_features[index] = cur_row.dot(item_features)
    # normalize the user_features
    w = np.sum(graph, axis = 1)
    user_features = user_features / w

    return u2index, labels, tvt_idx, user_features, p2index, item_features, graph

def load_data(ds, graph_num):
    """ Initialize u2index, labels, train/validation/test indices """
    u_all = set()
    pos_uids = set()
    labeled_uids = set()
    with open(f'../data/{ds}/userlabels', 'r') as f:
        for line in f:
            arr = line.strip('\r\n').split(',')
            u_all.add(arr[0])
            if arr[1] == 'anomaly':
                pos_uids.add(arr[0])
                labeled_uids.add(arr[0])
            elif arr[1] == 'benign':
                labeled_uids.add(arr[0])
    print(f'loaded labels, total of {len(pos_uids)} positive users and {len(labeled_uids)} labeled users')

    # get users' features
    u2index = pk.load(open(f'../data/{ds}/u2index.pkl', 'rb'))
    user_feats = np.load(open(f'../data/{ds}/user2vec.npy', 'rb'), allow_pickle=True)
    # Get prod features
    p2index = pk.load(open(f'../data/{ds}/p2index.pkl', 'rb'))
    item_feats = np.load(open(f'../data/{ds}/prod2vec.npy', 'rb'), allow_pickle=True)

    labels = np.zeros(len(u2index))
    for u in u2index:
        if u in pos_uids:
            labels[u2index[u]] = 1
    labels = labels.astype(int)

    tvt_idx = pk.load(open(f'../data/{ds}/tvt_idx.pkl', 'rb'))
    idx_train, idx_val, idx_test = tvt_idx
    print('Train: total of {:5} users with {:5} pos users and {:5} neg users'.format(len(idx_train), np.sum(labels[idx_train]), len(idx_train)-np.sum(labels[idx_train])))
    print('Val:   total of {:5} users with {:5} pos users and {:5} neg users'.format(len(idx_val), np.sum(labels[idx_val]), len(idx_val)-np.sum(labels[idx_val])))
    print('Test:  total of {:5} users with {:5} pos users and {:5} neg users'.format(len(idx_test), np.sum(labels[idx_test]), len(idx_test)-np.sum(labels[idx_test])))

    """ Get graph, graph features, and initialize u2index, p2index """
    edges = Counter()
    n = int(graph_num * 10)
    edgelist_file = f'../data/{ds}/splitted_edgelist_{n}' if n < 10 else f'../data/{ds}/edgelist'
    with open(edgelist_file, 'r') as f:
        for line in f:
            arr = line.strip('\r\n').split(',')
            u = arr[0]
            p = arr[1]
            t = int(arr[2])
            edges[(u2index[u], p2index[p])] += 1
    # Construct the graph
    row = []
    col = []
    entry = []
    for edge, w in edges.items():
        i, j = edge
        row.append(i)
        col.append(j)
        entry.append(w)
    graph = csr_matrix((entry, (row, col)), shape=(len(u2index), len(p2index)))
    return u2index, labels, tvt_idx, user_feats, p2index, item_feats, graph

def main(ds, graph_num=0.1, name='debug', baseline=False, gnnlayer_type='gcn', rnnlayer_type='lstm', device='cpu'):
    # Graph
    if ds == 'weibo':
        u2index, labels, tvt_nids, user_features, p2index, item_features, graph = load_data_weibo(graph_num)
    else:
        u2index, labels, tvt_nids, user_features, p2index, item_features, graph = load_data(ds, graph_num)
    if ds == 'amazon':
        base_pred = 500
    else:
        base_pred = 30
    # DataLoader
    n = int(graph_num * 10)
    edgelist_file = f'../data/{ds}/splitted_edgelist_{n}' if n < 10 else f'../data/{ds}/edgelist'
    dataset = MyDataSet(p2index, item_features, edgelist_file)
    lstm_dataloader = DataLoader(dataset, batch_size=300)
    if args.method in ('dominant', 'deepae'):
        eland = Eland_e2e_uns(graph, lstm_dataloader, user_features,
                item_features, labels, tvt_nids, u2index,
                p2index, item_features, lr=0.01, n_layers=2, name=name, pretrain_bm=25,
                pretrain_nc=25, epochs=400, method=args.method, rnn_type=rnnlayer_type, bmloss_type='mse', device=device, base_pred=base_pred)
    else:
        eland = Eland_e2e(graph, lstm_dataloader, user_features,
                item_features, labels, tvt_nids, u2index,
                p2index, item_features, lr=0.01, n_layers=2, name=name, pretrain_bm=25,
                pretrain_nc=300, gnnlayer_type=gnnlayer_type, rnn_type=rnnlayer_type, bmloss_type='mse', device=device, base_pred=base_pred)
    if not baseline:
        auc, ap = eland.train()
    else:
        auc, ap = eland.pretrain_nc_net(n_epochs=300)
    return auc, ap


if __name__ == '__main__':
    args = parseArgs()
    rates = [args.graph_num/10]

    for rate in rates:
        auc_res, ap_res = [], []
        for _ in range(20):
            auc, ap = main(args.dataset, rate, name=f'{args.dataset}_{args.method}_{rate}_{args.rnn}', baseline=args.baseline, gnnlayer_type=args.method, rnnlayer_type=args.rnn, device=args.device)
            auc_res.append(auc)
            ap_res.append(ap)
        with open(f'ELANDe2e_{args.dataset}_{args.method}_{rate}_{args.rnn}_result.txt', 'a') as f:
            f.write(f'auc: {np.mean(auc_res)} +- {np.std(auc_res)}, ap: {np.mean(ap_res)} +- {np.std(ap_res)}\n')
