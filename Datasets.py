from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch
from glob import glob
import pickle
import gc
from collections import defaultdict
import datetime
import numpy as np


class MyDataSet(Dataset):
	""" Dataset Container """
	def __init__(self, p2index, item_features, edgelist_file, max_len=30, thresh=0.004):
		self.p2idx = p2index
		self.idx2feats = item_features
		self.graph_path = edgelist_file
		self.tmax, self.tmin = None, None
		self.thresh = thresh
		self.init_data()
		if not max_len:
			self.max_len = self.get_max_len()
		else:
			self.max_len = max_len
		self.uids = list(self.u2pt.keys())

	def __len__(self):
		return len(self.u2pt)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		uid = self.uids[idx]
		# Read the data
		data = self.u2pt[uid]
		# Transform the data
		pids, feats, feature_length = self.transform(data)

		# Construct Flags
		repeat_flags = [1 if pid in pids[:idx] else 0 for (idx, pid) in enumerate(pids[1:])]
		target_length = min(len(repeat_flags), self.max_len)
		repeat_flags = repeat_flags + [0] * (self.max_len - target_length) if self.max_len > target_length else repeat_flags[-self.max_len:]
		# Repeat_labels: Binary nd-array. 1-if repeat, 0-if not
		return uid, np.array(feats, dtype=float), np.array(repeat_flags), np.array(feature_length) # torch.FloatTensor(repeat_flags)

	def transform(self, data):
		"""
			data: has shape of (n, 2) where dim-0 represents the number of posts,
					and dim-1 represents pid and time
			The functoin returns 2 lists; first is pids and the second are the features
		"""
		pids = []
		feats = []
		data = sorted(data, key=lambda x: x[1])
		for (pid, t) in data:
			pids.append(pid)
			feats.append(self.idx2feats[self.p2idx[pid]])
		# Using zero to pad sequence
		feature_length = min(len(feats), self.max_len)
		feats = feats + [[0]*len(feats[0]) for _ in range(self.max_len-feature_length)] if self.max_len > feature_length else feats[-self.max_len:]

		return pids, np.array(feats), feature_length

	def normalize_t(self, t):
		return (t - self.tmin) / (self.tmax - self.tmin)

	@staticmethod
	def get_p2feats(feats_path):
		gc.disable()

		idx2feats = np.load(open(feats_path + 'post2vec.npy', 'rb'), allow_pickle=True)
		p2idx = pickle.load(open(feats_path + 'post2idx.pkl', 'rb'))

		gc.enable()

		return idx2feats, p2idx

	def get_uids(self, graph_path):
		uids = glob(graph_path)
		return uids

	def init_data(self):
		self.u2pt = defaultdict(list)
		self.total_edges = 0
		file = open(self.graph_path)
		for line in tqdm(file):
			self.total_edges += 1
			tmp = line.split(',')
			uid, pid, t = tmp[0], tmp[1], int(tmp[2])
			self.u2pt[uid].append([pid, t])
		file.close()
		remove_ulist=[]
		for u in self.u2pt.keys():
			if len(self.u2pt[u]) < 2:
				remove_ulist.append(u)
				self.total_edges -= 1
		for u in remove_ulist:
			del self.u2pt[u]

	def get_max_len(self):
		ret = 0
		for u in self.u2pt.keys():
			cur_len = len(self.u2pt[u])
			if cur_len > ret:
				ret = cur_len
		return ret

