import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GCNLayer(nn.Module):
	""" one layer of GCN """
	def __init__(self, input_dim, output_dim, n_heads, activation, dropout, bias=True):
		super(GCNLayer, self).__init__()
		self.W = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
		self.activation = activation
		if bias:
			self.b = nn.Parameter(torch.FloatTensor(output_dim))
		else:
			self.b = None
		if dropout:
			self.dropout = nn.Dropout(p=dropout)
		else:
			self.dropout = 0
		self.init_params()

	def init_params(self):
		""" Initialize weights with xavier uniform and biases with all zeros """
		stdv = 1. / math.sqrt(self.W.size(1))
		self.W.data.uniform_(-stdv, stdv)
		if self.b is not None:
			self.b.data.uniform_(-stdv, stdv)

	# In order to save GPU memory, we pass n by m matrix instead of bipartite matrix here
	def forward(self, adj, h_u, h_v, D_u, D_v):
		"""
			adj: (n, m) tensor
			h_u: (n, f) tensor; user features
			h_v: (m, f) tensor; item features
			D_u: Normed Degree matrix of U
			D_v: Normed Degree matrix of V
		"""
		if self.dropout:
			h_u = self.dropout(h_u)
			h_v = self.dropout(h_v)
		x_u = h_u @ self.W
		x_v = h_v @ self.W

		x_u = x_tmp_u = x_u * D_u.unsqueeze(1)
		x_v = x_tmp_v = x_v * D_v.unsqueeze(1)

		x_u = adj @ x_v + x_tmp_u
		x_v = adj.T @ x_tmp_u + x_tmp_v

		x_u = x_u * D_u.unsqueeze(1)
		x_v = x_v * D_v.unsqueeze(1)

		if self.b is not None:
			x_u += self.b
			x_v += self.b

		if self.activation is not None:
			x_u = self.activation(x_u)
			x_v = self.activation(x_v)

		return x_u, x_v


class SAGELayer(nn.Module):
	""" one layer of GraphSAGE with gcn aggregator """
	def __init__(self, input_dim, output_dim, n_heads, activation, dropout, bias=True):
		torch.manual_seed(12345)
		super(SAGELayer, self).__init__()
		self.linear_neigh = nn.Linear(input_dim, output_dim, bias=True)
		self.linear_self = nn.Linear(input_dim, output_dim, bias=True)
		self.activation = activation
		if dropout:
			self.dropout = nn.Dropout(p=dropout)
		else:
			self.dropout = 0
		self.init_params()
	def init_params(self):
		""" Initialize weights with xavier uniform and biases with all zeros """
		gain = nn.init.calculate_gain('relu')
		for param in self.parameters():
			if len(param.size()) == 2:
				nn.init.xavier_uniform_(param, gain=gain)
			else:
				nn.init.constant_(param, 0.0)

	def forward(self, adj, h_u, h_v):
		if self.dropout:
			h_u = self.dropout(h_u)
			h_v = self.dropout(h_v)
		x_u = adj @ h_v + h_u 	# h_v: (n, emb)
		x_v = adj.T @ h_u + h_v 	# h_u: (m, emb)

		degs_u = torch.sum(adj, dim=1).squeeze() + 1
		degs_v = torch.sum(adj, dim=0).squeeze() + 1

		x_u = x_u / degs_u.unsqueeze(-1)
		x_v = x_v / degs_v.unsqueeze(-1)

		x_u = self.linear_neigh(x_u)
		x_v = self.linear_neigh(x_v)

		# x_u_self = self.linear_self(h_u)
		# x_v_self = self.linear_self(h_v)

		# alpha_u = self.a_neigh(x_u) + self.a_self(x_u_self)
		# alpha_v = self.a_neigh(x_v) + self.a_self(x_v_self)

		# x_u = sx_u_neigh + x_u_self
		# x_v = x_v_neigh + x_v_self

		if self.activation:
			x_u = self.activation(x_u)
			x_v = self.activation(x_v)

		# x_u = F.normalize(x_u, dim=1, p=2)
		# x_v = F.normalize(x_v, dim=1, p=2)
		return x_u, x_v

class HetLayer(nn.Module):
	""" One Layer for Heterolgenous GCN with mean-aggregator"""
	def __init__(self, input_dim, output_dim, n_heads, activation, dropout, bias=True):
		super(HetLayer, self).__init__()
		self.linear_neigh = nn.Linear(input_dim, output_dim, bias=bias)
		self.linear_self = nn.Linear(input_dim, output_dim, bias=bias)
		# Attention
		self.a_neigh = nn.Linear(output_dim, 1, bias=bias)
		self.a_self = nn.Linear(output_dim, 1, bias=bias)

		self.activation = activation
		if dropout:
			self.dropout = nn.Dropout(p=dropout)
		else:
			self.dropout = 0
		self.init_params()

	def init_params(self):
		""" Initialize weights with xavier uniform and biases with all zeros """
		gain = nn.init.calculate_gain('relu')
		for param in self.parameters():
			if len(param.size()) == 2:
				nn.init.xavier_uniform_(param, gain=gain)
			else:
				nn.init.constant_(param, 0.0)

	def forward(self, adj, h_u, h_v):
		if self.dropout:
			h_u = self.dropout(h_u)
			h_v = self.dropout(h_v)
		x_u = adj @ h_v + h_u 	# h_v: (n, emb)
		x_v = adj.T @ h_u + h_v 	# h_u: (m, emb)

		degs_u = torch.sum(adj, dim=1).squeeze() + 1
		degs_v = torch.sum(adj, dim=0).squeeze() + 1

		x_u = x_u / degs_u.unsqueeze(-1)
		x_v = x_v / degs_v.unsqueeze(-1)

		x_u = self.linear_neigh(x_u)
		x_v = self.linear_neigh(x_v)

		x_u_self = self.linear_self(h_u)
		x_v_self = self.linear_self(h_v)

		# Attention
		alpha_u = torch.sigmoid(self.a_neigh(x_u) + self.a_self(x_u_self))
		alpha_v = torch.sigmoid(self.a_neigh(x_v) + self.a_self(x_v_self))

		# Softmax
		alpha_u = F.softmax(torch.cuda.FloatTensor([alpha_u[0], 1]), dim=0)
		alpha_v = F.softmax(torch.cuda.FloatTensor([alpha_v[0], 1]), dim=0)

		# Neighbour + self
		x_u = alpha_u[0] * x_u + alpha_u[1] * x_u_self
		x_v = alpha_v[0] * x_v + alpha_v[1] * x_v_self

		if self.activation:
			x_u = self.activation(x_u)
			x_v = self.activation(x_v)

		# x_u = F.normalize(x_u, dim=1, p=2)
		# x_v = F.normalize(x_v, dim=1, p=2)

		return x_u, x_v
