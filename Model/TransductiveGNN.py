import torch
import torch.nn as nn
from Model.AttentionBlock import AttentionBlock

class TransductiveGNN(nn.Module):
	"""GNN for transductive learning"""
	def __init__(self, in_feature, out_feature, n_heads=(8, 1), dropout=0.5):
		super().__init__()
		self.n_heads = n_heads
		for i in range(n_heads[0]):
			setattr(self, f"layer1Head{i}", AttentionBlock(in_feature, 8, dropout=dropout))

		# Only 1 attention head for classification layer
		for i in range(n_heads[1]):
			setattr(self, f"layer2Head{i}", AttentionBlock(8*n_heads[0], out_feature, dropout=dropout))

		self.ELU = nn.ELU()
		self.dropout = nn.Dropout(p=dropout)
		self.softmax = nn.Softmax(dim=1)

	def forward(self, x):
		n_nodes = x['x'].size()[0]
		layer_list = self.__dict__['_modules']

		# Layer 1
		x['x'] = self.dropout(x['x'])
		# Concatenate features for each head at -1 dim
		head_output1 = torch.empty((n_nodes, 0), device='cuda')
		for i in range(self.n_heads[0]):
			x_tmp = x.clone()
			head_output_tmp = layer_list[f"layer1Head{i}"](x_tmp)
			head_output_tmp['x'] = self.ELU(head_output_tmp['x'])
			head_output1 = torch.cat((head_output1, head_output_tmp['x']), dim=-1)
		x['x'] = head_output1

		# Layer 2
		x['x'] = self.dropout(x['x'])
		# Concatenate features for each head at -1 dim
		head_output2 = torch.empty((n_nodes, 0), device='cuda')
		for i in range(self.n_heads[1]):
			x_tmp = x.clone()
			head_output_tmp = layer_list[f"layer2Head{i}"](x_tmp)
			head_output_tmp['x'] = self.ELU(head_output_tmp['x'])
			head_output2 = torch.cat((head_output2, head_output_tmp['x']), dim=-1)
		x['x'] = head_output2

		x['x'] = self.softmax(x['x'])
		return x
