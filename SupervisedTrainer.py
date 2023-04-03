import torch
import torch.optim as optim

class SupervisedTrainer:
	"""Trainer Class"""
	def __init__(self, net, lr, l2=1e-3, criterion=torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=100)):
		self.net = net
		self.criterion = criterion
		self.opt = optim.Adam(self.net.parameters(), lr, weight_decay=l2)

	def one_epoch(self, data_loader):
		self.net.train()
		for graph in data_loader:
			graph.to('cuda')
			self.opt.zero_grad()

			if hasattr(graph, 'train_mask'):
				# Cora dataset: 'train_mask', 'test_mask', 'val_mask' provided
				# Process training_labels: assign class 100 to be ignored for CE loss
				train_labels = graph.y + (1 - graph.train_mask.long())*100
				train_labels = torch.where(train_labels > 99, 100, train_labels)
			else:
				# CLUSTER dataset: train, val, and test dataset provided separately
				train_labels = graph.y

			with torch.cuda.amp.autocast():
				output = self.net(graph)
				loss = self.criterion(output['x'], train_labels)
				loss.backward()
				self.opt.step()