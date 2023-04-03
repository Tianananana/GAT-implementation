import torch
import numpy as np

class Evaluator:
	"""Evaluator Class: compute metrics"""
	def __init__(self, criterion=torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=100)):
		self.criterion = criterion

	def eval_loss(self, net, data_loader):
		net.eval()
		with torch.no_grad():
			loss = 0
			train_loss = 0
			val_loss = 0
			test_loss = 0
			for graph in data_loader:
				graph.to('cuda')
				output = net(graph)

				if hasattr(graph, 'train_mask'):
					# Cora dataset: 'train_mask', 'test_mask', 'val_mask' provided
					train_labels = graph.y + (1 - graph.train_mask.long())*100
					train_labels = torch.where(train_labels > 99, 100, train_labels)

					val_labels = graph.y + (1 - graph.val_mask.long())*100
					val_labels = torch.where(val_labels > 99, 100, val_labels)

					test_labels = graph.y + (1 - graph.test_mask.long())*100
					test_labels = torch.where(test_labels > 99, 100, test_labels)

					train_loss += self.criterion(output['x'], train_labels)
					val_loss += self.criterion(output['x'], val_labels)
					test_loss += self.criterion(output['x'], test_labels)

				else:
					# CLUSTER dataset: train, val, and test dataset provided separately
					loss += self.criterion(output['x'], graph.y)

		return loss, train_loss, val_loss, test_loss

	def eval_acc(self, net, data_loader):
		""" Computes accuracy """
		net.eval()
		acc_train = []
		acc_val = []
		acc_test = []
		accuracy = []
		with torch.no_grad():
			for graph in data_loader:
				graph.to('cuda')
				output = net(graph)
				preds = torch.argmax(output['x'], dim=1)

				if hasattr(graph, 'train_mask'):
					# Cora dataset: 'train_mask', 'test_mask', 'val_mask' provided
					preds_train = preds[torch.argwhere(graph.train_mask.long() == 1)]
					labels_train = graph.y[torch.argwhere(graph.train_mask.long() == 1)]

					preds_val = preds[torch.argwhere(graph.val_mask.long() == 1)]
					labels_val = graph.y[torch.argwhere(graph.val_mask.long() == 1)]

					preds_test = preds[torch.argwhere(graph.test_mask.long() == 1)]
					labels_test = graph.y[torch.argwhere(graph.test_mask.long() == 1)]

					accuracy_train = torch.sum(preds_train == labels_train) / labels_train.size()[0]
					acc_train.append(accuracy_train.item())

					accuracy_val = torch.sum(preds_val == labels_val) / labels_val.size()[0]
					acc_val.append(accuracy_val.item())

					accuracy_test = torch.sum(preds_test == labels_test) / labels_test.size()[0]
					acc_test.append(accuracy_test.item())

				else:
					# CLUSTER dataset: train, val, and test dataset provided separately
					acc = torch.sum(preds == graph.y) / graph.y.size()[0]
					accuracy.append(acc.item())

		return np.mean(accuracy), np.mean(acc_train), np.mean(acc_val), np.mean(acc_test)
