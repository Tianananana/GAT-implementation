import sys
import time
from torch_geometric.loader import DataLoader

def train_model(start, end, batch_size, patience, trainer, evaluator, dataset, val_set=None, test_set=None):
	"""
	start: start epoch
	end: end epoch
	batch_size: batch size per epoch
	patience: no. of epochs for early stopping
	"""
	# Early stopping on validation loss and accuracy
	acc_previous, loss_previous = -sys.maxsize, sys.maxsize
	p_count = 0

	print('START TRAINING')
	for i in range(start, end):
		start_t = time.time()
		data_loader = DataLoader(dataset, batch_size=batch_size)

		## TRAIN ##
		trainer.one_epoch(data_loader)

		## EVALUATE ##
		if hasattr(dataset[0], 'train_mask'):
			# Cora dataset: 'train_mask', 'test_mask', 'val_mask' provided.
			# Loss computed directly from dataset.
			_, train_loss, val_loss, test_loss = evaluator.eval_loss(trainer.net, data_loader)
			evaluator.eval_acc(trainer.net, data_loader)
			_, train_acc, val_acc, test_acc = evaluator.eval_acc(trainer.net, data_loader)

		else:
			# CLUSTER dataset: train, val, and test dataset provided separately
			# Loss computed separately for train, val, and test datasets.
			val_loader = DataLoader(val_set, batch_size=batch_size)
			test_loader = DataLoader(test_set, batch_size=batch_size)
			train_loss, _, _, _ = evaluator.eval_loss(trainer.net, data_loader)
			val_loss, _, _, _ = evaluator.eval_loss(trainer.net, val_loader)
			test_loss, _, _, _ = evaluator.eval_loss(trainer.net, test_loader)
			train_acc, _, _, _ = evaluator.eval_acc(trainer.net, data_loader)
			val_acc, _, _, _ = evaluator.eval_acc(trainer.net, val_loader)
			test_acc, _, _, _ = evaluator.eval_acc(trainer.net, test_loader)

		interval = (time.time() - start_t) / 60

		message = f"EPOCH {i}, TIME {interval:.2f}," \
			          f"TRAIN loss {train_loss:.6f}, acc {train_acc:.6f} " \
			          f"VALID loss {val_loss:.6f}, acc {val_acc:.6f} " \
			          f"TEST loss {test_loss:.6f}, acc {test_acc:.6f}"

		print(message)

		if val_acc > acc_previous or val_loss < loss_previous:
			p_count = 0
			print('weight updated') # We did not formally save the weights in this assignment.
			acc_previous = val_acc
			loss_previous = val_loss
		else:
			p_count += 1

		# Early stopping
		if p_count > patience:
			print(f'val_loss did not decrease for {patience} epochs consequently.')
			break

	print('END TRAINING')