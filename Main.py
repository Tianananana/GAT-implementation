import torch
import argparse
from torch_geometric.datasets import Planetoid, GNNBenchmarkDataset
from Model.TransductiveGNN import TransductiveGNN
from Model.InductiveGNN import InductiveGNN
from Model.MLPBlock import MLP
from SupervisedTrainer import SupervisedTrainer
from Evaluator import Evaluator
from ModelIO import train_model
from utils import set_seed

parser = argparse.ArgumentParser(description='Training Parameters')
parser.add_argument('--data', dest='data', type=str, help='dataset to run model on')
parser.add_argument('--model', dest='model', type=str, help='type of GAT model: transductive or inductive')
parser.add_argument('--nheads', dest='n_heads', type=int, nargs='+', default=[8, 1], help='no. of heads for each layer')
parser.add_argument('--dropout', dest='dropout', type=float, default=0.5, help='dropout value')
parser.add_argument('--l2', dest='l2', type=float, default=5e-4, help='l2 regularization')
parser.add_argument('--lr', dest='lr', type=float, default=5e-3, help='learning rate')
parser.add_argument('--nepoch', dest='n_epoch', type=int, default=1000, help='no. of epochs to run training')
parser.add_argument('--batchsize', dest='batch_size', type=int, default=1, help='batch size')
args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    set_seed(17322625)

    if args.data == 'CLUSTER':
        # Load CLUSTER dataset
        dataset = GNNBenchmarkDataset("./Data/", name="CLUSTER", split='train')
        val_dataset = GNNBenchmarkDataset("./Data/", name="CLUSTER", split='val')
        test_dataset = GNNBenchmarkDataset("./Data/", name="CLUSTER", split='test')
    else:
        # Load Cora dataset
        dataset = Planetoid(root='./Data', name=args.data)

    in_feature = dataset.num_node_features
    out_feature = dataset.num_classes

    if args.model == 'transductive':
        # Transductive GNN architecture
        model = TransductiveGNN(in_feature=in_feature, out_feature=out_feature, n_heads=args.n_heads,
                                dropout=args.dropout)
    if args.model == 'inductive':
        # Inductive GNN architecture
        model = InductiveGNN(in_feature=in_feature, out_feature=out_feature, mid_feature=5)
    if args.model == 'benchmark':
        # 3-layer MLP model for benchmarking
        model = MLP(in_feature=in_feature, out_feature=out_feature, dropout=args.dropout)
    model.to(device)

    trainer = SupervisedTrainer(model, lr=args.lr, l2=args.l2)
    evaluator = Evaluator()

    # Train and evaluate model #
    if args.data == 'CLUSTER':
        # include val and test set
        train_model(0, args.n_epoch, args.batch_size, patience=100, trainer=trainer, evaluator=evaluator,
                    dataset=dataset, val_set=val_dataset, test_set=test_dataset)
    else:
        train_model(0, args.n_epoch, args.batch_size, patience=100, trainer=trainer, evaluator=evaluator,
                    dataset=dataset)


if __name__ == "__main__":
    main()
