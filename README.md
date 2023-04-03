# Graph Attention Networks Implementation

Code prototype of Graph Attention Networks from Velickovic et al, 2018. GNN model is found under Model folder. Following the paper, a transductive and inductive version of GNN was implemented using the basic attention block. 



To improve stability of training, modifications were made to the model as follows.
1.	We replace the Softmax of attention coefficients with a L2 normalization layer. Then, we delay applying Softmax only at the last step of the GNN layer.
2.	For inductive GNN, we also used the Softmax instead of the Sigmoid layer. The Softmax function is more appropriate, given that the CLUSTER data is also performing a single class node classification. 

Training on the transductive GNN was done using the Cora dataset, following the original paper. To train the inductive GNN, we use the CLUSTER dataset, which consists of a large number of small graphs. we did not use the PPI dataset from the paper due to GPU memory constraints. A summary of the these two datasets are presented below.

Dataset | Cora  | CLUSTER
| --- | --- | ---
 **Task** | Transductive | Inductive
 **# Nodes** | 2708 | ~117.2k (10,000 graphs)
 **# Edges** | 5429 | ~4,303.9k
**# Features/Node** | 1433 | 7
**# Classes** | 7 | 6


Transductive GNN achieved a test accuracy of 68.7% on the Cora dataset, while inductive GNN had a test accuracy of 56.2% on CLUSTER. We also benchmarked both models with a simple MLP baseline model which obtained 55.5% and 21.2% for Cora and CLUSTER respectively. Although the results did not achieve the accuracies reported in the paper, we have confirmed that the implemented GAT still performs better compared to the baseline. Further optimisation could be done to improve the model's performance.
