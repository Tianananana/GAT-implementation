import torch
import torch.nn as nn
import torch_geometric.utils as utils


class AttentionBlock(nn.Module):
    """ One attention head """

    def __init__(self, in_feature, out_feature, dropout=0):
        super().__init__()

        # Set initial weights via glorot initialization #
        self.W = nn.Parameter(nn.init.xavier_normal_(torch.empty((in_feature, out_feature))))  # Shape: (F, F')
        self.a = nn.Parameter(nn.init.xavier_normal_(torch.empty((out_feature * 2, 1))))  # Shape: (2F', 1)
        self.leakyReLU = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        features, adj_matrix = x['x'], utils.to_dense_adj(x['edge_index']).squeeze()
        # Handle the case where edge index matrix do not contain all nodes in graph.
        no_edges_nodes = features.size()[0] - adj_matrix.size()[0]
        Pad = nn.ConstantPad2d((0, no_edges_nodes, 0, no_edges_nodes),
                               0)  # enforce equal size adj matrix and node-feature matrix
        adj_matrix = Pad(adj_matrix)
        HW = torch.matmul(features, self.W)

        ### Self attention ###
        HW_broadcast = torch.unsqueeze(HW, 0)
        HW_broadcast = HW.expand(HW.size()[0], HW.size()[0], HW.size()[1])
        # Concatenate pairwise node's features #
        pairwise_features = torch.cat((torch.transpose(HW_broadcast, 0, 1), HW_broadcast), -1)
        # Mask with adjacency matrix: drop node pairs without edges #
        pairwise_features_masked = pairwise_features * adj_matrix[..., None]
        # Apply attention mechanism #
        edge_ij = torch.matmul(pairwise_features_masked, self.a).squeeze()
        edge_ij = self.leakyReLU(edge_ij)

        # We replace Softmax layer of attention coefficients with a L2 normalization layer. We delay applying Softmax
        # only at the last step of the GNN model.
        alpha_ij = torch.nn.functional.normalize(edge_ij, dim=0)
        alpha_ij = self.dropout(alpha_ij)

        output = torch.matmul(alpha_ij, HW)
        x['x'] = output

        return x
