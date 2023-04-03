import torch
import torch.nn as nn
from Model.AttentionBlock import AttentionBlock


class InductiveGNN(nn.Module):
    """GNN for inductive learning"""

    def __init__(self, in_feature, out_feature, mid_feature, n_heads=(4, 4, 6), dropout=0):
        super().__init__()
        self.n_heads = n_heads
        self.mid_feature = mid_feature
        self.out_feature = out_feature
        for i in range(n_heads[0]):
            setattr(self, f"layer1Head{i}", AttentionBlock(in_feature, mid_feature, dropout=dropout))
        for i in range(n_heads[1]):
            setattr(self, f"layer2Head{i}", AttentionBlock(mid_feature * n_heads[0], mid_feature, dropout=dropout))
        for i in range(n_heads[2]):  # paper uses out_feature=121
            setattr(self, f"layer3Head{i}", AttentionBlock(mid_feature * n_heads[1], out_feature, dropout=dropout))

        self.ELU = nn.ELU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        n_nodes = x['x'].size()[0]
        layer_list = self.__dict__['_modules']

        # Layer 1
        head_output1 = torch.empty((n_nodes, 0), device='cuda')
        for i in range(self.n_heads[0]):
            # Concatenate features for each head at -1 dim
            x_tmp = x.clone()
            head_output_tmp = layer_list[f"layer1Head{i}"](x_tmp)
            head_output_tmp['x'] = self.ELU(head_output_tmp['x'])
            head_output1 = torch.cat((head_output1, head_output_tmp['x']), dim=-1)
        x['x'] = head_output1

        # Layer 2
        head_output2 = torch.empty((n_nodes, 0), device='cuda')
        # Concatenate features for each head at -1 dim
        for i in range(self.n_heads[1]):
            x_tmp = x.clone()
            head_output_tmp = layer_list[f"layer2Head{i}"](x_tmp)
            head_output_tmp['x'] = self.ELU(head_output_tmp['x'])
            head_output2 = torch.cat((head_output2, head_output_tmp['x']), dim=-1)
        x['x'] = head_output2
        # Skip connections on layer 2 attention
        x['x'] = head_output2 + head_output1

        # Layer 3
        head_output3 = torch.empty((n_nodes, self.out_feature, 0), device='cuda')
        # Concatenate features for each head on new dim
        for i in range(self.n_heads[2]):
            x_tmp = x.clone()
            head_output_tmp = layer_list[f"layer3Head{i}"](x_tmp)
            head_output_tmp['x'] = torch.unsqueeze(self.ELU(head_output_tmp['x']), dim=-1)
            head_output3 = torch.cat((head_output3, head_output_tmp['x']), dim=-1)
        x['x'] = head_output3

        # Average over each head
        x['x'] = torch.mean(x['x'], dim=-1)
        x['x'] = self.softmax(x['x'])

        return x
