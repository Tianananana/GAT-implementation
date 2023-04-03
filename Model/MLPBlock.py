import torch.nn as nn

class MLP(nn.Module):
    """ Simple 3 layer MLP for benchmarking"""
    def __init__(self, in_feature, out_feature, dropout=0):
        super().__init__()
        self.layer1 = nn.Linear(in_feature, 100)
        self.layer2 = nn.Linear(100, 100)
        self.layer3 = nn.Linear(100, out_feature)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x['x'] = self.dropout(x['x'])
        x['x'] = self.layer1(x['x'])
        x['x'] = self.relu(x['x'])
        x['x'] = self.dropout(x['x'])
        x['x'] = self.layer2(x['x'])
        x['x'] = self.relu(x['x'])
        x['x'] = self.dropout(x['x'])
        x['x'] = self.layer3(x['x'])
        x['x'] = self.sigmoid(x['x'])
        return x