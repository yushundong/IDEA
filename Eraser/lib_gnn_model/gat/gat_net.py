import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GATNet(torch.nn.Module):
    def __init__(self, num_feats, num_classes, dropout=0.1):
        super(GATNet, self).__init__()
        self.dropout = dropout

        self.conv1 = GATConv(
            num_feats, 16, heads=8, dropout=self.dropout, add_self_loops=False
        )
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(
            16 * 8,
            32,
            heads=1,
            concat=False,
            dropout=self.dropout,
            add_self_loops=False,
        )
        # self.conv2 = GATConv(8 * 8, num_classes, heads=8, concat=False, dropout=self.dropout, add_self_loops=False)
        self.batch_norm1 = torch.nn.BatchNorm1d(16 * 8)
        self.batch_norm2 = torch.nn.BatchNorm1d(32)
        self.fc1 = torch.nn.Linear(32, 32)
        self.fc2 = torch.nn.Linear(32, num_classes)
        self.reset_parameters()

    def forward(self, data):
        x = F.dropout(data.x, p=self.dropout, training=self.training)
        x = F.relu(self.conv1(x, data.edge_index))
        x = self.batch_norm1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, data.edge_index))
        x = self.batch_norm2(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
