import torch
import torch.nn.functional as F
from lib_gnn_model.sgc.sgc_conv import SGConvBatch

import pdb


class SGCNet(torch.nn.Module):
    def __init__(self, num_feats, num_classes):
        super(SGCNet, self).__init__()
        self.num_feats = num_feats
        self.conv1 = SGConvBatch(
            num_feats, 16, cached=False, add_self_loops=True, bias=False
        )
        self.conv2 = SGConvBatch(
            16, num_classes, cached=False, add_self_loops=True, bias=False
        )
        # pdb.set_trace()

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_weight)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)

        return F.log_softmax(x, dim=-1)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
