import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, MessagePassing
from torch_geometric.data import Data
from torch.nn import BatchNorm1d, LayerNorm

class GCN_Encoder(torch.nn.Module):
    """
    Graph Convolutional Network (GCN) with optional batch normalization and layer normalization.

    Parameters:
    - input_dim (int): Dimension of input features.
    - hidden_dim (int): Dimension of hidden layer.
    - output_dim (int): Dimension of output layer.
    - use_batch_norm (bool): If True, apply batch normalization (default: False).
    - use_layer_norm (bool): If True, apply layer normalization (default: False).
    """

    def __init__(self, input_dim, hidden_dim, output_dim, use_batch_norm=False, use_layer_norm=False):
        super(GCN_Encoder, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm

        if use_batch_norm:
            self.bn1 = BatchNorm1d(hidden_dim)
            self.bn2 = BatchNorm1d(output_dim)
        if use_layer_norm:
            self.ln1 = LayerNorm(hidden_dim)
            self.ln2 = LayerNorm(output_dim)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        x = self.conv1(x, edge_index)

        if self.use_batch_norm:
            x = self.bn1(x)
        if self.use_layer_norm:
            x = self.ln1(x)

        x = F.relu(x)
        x = self.conv2(x, edge_index)

        if self.use_batch_norm:
            x = self.bn2(x)
        if self.use_layer_norm:
            x = self.ln2(x)

        return x