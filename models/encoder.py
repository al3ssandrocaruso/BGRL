import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, MessagePassing
from torch_geometric.data import Data
from torch.nn import BatchNorm1d, LayerNorm

class GCN(torch.nn.Module):
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
        super(GCN, self).__init__()
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

class GAT(torch.nn.Module):
    """
    Graph Attention Network (GAT) with optional batch normalization and layer normalization.

    Parameters:
    - input_dim (int): Dimension of input features.
    - hidden_dim (int): Dimension of hidden layer.
    - output_dim (int): Dimension of output layer.
    - heads (int): Number of attention heads (default: 1).
    - use_batch_norm (bool): If True, apply batch normalization (default: False).
    - use_layer_norm (bool): If True, apply layer normalization (default: False).
    """

    def __init__(self, input_dim, hidden_dim, output_dim, heads=1, use_batch_norm=False, use_layer_norm=False):
        super(GAT, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads)
        self.conv2 = GATConv(hidden_dim * heads, output_dim, heads=heads)
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm

        if use_batch_norm:
            self.bn1 = BatchNorm1d(hidden_dim * heads)
            self.bn2 = BatchNorm1d(output_dim * heads)
        if use_layer_norm:
            self.ln1 = LayerNorm(hidden_dim * heads)
            self.ln2 = LayerNorm(output_dim * heads)

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

class MPNN(MessagePassing):
    """
    Message Passing Neural Network (MPNN) with optional batch normalization and layer normalization.

    Parameters:
    - input_dim (int): Dimension of input features.
    - hidden_dim (int): Dimension of hidden layer.
    - output_dim (int): Dimension of output layer.
    - use_batch_norm (bool): If True, apply batch normalization (default: False).
    - use_layer_norm (bool): If True, apply layer normalization (default: False).
    """

    def __init__(self, input_dim, hidden_dim, output_dim, use_batch_norm=False, use_layer_norm=False):
        super(MPNN, self).__init__(aggr='add')
        self.lin1 = torch.nn.Linear(input_dim, hidden_dim)
        self.lin2 = torch.nn.Linear(hidden_dim, output_dim)
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
        x = self.lin1(x)

        if self.use_batch_norm:
            x = self.bn1(x)
        if self.use_layer_norm:
            x = self.ln1(x)

        x = self.propagate(edge_index, x=x)
        x = F.relu(x)
        x = self.lin2(x)

        if self.use_batch_norm:
            x = self.bn2(x)
        if self.use_layer_norm:
            x = self.ln2(x)

        return x

    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        return aggr_out

class GNNEncoder(torch.nn.Module):
    """
    Generic GNN Encoder that supports GCN, GAT, and MPNN models.

    Parameters:
    - input_dim (int): Dimension of input features.
    - hidden_dim (int): Dimension of hidden layer.
    - output_dim (int): Dimension of output layer.
    - model_type (str): Type of model to use ('GCN', 'GAT', 'MPNN').
    - heads (int): Number of attention heads (for GAT, default: 1).
    - batch_norm (bool): If True, apply batch normalization (default: False).
    - layer_norm (bool): If True, apply layer normalization (default: False).
    """

    def __init__(self, input_dim, hidden_dim, output_dim, model_type='GCN', heads=1, batch_norm=False, layer_norm=False):
        super(GNNEncoder, self).__init__()
        if model_type == 'GCN':
            self.model = GCN(input_dim, hidden_dim, output_dim, batch_norm, layer_norm)
        elif model_type == 'GAT':
            self.model = GAT(input_dim, hidden_dim, output_dim, heads, batch_norm, layer_norm)
        elif model_type == 'MPNN':
            self.model = MPNN(input_dim, hidden_dim, output_dim, batch_norm, layer_norm)
        else:
            raise ValueError("Invalid model_type. Choose from 'GCN', 'GAT', or 'MPNN'.")

    def forward(self, data):
        return self.model(data)
