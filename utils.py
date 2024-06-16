import torch
import psutil
import os

def augment_graph(data, pf, pe):
    """
    Apply graph-wise augmentation by masking features and edges.

    Parameters:
    - data (torch_geometric.data.Data): Input graph data containing node features and edge indices.
    - pf (float): Probability of masking each feature.
    - pe (float): Probability of masking each edge.

    Returns:
    - augmented_data (torch_geometric.data.Data): Augmented graph data with masked features and edges.
    """

    F = data.x.size(1)  # Number of features in the original graph
    E = data.edge_index.size(1)  # Number of edges in the original graph

    # Generate random binary mask for features
    feature_mask = torch.bernoulli(torch.full((F,), 1 - pf)).bool()
    augmented_data = data.clone()
    augmented_data.x[:, ~feature_mask] = 0

    # Generate random binary mask for edges
    edge_mask = torch.bernoulli(torch.full((E,), 1 - pe)).bool()
    augmented_data.edge_index = data.edge_index[:, edge_mask]

    return augmented_data

def print_memory_usage():
    """
    Print the current memory usage of the process in gigabytes (GB).
    Usage:
    Call this function to print the current memory usage in gigabytes (GB).
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"Current memory usage: {memory_info.rss / (1024 ** 2)} MB")
