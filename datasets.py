import torch
from torch_geometric.datasets import WikiCS, Coauthor, Amazon, CitationFull
from sklearn.preprocessing import StandardScaler

def load_dataset(name):
    """
    Load and preprocess a dataset.

    Parameters:
    - name (str): Name of the dataset to load. Must be one of the following:
        'Amazon_Computers', 'Amazon_Photos', 'Coauthor_CS', 'Coauthor_Physics', 'Cora', 'CiteSeer',
        'PubMed', 'DBLP', 'WikiCS'.

    Returns:
    - data (torch_geometric.data.Data): The processed dataset with scaled features.

    Raises:
    - ValueError: If the dataset name is not supported.
    """

    dataset_mapping = {
        'Amazon_Computers': (Amazon, 'computers'),
        'Amazon_Photos': (Amazon, 'photo'),
        'Coauthor_CS': (Coauthor, 'CS'),
        'Coauthor_Physics': (Coauthor, 'Physics'),
        'Cora': (CitationFull, 'Cora'),
        'WikiCS': (WikiCS, None)
    }

    if name not in dataset_mapping:
        raise ValueError(f"Dataset {name} is not supported. Please choose from {list(dataset_mapping.keys())}.")

    dataset_class, dataset_name = dataset_mapping[name]

    if dataset_name is None:
        data = dataset_class("../data")[0]
    else:
        data = dataset_class("../data", name=dataset_name)[0]

    features = data.x.numpy()

    # Apply StandardScaler to the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    data.x = torch.tensor(scaled_features, dtype=torch.float)

    return data
