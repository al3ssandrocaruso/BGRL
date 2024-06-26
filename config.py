from torch_geometric.datasets import WikiCS, Coauthor, Amazon, CitationFull

# config.py
hidden_dim_encoder = 512
g_embedding_dim = 256
hidden_dim_predictor = 512
num_epochs = 300
early_stop_max = 50
pf_view_1 = 0.3
pf_view_2 = 0.2
pe_view_1 = 0.3
pe_view_2 = 0.4
lr = 1e-5
use_batch_norm = False
use_layer_norm = False
save_weights = False
optimizer = 'adam'


# Available datasets
datasets = ['WikiCS', 'Amazon_Computers', 'Amazon_Photos', 'Coauthor_CS', 'Cora','Coauthor_Physics']
default_dataset = 'Amazon_Photos'

# Mapping for datasets
dataset_mapping = {
        'Amazon_Computers': (Amazon, 'computers'),
        'Amazon_Photos': (Amazon, 'photo'),
        'Coauthor_CS': (Coauthor, 'CS'),
        'Coauthor_Physics': (Coauthor, 'Physics'),
        'Cora': (CitationFull, 'Cora'),
        'WikiCS': (WikiCS, None)
    }