# args.py
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
batch_norm = False
layer_norm = False
optimizer = 'adam'
encoder_type = 'GCN'

# Available datasets
datasets = ['WikiCS', 'Amazon_Computers', 'Amazon_Photos', 'Coauthor_CS', 'Cora']
default_dataset = 'Amazon_Photos'