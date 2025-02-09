import torch

CONFIG = {
    'data_path': './data/FB15K-237/',
    'model_save_path': './checkpoints/',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'batch_size': 32,
    'max_length': 128,
    'learning_rate': 1e-5,
    'num_epochs': 3,
    'models_to_try': [
        'bert-base-uncased',
        'roberta-base',
        'microsoft/deberta-base',
        'albert-base-v2'
    ]
}
