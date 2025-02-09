import torch
from torch.utils.data import Dataset
import pandas as pd
import os
from typing import Dict, List, Tuple

import os

file_path = os.path.join('data', 'FB15K-237', 'train.txt')


class KGDataset(Dataset):
    def __init__(self, data_path: str, split: str, entity_descriptions: Dict,
                 relation_descriptions: Dict, tokenizer, max_length: int = 128):
        """
        Args:
            data_path: Path to FB15K-237 dataset
            split: 'train', 'valid', or 'test'
            entity_descriptions: Dictionary of entity descriptions
            relation_descriptions: Dictionary of relation descriptions
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length for tokenization
        """
        self.data_path = data_path
        self.split = split
        self.entity_descriptions = entity_descriptions
        self.relation_descriptions = relation_descriptions
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load triples
        self.triples = self._load_triples()

    def _load_triples(self) -> List[Tuple]:
        file_path = os.path.join(self.data_path, f'{self.split}.txt')
        df = pd.read_csv(file_path, sep='\t', header=None,
                         names=['head', 'relation', 'tail'])
        return list(df.itertuples(index=False, name=None))

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]

        # Encode descriptions
        head_encoded = self.tokenizer(
            self.entity_descriptions.get(head, "Unknown entity"),
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        relation_encoded = self.tokenizer(
            self.relation_descriptions.get(relation, "Unknown relation"),
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        tail_encoded = self.tokenizer(
            self.entity_descriptions.get(tail, "Unknown entity"),
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'head': {k: v.squeeze(0) for k, v in head_encoded.items()},
            'relation': {k: v.squeeze(0) for k, v in relation_encoded.items()},
            'tail': {k: v.squeeze(0) for k, v in tail_encoded.items()},
            'triple': (head, relation, tail)
        }


def load_entity_descriptions(data_path: str) -> Dict[str, str]:
    """Load entity descriptions from file"""
    descriptions = {}
    desc_file = os.path.join(data_path, 'entity2text.txt')

    if os.path.exists(desc_file):
        with open(desc_file, 'r', encoding='utf-8') as f:
            for line in f:
                entity, desc = line.strip().split('\t')
                descriptions[entity] = desc

    return descriptions


def load_relation_descriptions(data_path: str) -> Dict[str, str]:
    """Load relation descriptions from file"""
    descriptions = {}
    desc_file = os.path.join(data_path, 'relation2text.txt')

    if os.path.exists(desc_file):
        with open(desc_file, 'r', encoding='utf-8') as f:
            for line in f:
                relation, desc = line.strip().split('\t')
                descriptions[relation] = desc

    return descriptions
