import torch
import numpy as np
import logging
import json
import os
from typing import List, Dict, Tuple, Any
from datetime import datetime


def setup_logging(log_dir: str = "logs") -> None:
    """
    Set up logging configuration for the project.

    Args:
        log_dir: Directory to store log files
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"kg_llm_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def load_kg_data(file_path: str) -> List[Tuple]:
    """
    Load knowledge graph triples from a file.

    Args:
        file_path: Path to the file containing KG triples

    Returns:
        List of (head, relation, tail) tuples
    """
    triples = []
    with open(file_path, 'r') as f:
        for line in f:
            h, r, t = line.strip().split('\t')
            triples.append((h, r, t))
    return triples


def create_entity_and_relation_mappings(triples: List[Tuple]) -> Tuple[Dict, Dict]:
    """
    Create entity and relation mappings from triples.

    Args:
        triples: List of (head, relation, tail) tuples

    Returns:
        entity_to_id: Dictionary mapping entities to IDs
        relation_to_id: Dictionary mapping relations to IDs
    """
    entities = set()
    relations = set()

    for h, r, t in triples:
        entities.add(h)
        entities.add(t)
        relations.add(r)

    entity_to_id = {ent: idx for idx, ent in enumerate(sorted(entities))}
    relation_to_id = {rel: idx for idx, rel in enumerate(sorted(relations))}

    return entity_to_id, relation_to_id


def save_mappings(mappings: Dict, file_path: str) -> None:
    """
    Save mappings to a JSON file.

    Args:
        mappings: Dictionary of mappings
        file_path: Path to save the mappings
    """
    with open(file_path, 'w') as f:
        json.dump(mappings, f, indent=2)


def load_mappings(file_path: str) -> Dict:
    """
    Load mappings from a JSON file.

    Args:
        file_path: Path to the mappings file

    Returns:
        Dictionary of mappings
    """
    with open(file_path, 'r') as f:
        return json.load(f)


def convert_triple_to_ids(
    triple: Tuple[str, str, str],
    entity_to_id: Dict[str, int],
    relation_to_id: Dict[str, int]
) -> Tuple[int, int, int]:
    """
    Convert a triple of strings to a triple of IDs.

    Args:
        triple: (head, relation, tail) tuple of strings
        entity_to_id: Entity to ID mapping
        relation_to_id: Relation to ID mapping

    Returns:
        Triple of IDs
    """
    h, r, t = triple
    return (entity_to_id[h], relation_to_id[r], entity_to_id[t])


def create_negative_samples(
    triple: Tuple[int, int, int],
    num_entities: int,
    num_samples: int = 10
) -> torch.Tensor:
    """
    Create negative samples for a triple by corrupting head or tail.

    Args:
        triple: (head, relation, tail) tuple of IDs
        num_entities: Total number of entities
        num_samples: Number of negative samples to generate

    Returns:
        Tensor of negative samples
    """
    h, r, t = triple
    negative_samples = []

    for _ in range(num_samples):
        if np.random.random() < 0.5:
            # Corrupt head
            h_corrupt = np.random.randint(num_entities)
            while h_corrupt == h:
                h_corrupt = np.random.randint(num_entities)
            negative_samples.append((h_corrupt, r, t))
        else:
            # Corrupt tail
            t_corrupt = np.random.randint(num_entities)
            while t_corrupt == t:
                t_corrupt = np.random.randint(num_entities)
            negative_samples.append((h, r, t_corrupt))

    return torch.tensor(negative_samples)


def calculate_metrics(
    scores: torch.Tensor,
    positive_sample: torch.Tensor,
    negative_samples: torch.Tensor
) -> Dict[str, float]:
    """
    Calculate ranking metrics (MRR, Hits@1, Hits@5, Hits@10).

    Args:
        scores: Predicted scores for all samples
        positive_sample: The positive triple
        negative_samples: The negative triples

    Returns:
        Dictionary of metrics
    """
    all_scores = torch.cat([scores[None, :], negative_samples])
    positive_score = scores[0]

    # Calculate rank
    rank = (all_scores >= positive_score).sum().item()

    metrics = {
        'MRR': 1.0 / rank,
        'Hits@1': float(rank <= 1),
        'Hits@5': float(rank <= 5),
        'Hits@10': float(rank <= 10)
    }

    return metrics


def save_model_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    checkpoint_dir: str
) -> None:
    """
    Save model checkpoint.

    Args:
        model: The PyTorch model
        optimizer: The optimizer
        epoch: Current epoch number
        loss: Current loss value
        checkpoint_dir: Directory to save checkpoints
    """
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_path = os.path.join(
        checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, checkpoint_path)


def load_model_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None
) -> Tuple[torch.nn.Module, torch.optim.Optimizer, int, float]:
    """
    Load model checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file
        model: The PyTorch model
        optimizer: The optimizer (optional)

    Returns:
        Tuple of (model, optimizer, epoch, loss)
    """
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer, checkpoint['epoch'], checkpoint['loss']
