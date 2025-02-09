import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from typing import Dict, Optional
import os


class KGETrainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        valid_dataloader: Optional[DataLoader] = None,
        lr: float = 1e-5,
        device: str = 'cuda',
        model_save_path: str = './checkpoints'
    ):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.device = device
        self.model_save_path = model_save_path

        # Create checkpoint directory if it doesn't exist
        os.makedirs(model_save_path, exist_ok=True)

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0

        # Use tqdm for progress bar
        pbar = tqdm(self.train_dataloader, desc="Training")

        for batch in pbar:
            # Move batch to device
            batch = {k: {k2: v2.to(self.device) for k2, v2 in v.items()
                         if isinstance(v2, torch.Tensor)}
                     for k, v in batch.items()}

            # Forward pass
            scores = self.model(batch)

            # Compute loss
            positive_score = scores.diagonal()
            negative_scores = scores.view(-1)

            # Margin ranking loss
            loss = torch.mean(torch.max(
                torch.zeros_like(positive_score),
                1.0 - positive_score.unsqueeze(1) + negative_scores
            ))

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        return total_loss / len(self.train_dataloader)

    def validate(self) -> Dict[str, float]:
        self.model.eval()
        metrics = {
            'hits@1': 0,
            'hits@3': 0,
            'hits@10': 0,
            'mrr': 0
        }

        with torch.no_grad():
            for batch in tqdm(self.valid_dataloader, desc="Validating"):
                batch = {k: {k2: v2.to(self.device) for k2, v2 in v.items()
                             if isinstance(v2, torch.Tensor)}
                         for k, v in batch.items()}

                scores = self.model(batch)

                # Calculate metrics
                ranks = self._get_ranks(scores)
                metrics = self._update_metrics(metrics, ranks)

        # Average metrics
        num_samples = len(self.valid_dataloader.dataset)
        return {k: v/num_samples for k, v in metrics.items()}

    def _get_ranks(self, scores: torch.Tensor) -> torch.Tensor:
        """Calculate the rank of each positive sample"""
        positive_scores = scores.diagonal()
        sorted_indices = torch.argsort(scores, dim=1, descending=True)
        ranks = torch.where(sorted_indices == torch.arange(
            len(scores))[:, None].to(self.device))[1] + 1
        return ranks

    def _update_metrics(self, metrics: Dict[str, float], ranks: torch.Tensor) -> Dict[str, float]:
        """Update metrics based on ranks"""
        metrics['hits@1'] += (ranks <= 1).float().sum().item()
        metrics['hits@3'] += (ranks <= 3).float().sum().item()
        metrics['hits@10'] += (ranks <= 10).float().sum().item()
        metrics['mrr'] += (1.0 / ranks).sum().item()
        return metrics

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }
        path = os.path.join(self.model_save_path,
                            f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, path)
        self.logger.info(f"Saved checkpoint to {path}")
