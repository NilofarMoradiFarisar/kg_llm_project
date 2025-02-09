from src.dataset import KGDataset, load_entity_descriptions, load_relation_descriptions
from src.model import LLMKGEModel
from src.trainer import KGETrainer
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from config import CONFIG
import torch


def main():
    # Load descriptions
    entity_descriptions = load_entity_descriptions(CONFIG['data_path'])
    relation_descriptions = load_relation_descriptions(CONFIG['data_path'])

    results = {}
    for model_name in CONFIG['models_to_try']:
        print(f"\nTraining with {model_name}")

        # Initialize tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = LLMKGEModel(model_name)

        # Create datasets
        train_dataset = KGDataset(
            CONFIG['data_path'],
            'train',
            entity_descriptions,
            relation_descriptions,
            tokenizer,
            CONFIG['max_length']
        )

        valid_dataset = KGDataset(
            CONFIG['data_path'],
            'valid',
            entity_descriptions,
            relation_descriptions,
            tokenizer,
            CONFIG['max_length']
        )

        # Create dataloaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=CONFIG['batch_size'],
            shuffle=True
        )

        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=CONFIG['batch_size']
        )

        # Initialize trainer
        trainer = KGETrainer(
            model=model,
            train_dataloader=train_dataloader,
            valid_dataloader=valid_dataloader,
            lr=CONFIG['learning_rate'],
            device=CONFIG['device']
        )

        # Training loop
        for epoch in range(CONFIG['num_epochs']):
            print(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']}")

            # Train
            train_loss = trainer.train_epoch()
            print(f"Training Loss: {train_loss:.4f}")

            # Validate
            metrics = trainer.validate()
            print("Validation Metrics:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")

            # Save checkpoint
            trainer.save_checkpoint(epoch, metrics)

        results[model_name] = metrics

    # Print final results
    print("\nFinal Results:")
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    main()
