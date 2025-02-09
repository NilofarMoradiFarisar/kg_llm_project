import torch
import torch.nn as nn
from transformers import AutoModel


class LLMKGEModel(nn.Module):
    def __init__(self, llm_model_name: str, embedding_dim: int = 768):
        super().__init__()
        self.llm = AutoModel.from_pretrained(llm_model_name)
        self.embedding_dim = embedding_dim

        # Projection layers
        self.entity_projection = nn.Linear(
            self.llm.config.hidden_size, embedding_dim)
        self.relation_projection = nn.Linear(
            self.llm.config.hidden_size, embedding_dim)

    def encode_text(self, input_ids, attention_mask):
        outputs = self.llm(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]  # Use [CLS] token embedding

    def forward(self, batch):
        # Encode head entity
        head_emb = self.encode_text(
            batch['head']['input_ids'],
            batch['head']['attention_mask']
        )
        head_emb = self.entity_projection(head_emb)

        # Encode relation
        rel_emb = self.encode_text(
            batch['relation']['input_ids'],
            batch['relation']['attention_mask']
        )
        rel_emb = self.relation_projection(rel_emb)

        # Encode tail entity
        tail_emb = self.encode_text(
            batch['tail']['input_ids'],
            batch['tail']['attention_mask']
        )
        tail_emb = self.entity_projection(tail_emb)

        # TransE-style scoring
        score = -torch.norm(head_emb + rel_emb - tail_emb, p=2, dim=1)
        return score
