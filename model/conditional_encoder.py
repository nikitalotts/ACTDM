"""
Классификатор для classifier guidance в текстовой диффузии.

Согласно PDF:
- Модель выдаёт логит f(x_t, t, y) ∈ R
- p(y | x_t) = σ(f(x_t, t, y))
- При инференсе: ∇_{x_t} log p(y|x_t) = (1 - σ(f)) · ∇_{x_t} f
"""

import math
import torch
import torch.nn as nn
from transformers import BertModel


class ConditionalEncoder(nn.Module):
    """
    Классификатор для classifier guidance.

    Входы:
    - y (src_embeds): префикс/условие [batch, seq_len, hidden]
    - x_t (noisy_trg_embeds): зашумлённое продолжение [batch, seq_len, hidden]
    - t: временной шаг диффузии [batch]
    - src_mask: маска для src (1=valid, 0=padding)

    Выход:
    - logits f(x_t, t, y): [batch]

    Примечание: trg_mask НЕ используется для консистентности между обучением и инференсом.
    При инференсе диффузия генерирует все позиции, padding нет.
    """

    def __init__(self, encoder_link, tokenizer, hidden_dim=768):
        super().__init__()
        self.encoder_link = encoder_link
        self.hidden_dim = hidden_dim

        if "bert" in encoder_link.lower():
            self.bert = BertModel.from_pretrained(encoder_link)
            self.tokenizer = tokenizer

            # Получаем эмбеддинги специальных токенов
            cls_embedding = self.bert.embeddings.word_embeddings.weight[self.tokenizer.cls_token_id].detach().clone()
            sep_embedding = self.bert.embeddings.word_embeddings.weight[self.tokenizer.sep_token_id].detach().clone()

            self.register_buffer('cls_embedding', cls_embedding)
            self.register_buffer('sep_embedding', sep_embedding)
        else:
            raise Exception("Unknown encoder name")

        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        # Score head: предсказывает скалярный логит f(x_t, t, y) ∈ R
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def timestep_embedding(self, timesteps, dim, max_period=10000):
        """Sinusoidal time embedding"""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, src_embeds, noisy_trg_embeds, t, src_mask=None):
        device = src_embeds.device
        batch_size = src_embeds.shape[0]
        seq_len_src = src_embeds.shape[1]
        seq_len_trg = noisy_trg_embeds.shape[1]

        # 1. Time embedding
        t_emb = self.timestep_embedding(t, self.hidden_dim).to(device)
        t_embed = self.time_mlp(t_emb)  # [batch, hidden]

        # 2. Специальные токены
        cls_token = self.cls_embedding.unsqueeze(0).expand(batch_size, 1, -1)
        sep_token = self.sep_embedding.unsqueeze(0).expand(batch_size, 1, -1)

        # 3. Стандартный BERT формат:
        # [CLS] time src... [SEP] noisy_trg... [SEP]
        inputs_embeds = torch.cat([
            cls_token,  # [batch, 1, hidden]
            t_embed.unsqueeze(1),  # [batch, 1, hidden]
            src_embeds,  # [batch, seq_len_src, hidden]
            sep_token,  # [batch, 1, hidden]
            noisy_trg_embeds,  # [batch, seq_len_trg, hidden]
            sep_token,  # [batch, 1, hidden]
        ], dim=1)

        # 4. Attention mask
        ones = torch.ones(batch_size, 1, device=device, dtype=torch.long)
        trg_ones = torch.ones(batch_size, seq_len_trg, device=device, dtype=torch.long)

        if src_mask is None:
            src_mask = torch.ones(batch_size, seq_len_src, device=device, dtype=torch.long)

        attention_mask = torch.cat([
            ones,  # CLS
            ones,  # time
            src_mask,  # src
            ones,  # SEP
            trg_ones,  # trg
            ones,  # SEP
        ], dim=1)

        # 5. BERT
        outputs = self.bert(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True
        )

        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.score_head(cls_output).squeeze(-1)

        return logits