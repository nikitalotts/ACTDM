import math
import torch
import torch.nn as nn

from transformers import BertModel
import math
import torch
import torch.nn as nn
from transformers import BertForNextSentencePrediction

# 30 epochs 3503527
class ConditionalEncoder(nn.Module):
    def __init__(self, encoder_link, tokenizer, hidden_dim=768):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.inner_dim = hidden_dim * 2  # Увеличенная внутренняя размерность

        # Проекция входов в большее пространство
        self.src_proj = nn.Linear(hidden_dim, self.inner_dim)
        self.trg_proj = nn.Linear(hidden_dim, self.inner_dim)

        # Cross-attention: 2 слоя вместо 1
        self.cross_attn_1 = nn.MultiheadAttention(
            embed_dim=self.inner_dim,
            num_heads=16,  # Больше голов
            dropout=0.1,
            batch_first=True
        )
        self.cross_attn_2 = nn.MultiheadAttention(
            embed_dim=self.inner_dim,
            num_heads=16,
            dropout=0.1,
            batch_first=True
        )

        # LayerNorm после каждого attention
        self.norm1 = nn.LayerNorm(self.inner_dim)
        self.norm2 = nn.LayerNorm(self.inner_dim)
        self.norm3 = nn.LayerNorm(self.inner_dim)

        # Feed-forward после attention
        self.ffn = nn.Sequential(
            nn.Linear(self.inner_dim, self.inner_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.inner_dim * 2, self.inner_dim),
            nn.Dropout(0.1)
        )

        # Time embedding - увеличенный
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, self.inner_dim),
            nn.SiLU(),
            nn.Linear(self.inner_dim, self.inner_dim),
            nn.SiLU(),
            nn.Linear(self.inner_dim, self.inner_dim)
        )

        # Увеличенный classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.inner_dim * 2, self.inner_dim * 2),
            nn.LayerNorm(self.inner_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.inner_dim * 2, self.inner_dim),
            nn.LayerNorm(self.inner_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.inner_dim, 1)
        )

    def timestep_embedding(self, timesteps, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half
        )
        args = timesteps[:, None].float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, src_embeds, noisy_trg_embeds, t):
        # src_embeds: [batch, hidden]
        # noisy_trg_embeds: [batch, hidden]

        # Проекция в большее пространство
        src_proj = self.src_proj(src_embeds)  # [batch, inner_dim]
        trg_proj = self.trg_proj(noisy_trg_embeds)  # [batch, inner_dim]

        # Time conditioning
        t_emb = self.timestep_embedding(t, self.hidden_dim)
        t_emb = self.time_mlp(t_emb)  # [batch, inner_dim]

        # Condition trg on time
        trg_cond = trg_proj + t_emb  # [batch, inner_dim]

        # Reshape for attention: [batch, seq=1, inner_dim]
        src_seq = src_proj.unsqueeze(1)
        trg_seq = trg_cond.unsqueeze(1)

        # Cross-attention layer 1
        attn_out_1, _ = self.cross_attn_1(
            query=trg_seq,
            key=src_seq,
            value=src_seq
        )
        trg_seq = self.norm1(trg_seq + attn_out_1)  # Residual + LayerNorm

        # Cross-attention layer 2
        attn_out_2, _ = self.cross_attn_2(
            query=trg_seq,
            key=src_seq,
            value=src_seq
        )
        trg_seq = self.norm2(trg_seq + attn_out_2)  # Residual + LayerNorm

        # Feed-forward
        ffn_out = self.ffn(trg_seq)
        trg_seq = self.norm3(trg_seq + ffn_out)  # Residual + LayerNorm

        attn_out = trg_seq.squeeze(1)  # [batch, inner_dim]

        # Concatenate and classify
        combined = torch.cat([attn_out, trg_cond], dim=-1)  # [batch, inner_dim * 2]
        logits = self.classifier(combined).squeeze(-1)

        return logits


# 3503512
# class ConditionalEncoder(nn.Module):
#     def __init__(self, encoder_link, tokenizer, hidden_dim=768):
#         super().__init__()
#         self.hidden_dim = hidden_dim
#
#         # Cross-attention: trg attends to src
#         self.cross_attn = nn.MultiheadAttention(
#             embed_dim=hidden_dim,
#             num_heads=8,
#             dropout=0.1,
#             batch_first=True
#         )
#
#         self.time_mlp = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.SiLU(),
#             nn.Linear(hidden_dim, hidden_dim)
#         )
#
#         # После cross-attention
#         self.classifier = nn.Sequential(
#             nn.Linear(hidden_dim * 2, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(hidden_dim, 1)
#         )
#
#     def timestep_embedding(self, timesteps, dim, max_period=10000):
#         half = dim // 2
#         freqs = torch.exp(
#             -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half
#         )
#         args = timesteps[:, None].float() * freqs[None]
#         return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
#
#     def forward(self, src_embeds, noisy_trg_embeds, t):
#         # src_embeds: [batch, hidden]
#         # noisy_trg_embeds: [batch, hidden]
#
#         batch_size = src_embeds.shape[0]
#
#         # Time conditioning
#         t_emb = self.timestep_embedding(t, self.hidden_dim)
#         t_emb = self.time_mlp(t_emb)  # [batch, hidden]
#
#         # Condition trg on time
#         noisy_trg_cond = noisy_trg_embeds + t_emb  # [batch, hidden]
#
#         # Reshape for attention: [batch, seq=1, hidden]
#         src_seq = src_embeds.unsqueeze(1)
#         trg_seq = noisy_trg_cond.unsqueeze(1)
#
#         # Cross-attention: trg queries, src is key/value
#         attn_out, _ = self.cross_attn(
#             query=trg_seq,
#             key=src_seq,
#             value=src_seq
#         )  # [batch, 1, hidden]
#
#         attn_out = attn_out.squeeze(1)  # [batch, hidden]
#
#         # Concatenate and classify
#         combined = torch.cat([attn_out, noisy_trg_cond], dim=-1)
#         logits = self.classifier(combined).squeeze(-1)
#
#         return logits


# 3503496 (10 epochs)
# class ConditionalEncoder(nn.Module):
#     def __init__(self, encoder_link, tokenizer):
#         super().__init__()
#         self.encoder_link = encoder_link

#         if "bert" in encoder_link.lower():
#             # Загружаем BERT
#             self.bert = BertModel.from_pretrained(encoder_link)
#             self.tokenizer = tokenizer
#             hidden_dim = self.bert.config.hidden_size

#             # # Получаем эмбеддинги специальных токенов
#             cls_embedding = self.bert.embeddings.word_embeddings.weight[self.tokenizer.cls_token_id].detach().clone()
#             sep_embedding = self.bert.embeddings.word_embeddings.weight[self.tokenizer.sep_token_id].detach().clone()

#             self.register_buffer('cls_embedding', cls_embedding)
#             self.register_buffer('sep_embedding', sep_embedding)

#         else:
#             raise Exception("Unknown encoder name")

#         self.hidden_dim = hidden_dim

#         # Time embedding MLP
#         self.time_mlp = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim * 2),
#             nn.SiLU(),
#             nn.Linear(hidden_dim * 2, hidden_dim)
#         )

#         # Score head: предсказывает скалярный логит f(xt, t, y) ∈ ℝ
#         # Согласно PDF: p(y|xt) = σ(f)
#         self.score_head = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 1),  # Выход: скалярный логит
#         )

#     def timestep_embedding(self, timesteps, dim, max_period=10000):
#         """Sinusoidal time embedding"""
#         half = dim // 2
#         freqs = torch.exp(
#             -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
#         ).to(device=timesteps.device)
#         args = timesteps[:, None].float() * freqs[None]
#         embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
#         if dim % 2:
#             embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
#         return embedding

#     def forward(self, src_embeds, noisy_trg_embeds, t):
#         """
#         Прямой проход: предсказывает логит f(xt, t, y) ∈ ℝ

#         Args:
#             src_embeds: [batch, hidden] - CLS токены source (префикс y)
#             noisy_trg_embeds: [batch, hidden] - зашумленные target embeddings (xt)
#             t: [batch] - временные шаги

#         Returns:
#             logits: [batch] - скалярный логит f для каждого примера
#         """
#         device = src_embeds.device
#         batch_size = src_embeds.shape[0]

#         # 1. Time embedding
#         t_emb = self.timestep_embedding(t, self.hidden_dim).to(device)
#         t_embed = self.time_mlp(t_emb)  # [batch, hidden]

#         # 2. Подготавливаем специальные токены
#         cls_token = self.cls_embedding.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden]
#         cls_token = cls_token.expand(batch_size, 1, self.hidden_dim).to(device)

#         sep_token = self.sep_embedding.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden]
#         sep_token = sep_token.expand(batch_size, 1, self.hidden_dim).to(device)

#         # 3. Собираем последовательность для BERT
#         # Формат: [CLS] time [SEP] source [SEP] noisy_target
#         inputs_embeds = torch.cat([
#             cls_token,  # позиция 0
#             t_embed.unsqueeze(1),  # позиция 1
#             src_embeds.unsqueeze(1),  # позиция 2
#             sep_token,  # позиция 3
#             noisy_trg_embeds.unsqueeze(1),  # позиция 4
#         ], dim=1)

#         # token_type_ids = torch.tensor([
#         #     [0, 0, 0, 0, 1]  # segment 0: CLS+time+src+SEP, segment 1: trg
#         # ]).expand(batch_size, -1).to(device)

#         # 4. Пропускаем через BERT
#         outputs = self.bert(
#             inputs_embeds=inputs_embeds,
#             # token_type_ids=token_type_ids,
#             return_dict=True
#         )

#         # 5. Берем CLS токен из выхода (позиция 0)
#         cls_output = outputs.last_hidden_state[:, 0, :]  # [batch, hidden]

#         # 6. Пропускаем через score head для получения скалярного логита
#         logits = self.score_head(cls_output)  # [batch, 1]

#         # ИСПРАВЛЕНИЕ: squeeze для получения [batch] вместо [batch, 1]
#         logits = logits.squeeze(-1)  # [batch]

#         return logits


################################################################
# class ConditionalEncoder(nn.Module):
#     def __init__(self, encoder_link, tokenizer):
#         super().__init__()
#         self.encoder_link = encoder_link
#
#         if "bert" in encoder_link.lower():
#             # Загружаем BERT
#             self.bert = BertModel.from_pretrained(encoder_link)
#             self.tokenizer = tokenizer
#             hidden_dim = self.bert.config.hidden_size
#
#             # Получаем эмбеддинги специальных токенов
#             self.cls_embedding = self.bert.embeddings.word_embeddings(
#                 torch.tensor([self.tokenizer.cls_token_id], device=self.bert.device)
#             ).squeeze(0)  # [hidden_dim]
#
#             self.sep_embedding = self.bert.embeddings.word_embeddings(
#                 torch.tensor([self.tokenizer.sep_token_id], device=self.bert.device)
#             ).squeeze(0)  # [hidden_dim]
#         else:
#             raise Exception("Unknown encoder name")
#
#         self.hidden_dim = hidden_dim
#
#         # Time embedding MLP
#         self.time_mlp = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim * 2),
#             nn.SiLU(),
#             nn.Linear(hidden_dim * 2, hidden_dim)
#         )
#
#         # Score head: предсказывает ∇_trg log p(trg|src, t)
#         self.score_head = nn.Sequential(
#             nn.Linear(hidden_dim, 1),
#         )
#
#     def timestep_embedding(self, timesteps, dim, max_period=10000):
#         half = dim // 2
#         freqs = torch.exp(
#             -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
#         ).to(device=timesteps.device)
#         args = timesteps[:, None].float() * freqs[None]
#         embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
#         if dim % 2:
#             embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
#         return embedding
#
#     def forward(self, src_embeds, noisy_trg_embeds, t):
#         """
#         Прямой проход: предсказывает score = ∇_trg log p(trg|src, t)
#
#         Args:
#             src_embeds: [batch, hidden] - CLS токены source
#             noisy_trg_embeds: [batch, hidden] - зашумленные target
#             t: [batch] - временные шаги
#
#         Returns:
#             score: [batch, hidden] - логарифмический градиент
#         """
#         device = src_embeds.device
#         batch_size = src_embeds.shape[0]
#
#         # 1. Time embedding
#         t_emb = self.timestep_embedding(t, self.hidden_dim).to(device)
#         t_embed = self.time_mlp(t_emb)  # [batch, hidden]
#
#         # 2. Подготавливаем специальные токены
#         cls_token = self.cls_embedding.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden]
#         cls_token = cls_token.expand(batch_size, 1, self.hidden_dim).to(device)
#
#         sep_token = self.sep_embedding.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden]
#         sep_token = sep_token.expand(batch_size, 1, self.hidden_dim).to(device)
#
#         # 3. Собираем последовательность для BERT
#         # Формат: [CLS] time [SEP] source [SEP] target
#         # Только один [SEP] между source и target, время в начале
#         inputs_embeds = torch.cat([
#             cls_token,  # [CLS] - позиция 0
#             t_embed.unsqueeze(1),  # time token - позиция 1
#             sep_token,  # [SEP] - позиция 2
#             src_embeds.unsqueeze(1),  # source - позиция 3
#             sep_token,  # [SEP] - позиция 4 (второй SEP не нужен)
#             noisy_trg_embeds.unsqueeze(1),  # target - позиция 5
#         ], dim=1)  # [batch, 6, hidden]
#
#         # 4. Пропускаем через BERT
#         outputs = self.bert(
#             inputs_embeds=inputs_embeds,
#             return_dict=True
#         )
#
#         # 5. Берем CLS токен из выхода (позиция 0)
#         cls_output = outputs.last_hidden_state[:, 0, :]  # [batch, hidden]
#
#         # 6. Пропускаем через score head
#         score = self.score_head(cls_output)  # [batch, hidden]
#
#         return score


# class ConditionalEncoder(nn.Module):
#     def __init__(self, encoder_link):
#         super().__init__()
#         self.encoder_link = encoder_link
#         if "bert" in encoder_link.lower():
#             self.model = BertModel.from_pretrained(self.encoder_link)
#         else:
#             raise Exception("Unknown encoder name. Add encoder to ./model/conditional_encoder.py")
#
#         hidden_dim = self.model.config.hidden_size
#
#         self.time_mlp = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim * 2),
#             nn.SiLU(),
#             nn.Linear(hidden_dim * 2, hidden_dim)
#         )
#
#         # Классификатор: linear -> ReLU -> linear
#         self.classifier = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 2)
#         )
#
#     def timestep_embedding(self, timesteps, dim, max_period=10000):
#         half = dim // 2
#         freqs = torch.exp(
#             -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
#         ).to(device=timesteps.device)
#         args = timesteps[:, None].float() * freqs[None]
#         embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
#         if dim % 2:
#             embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
#         return embedding
#
#     def forward(self, inputs_embeds, t):
#         hidden_dim = inputs_embeds.shape[-1]
#
#         t_emb = self.timestep_embedding(t, hidden_dim)
#         t_embed = self.time_mlp(t_emb)
#
#         inputs_embeds = torch.cat([t_embed, inputs_embeds], dim=1)
#
#         outputs = self.model(
#             inputs_embeds=inputs_embeds,
#         )
#
#         return outputs.logits


# class ConditionalEncoder(nn.Module):
#     def __init__(self, encoder_link, temperature=2.0):
#         super().__init__()
#         self.encoder_link = encoder_link
#         self.temperature = temperature
#
#         if "bert" in encoder_link.lower():
#             self.model = BertModel.from_pretrained(self.encoder_link)
#         else:
#             raise Exception("Unknown encoder name.")
#
#         hidden_dim = self.model.config.hidden_size
#
#         # ДОБАВЬ segment embeddings
#         self.segment_embeddings = nn.Embedding(2, hidden_dim)
#
#         self.time_mlp = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim * 2),
#             nn.SiLU(),
#             nn.Linear(hidden_dim * 2, hidden_dim)
#         )
#
#         self.classifier = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 2)
#         )
#
#     def timestep_embedding(self, timesteps, dim, max_period=10000):
#         half = dim // 2
#         freqs = torch.exp(
#             -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
#         ).to(device=timesteps.device)
#         args = timesteps[:, None].float() * freqs[None]
#         embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
#         if dim % 2:
#             embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
#         return embedding
#
#     def forward(self, inputs_embeds, attention_mask, t):
#         batch_size, seq_len, hidden_dim = inputs_embeds.shape
#
#         # Segment embeddings: первая половина = 0 (src), вторая = 1 (trg)
#         half_len = seq_len // 2
#         segment_ids = torch.cat([
#             torch.zeros(batch_size, half_len, dtype=torch.long, device=inputs_embeds.device),
#             torch.ones(batch_size, half_len, dtype=torch.long, device=inputs_embeds.device)
#         ], dim=1)
#
#         segment_embeds = self.segment_embeddings(segment_ids)
#         inputs_embeds = inputs_embeds + segment_embeds
#
#         # Time embedding
#         t_emb = self.timestep_embedding(t, hidden_dim)
#         t_embed = self.time_mlp(t_emb).unsqueeze(1)
#
#         inputs_embeds = torch.cat([t_embed, inputs_embeds], dim=1)
#
#         t_mask = torch.ones(batch_size, 1, dtype=attention_mask.dtype, device=attention_mask.device)
#         attention_mask = torch.cat([t_mask, attention_mask], dim=1)
#
#         outputs = self.model(
#             inputs_embeds=inputs_embeds,
#             attention_mask=attention_mask
#         )
#
#         pooled = outputs.pooler_output
#         logits = self.classifier(pooled) / self.temperature
#
#         return logits


# class ConditionalEncoder(nn.Module):
#     def __init__(self, encoder_link):
#         super().__init__()
#         self.encoder_link = encoder_link
#         if "bert" in encoder_link.lower():
#             self.model = BertModel.from_pretrained(self.encoder_link)
#         else:
#             raise Exception("Unknown encoder name. Add encoder to ./model/conditional_encoder.py")
#
#         hidden_dim = self.model.config.hidden_size
#
#         self.time_mlp = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim * 2),
#             nn.SiLU(),
#             nn.Linear(hidden_dim * 2, hidden_dim)
#         )
#
#         # Классификатор: linear -> ReLU -> linear
#         self.classifier = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 2)
#         )
#
#     def timestep_embedding(self, timesteps, dim, max_period=10000):
#         half = dim // 2
#         freqs = torch.exp(
#             -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
#         ).to(device=timesteps.device)
#         args = timesteps[:, None].float() * freqs[None]
#         embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
#         if dim % 2:
#             embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
#         return embedding
#
#     def forward(self, inputs_embeds, attention_mask, t):
#         batch_size = inputs_embeds.shape[0]
#         hidden_dim = inputs_embeds.shape[-1]
#
#         t_emb = self.timestep_embedding(t, hidden_dim)
#         t_embed = self.time_mlp(t_emb)
#         t_embed = t_embed.unsqueeze(1)
#
#         inputs_embeds = torch.cat([t_embed, inputs_embeds], dim=1)
#
#         t_mask = torch.ones(batch_size, 1, dtype=attention_mask.dtype, device=attention_mask.device)
#         attention_mask = torch.cat([t_mask, attention_mask], dim=1)
#
#         outputs = self.model(
#             inputs_embeds=inputs_embeds,
#             attention_mask=attention_mask
#         )
#
#         # Используем pooler_output (CLS token)
#         pooled = outputs.pooler_output  # [batch_size, hidden_dim]
#         logits = self.classifier(pooled)
#
#         return logits


# from transformers import BertForNextSentencePrediction
# class ConditionalEncoder(nn.Module):
#     def __init__(self, encoder_link):
#         super().__init__()
#         self.encoder_link = encoder_link
#         if "bert" in encoder_link.lower():
#             self.model = BertForNextSentencePrediction.from_pretrained(self.encoder_link)
#
#             # ДОБАВЬ ЭТО - переинициализация последнего слоя
#             self.model.cls.seq_relationship.weight.data.normal_(mean=0.0, std=0.02)
#             self.model.cls.seq_relationship.bias.data.zero_()
#         else:
#             raise Exception("Unknown encoder name.")
#
#     def forward(self, inputs_embeds, attention_mask):
#         outputs = self.model(
#             inputs_embeds=inputs_embeds,
#             attention_mask=attention_mask
#         )
#         return outputs.logits


class ConditionalEncoderOld(nn.Module):
    def __init__(self, encoder_link):
        super().__init__()
        self.encoder_link = encoder_link
        if "bert" in encoder_link.lower():
            self.model = BertForNextSentencePrediction.from_pretrained( self.encoder_link)
        else:
            raise Exception("Unknown encoder name. Add encoder to ./model/conditional_encoder.py")
        hidden_dim = self.model.config.hidden_size

        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )


    def timestep_embedding(self, timesteps, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, inputs_embeds, attention_mask, t=None):
        # Закомментируй добавление time embedding
        # batch_size = inputs_embeds.shape[0]
        # hidden_dim = inputs_embeds.shape[-1]
        # t_emb = self.timestep_embedding(t, hidden_dim)
        # t_embed = self.time_mlp(t_emb)
        # t_embed = t_embed.unsqueeze(1)
        # inputs_embeds = torch.cat([t_embed, inputs_embeds], dim=1)
        # t_mask = torch.ones(batch_size, 1, dtype=attention_mask.dtype, device=attention_mask.device)
        # attention_mask = torch.cat([t_mask, attention_mask], dim=1)

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask
        )
        return outputs.logits

    def forward(self, inputs_embeds, attention_mask, t):
        batch_size = inputs_embeds.shape[0]
        hidden_dim = inputs_embeds.shape[-1]

        t_emb = self.timestep_embedding(t, hidden_dim)
        t_embed = self.time_mlp(t_emb)
        t_embed = t_embed.unsqueeze(1)

        inputs_embeds = torch.cat([t_embed, inputs_embeds], dim=1)

        t_mask = torch.ones(batch_size, 1, dtype=attention_mask.dtype, device=attention_mask.device)
        attention_mask = torch.cat([t_mask, attention_mask], dim=1)

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask
        )

        return outputs.logits
