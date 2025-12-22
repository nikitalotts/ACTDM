import math
import torch
import torch.nn as nn

from transformers import BertModel
import math
import torch
import torch.nn as nn
from transformers import BertForNextSentencePrediction


class ConditionalEncoder(nn.Module):
    def __init__(self, encoder_link, tokenizer):
        super().__init__()
        self.encoder_link = encoder_link

        if "bert" in encoder_link.lower():
            # Загружаем BERT
            self.bert = BertModel.from_pretrained(encoder_link)
            self.tokenizer = tokenizer
            hidden_dim = self.bert.config.hidden_size

            # Получаем эмбеддинги специальных токенов
            self.cls_embedding = self.bert.embeddings.word_embeddings(
                torch.tensor([self.tokenizer.cls_token_id], device=self.bert.device)
            ).squeeze(0)  # [hidden_dim]

            self.sep_embedding = self.bert.embeddings.word_embeddings(
                torch.tensor([self.tokenizer.sep_token_id], device=self.bert.device)
            ).squeeze(0)  # [hidden_dim]
        else:
            raise Exception("Unknown encoder name")

        self.hidden_dim = hidden_dim

        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        # Score head: предсказывает ∇_trg log p(trg|src, t)
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
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

    def forward(self, src_embeds, noisy_trg_embeds, t):
        """
        Прямой проход: предсказывает score = ∇_trg log p(trg|src, t)

        Args:
            src_embeds: [batch, hidden] - CLS токены source
            noisy_trg_embeds: [batch, hidden] - зашумленные target
            t: [batch] - временные шаги

        Returns:
            score: [batch, hidden] - логарифмический градиент
        """
        device = src_embeds.device
        batch_size = src_embeds.shape[0]

        # 1. Time embedding
        t_emb = self.timestep_embedding(t, self.hidden_dim).to(device)
        t_embed = self.time_mlp(t_emb)  # [batch, hidden]

        # 2. Подготавливаем специальные токены
        cls_token = self.cls_embedding.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden]
        cls_token = cls_token.expand(batch_size, 1, self.hidden_dim).to(device)

        sep_token = self.sep_embedding.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden]
        sep_token = sep_token.expand(batch_size, 1, self.hidden_dim).to(device)

        # 3. Собираем последовательность для BERT
        # Формат: [CLS] time [SEP] source [SEP] target
        # Только один [SEP] между source и target, время в начале
        inputs_embeds = torch.cat([
            cls_token,  # [CLS] - позиция 0
            t_embed.unsqueeze(1),  # time token - позиция 1
            sep_token,  # [SEP] - позиция 2
            src_embeds.unsqueeze(1),  # source - позиция 3
            sep_token,  # [SEP] - позиция 4 (второй SEP не нужен)
            noisy_trg_embeds.unsqueeze(1),  # target - позиция 5
        ], dim=1)  # [batch, 6, hidden]

        # 4. Пропускаем через BERT
        outputs = self.bert(
            inputs_embeds=inputs_embeds,
            return_dict=True
        )

        # 5. Берем CLS токен из выхода (позиция 0)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [batch, hidden]

        # 6. Пропускаем через score head
        score = self.score_head(cls_output)  # [batch, hidden]

        return score


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
