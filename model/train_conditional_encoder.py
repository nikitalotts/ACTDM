import os
import torch
import wandb
from create_config import create_config
import sys
import os
import torch
import wandb
import numpy as np
import argparse
from tqdm import tqdm
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

from utils.util import dict_to_cuda
from model.decoder import BertDecoder
from data.dataset import get_dataset_iter
from model.encoder import Encoder
from create_config import create_config
from model.enc_normalizer import EncNormalizer
from diffusion_utils.dynamic import DynamicSDE
from utils.util import parse
from model.conditional_encoder import ConditionalEncoder
from transformers import get_linear_schedule_with_warmup


def get_loaders(train_dataset, valid_dataset, batch_size):
    train_loader = DataLoader(
        next(train_dataset),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    valid_loader = DataLoader(
        next(valid_dataset),
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
    )

    return train_loader, valid_loader


def get_datasets(config):
    train_dataset = get_dataset_iter(
        config,
        dataset_name=config.cond_encoder.dataset,
        split="train",
        task='train_coniditonal_encoder'
    )
    test_dataset = get_dataset_iter(
        config,
        dataset_name=config.cond_encoder.dataset,
        split="test",
        task='train_coniditonal_encoder'
    )
    return train_dataset, test_dataset


def save_checkpoint(model, config):
    os.makedirs(config.training.checkpoints_folder, exist_ok=True)

    model.eval()
    torch.save(
        {
            "cond_encoder": model.state_dict(),
        },
        config.cond_encoder.cond_encoder_path
    )
    print(f"Save model to: {config.cond_encoder.cond_encoder_path}")


def get_empty_embedding(encoder, tokenizer, config, device):
    # Создаем zero embedding вместо токенизации пустой строки
    batch_size = 1
    seq_len = config.cond_encoder.max_sequence_len

    if isinstance(encoder, Encoder):
        hidden_dim = encoder.encoder.config.hidden_size
    else:
        hidden_dim = encoder.config.hidden_size

    # Zero embeddings
    empty_embeds = torch.zeros(batch_size, seq_len, hidden_dim, device=device)

    # Zero mask (все токены - padding)
    empty_mask = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)

    return empty_embeds, empty_mask

# def get_empty_embedding(encoder, tokenizer, config, device):
#     empty_tokens = tokenizer(
#         [""],
#         add_special_tokens=False,
#         padding=False,
#         truncation=True,
#         max_length=config.cond_encoder.max_sequence_len,
#         return_tensors="pt",
#         return_token_type_ids=False,
#     ).to(device)
#
#     with torch.no_grad():
#         if device.type == 'cuda':
#             with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
#                 empty_embeds = encoder(
#                     input_ids=empty_tokens["input_ids"].long(),
#                     attention_mask=empty_tokens["attention_mask"]
#                 )
#                 empty_embeds = empty_embeds if isinstance(encoder, Encoder) else empty_embeds.last_hidden_state
#         else:
#             empty_embeds = encoder(
#                 input_ids=empty_tokens["input_ids"].long(),
#                 attention_mask=empty_tokens["attention_mask"]
#             )
#             empty_embeds = empty_embeds if isinstance(encoder, Encoder) else empty_embeds.last_hidden_state
#
#     empty_mask = empty_tokens["attention_mask"]
#
#     return empty_embeds, empty_mask


def loss_step(batch, tokenizer, encoder, cond_encoder, config, device, empty_embeds, empty_mask, eval=False, batch_idx=0):
    if not eval and batch_idx == 0:
        print(f"\n=== RAW TEXT CHECK ===", file=sys.stderr, flush=True)
        print(f"text_src[0]: '{batch['text_src'][0]}'", file=sys.stderr, flush=True)
        print(f"text_trg[0]: '{batch['text_trg'][0]}'", file=sys.stderr, flush=True)
        print(f"Are texts identical? {batch['text_src'][0] == batch['text_trg'][0]}", file=sys.stderr, flush=True)

        # src = tokenizer(
        #     batch['text_src'],
        #     add_special_tokens=True,
        #     padding=True,
        #     truncation=True,
        #     max_length=config.decoder.max_sequence_len,
        #     return_tensors="pt",
        #     return_special_tokens_mask=True,
        #     return_token_type_ids=False,
        # ).to("cuda:0")
        #
        # with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        #     src_latent = encoder(
        #         input_ids=src["input_ids"],
        #         attention_mask=src["attention_mask"]
        #     )
        #     if not config.emb:
        #         src_latent = encoder.module.enc_normalizer.denormalize(src_latent)
        # src_mask = src["attention_mask"]

    src = tokenizer(
        batch['text_src'],
        add_special_tokens=True,
        padding=True,
        truncation=True,
        max_length=config.cond_encoder.max_sequence_len,
        return_tensors="pt",
        return_special_tokens_mask=True,
        return_token_type_ids=False
    ).to(device)

    trg = tokenizer(
        batch['text_trg'],
        add_special_tokens=True,
        padding=True,
        truncation=True,
        max_length=config.cond_encoder.max_sequence_len,
        return_tensors="pt",
        return_special_tokens_mask=True,
        return_token_type_ids=False
    ).to(device)


    if not eval and batch_idx == 0:
        print(f"\n=== TOKENIZATION CHECK ===", file=sys.stderr, flush=True)
        print(f"src input_ids[0]: {src['input_ids'][0]}", file=sys.stderr, flush=True)
        print(f"trg input_ids[0]: {trg['input_ids'][0]}", file=sys.stderr, flush=True)
        print(f"Are input_ids identical? {torch.equal(src['input_ids'][0], trg['input_ids'][0])}", file=sys.stderr,
              flush=True)

    with torch.no_grad():
        src_latent = encoder(
            input_ids=src["input_ids"].long(),
            attention_mask=src["attention_mask"]
        )
        src_latent = src_latent if isinstance(encoder, Encoder) else src_latent.last_hidden_state
        if not config.emb:
            src_latent = (encoder.module if device == 'cuda' else encoder).enc_normalizer.denormalize(src_latent)
        cls_src_latent = src_latent[:, 0, :]  # CLS tokens only
        trg_latent = encoder(
            input_ids=trg["input_ids"].long(),
            attention_mask=trg["attention_mask"]
        )
        trg_latent = trg_latent if isinstance(encoder, Encoder) else trg_latent.last_hidden_state
        if not config.emb:
            trg_latent = (encoder.module if device == 'cuda' else encoder).enc_normalizer.denormalize(trg_latent)
        cls_trg_latent = trg_latent[:, 0, :]  # CLS tokens only

    if not eval and batch_idx == 0:
        print(f"\n=== EMBEDDINGS CHECK ===", file=sys.stderr, flush=True)
        print(f"CLS embeddings:", file=sys.stderr, flush=True)
        print(f" sizes: {cls_src_latent.shape}(src) / {cls_trg_latent.shape}(tgt)")
        print(f"  src_cls mean: {src_latent.mean():.4f}, std: {src_latent.std():.4f}", file=sys.stderr, flush=True)
        print(f"  trg_cls mean: {trg_latent.mean():.4f}, std: {trg_latent.std():.4f}", file=sys.stderr, flush=True)
        print(f"  Are different? {F.cosine_similarity(cls_trg_latent[0].unsqueeze(0), cls_src_latent[0].unsqueeze(0))}", file=sys.stderr, flush=True)
        print(f"  Are different? {F.cosine_similarity(cls_trg_latent[1].unsqueeze(0), cls_src_latent[1].unsqueeze(0))}", file=sys.stderr, flush=True)
        print(f"  Are different? {F.cosine_similarity(cls_trg_latent[2].unsqueeze(0), cls_src_latent[2].unsqueeze(0))}", file=sys.stderr, flush=True)
        print(f"  Are different? {F.cosine_similarity(cls_trg_latent[3].unsqueeze(0), cls_src_latent[3].unsqueeze(0))}", file=sys.stderr, flush=True)
        print(f"  Are different? {F.cosine_similarity(cls_trg_latent[4].unsqueeze(0), cls_src_latent[4].unsqueeze(0))}", file=sys.stderr, flush=True)
        print(f"  Are different? {F.cosine_similarity(cls_trg_latent[5].unsqueeze(0), cls_src_latent[5].unsqueeze(0))}", file=sys.stderr, flush=True)
        print(f"  Are different? {F.cosine_similarity(cls_trg_latent[6].unsqueeze(0), cls_src_latent[6].unsqueeze(0))}", file=sys.stderr, flush=True)
        print(f"  Are different? {F.cosine_similarity(cls_trg_latent[7].unsqueeze(0), cls_src_latent[7].unsqueeze(0))}", file=sys.stderr, flush=True)
        print(f"  Are different? {F.cosine_similarity(cls_trg_latent[8].unsqueeze(0), cls_src_latent[8].unsqueeze(0))}", file=sys.stderr, flush=True)
        print(f"  Are different? {F.cosine_similarity(cls_trg_latent[9].unsqueeze(0), cls_src_latent[9].unsqueeze(0))}", file=sys.stderr, flush=True)
        print(f"  Are different? {F.cosine_similarity(cls_trg_latent[10].unsqueeze(0), cls_src_latent[10].unsqueeze(0))}", file=sys.stderr, flush=True)
        print(f"  Are different? {not torch.allclose(cls_trg_latent, cls_src_latent, atol=1e-4)}", file=sys.stderr, flush=True)

    batch_size = cls_src_latent.shape[0]

    # === EMBEDDINGS CHECK ===
    # CLS embeddings:
    #   src_cls mean: -0.0034, std: 0.5047
    #   trg_cls mean: -0.0033, std: 0.5000
    #   Are different? tensor([0.9552])
    #   Are different? tensor([0.9681])
    #   Are different? tensor([0.9704])
    #   Are different? tensor([0.9756])
    #   Are different? tensor([0.9580])
    #   Are different? tensor([0.9684])
    #  sizes: torch.Size([64, 768])(src) / torch.Size([64, 768])(tgt)
    #   Are different? tensor([0.9687])
    #   Are different? tensor([0.9610])
    #   Are different? tensor([0.9648])
    #   Are different? tensor([0.9698])
    #   Are different? tensor([0.9613])
    #   Are different? True

    # positive pairs
    src_embeds_pos = cls_src_latent
    trg_embeds_pos = cls_trg_latent
    # src_mask_pos = src["attention_mask"]
    # trg_mask_pos = trg["attention_mask"]
    labels_pos = torch.ones(batch_size, dtype=torch.float32, device=device)

    # negative pairs
    indices = torch.randperm(batch_size, device=device)
    while (indices == torch.arange(batch_size, device=device)).any():
        indices = torch.randperm(batch_size, device=device)

    trg_embeds_neg = cls_trg_latent[indices].clone()
    # trg_mask_neg = trg["attention_mask"][indices].clone()

    if not eval and config.cond_encoder.empty_trg_prob > 0:
        empty_mask_bool = torch.rand(batch_size, device=device) < config.cond_encoder.empty_trg_prob

        if empty_mask_bool.any():
            trg_embeds_neg[empty_mask_bool] = empty_embeds
            # trg_mask_neg[empty_mask_bool] = empty_mask

    src_embeds_neg = cls_src_latent
    # src_mask_neg = src["attention_mask"]
    labels_neg = torch.zeros(batch_size, dtype=torch.float32, device=device)

    src_embeds_all = torch.cat([src_embeds_pos, src_embeds_neg], dim=0)
    trg_embeds_all = torch.cat([trg_embeds_pos, trg_embeds_neg], dim=0)
    # src_mask_all = torch.cat([src_mask_pos, src_mask_neg], dim=0)
    # trg_mask_all = torch.cat([trg_mask_pos, trg_mask_neg], dim=0)
    labels_all = torch.cat([labels_pos, labels_neg], dim=0)

    total_batch_size = src_embeds_all.shape[0]

    if not eval and batch_idx < 3:
        for i in range(min(3, batch_size)):
            print(f"Pos src: {batch['text_src'][i][:50]}...", file=sys.stderr, flush=True)
            print(f"Pos trg: {batch['text_trg'][i][:50]}...", file=sys.stderr, flush=True)
            print(f"Neg trg: {batch['text_trg'][indices[i].item()][:50]}...", file=sys.stderr, flush=True)
            print("---", file=sys.stderr, flush=True)

    # Зашумление
    if not eval:
        dynamic = DynamicSDE(config=config)
        if device.type == 'cuda':
            t = torch.cuda.FloatTensor(total_batch_size).uniform_() * (
                    config.cond_encoder.T - config.cond_encoder.eps) + config.cond_encoder.eps
        else:
            t = torch.FloatTensor(total_batch_size).uniform_() * (
                    config.cond_encoder.T - config.cond_encoder.eps) + config.cond_encoder.eps
            t = t.to(device)
        marg_forward = dynamic.marginal(trg_embeds_all, t)
        noisy_trg_embeds = marg_forward['x_t']
    else:
        t = torch.ones(total_batch_size, device=device) * config.cond_encoder.eps
        noisy_trg_embeds = trg_embeds_all

    preds = cond_encoder(src_embeds_all, noisy_trg_embeds, t)

    # if not eval and batch_idx < 3:
    #     print(f"Logits mean per class: {logits.mean(dim=0)}", file=sys.stderr, flush=True)
    #     print(f"Logits std per class: {logits.std(dim=0)}", file=sys.stderr, flush=True)
    #     print(f"Logits[:5]:\n{logits[:5]}", file=sys.stderr, flush=True)
    #     print(f"Probs[:5]:\n{F.softmax(logits[:5], dim=1)}", file=sys.stderr, flush=True)
    #
    #     # Cosine similarity между первыми примерами
    #     src_mask_bool = src["attention_mask"][0].bool()
    #     trg_mask_bool = trg["attention_mask"][0].bool()
    #     src_valid = src_embeds[0][src_mask_bool].mean(dim=0, keepdim=True)
    #     trg_valid = trg_embeds[0][trg_mask_bool].mean(dim=0, keepdim=True)
    #
    #     cos_sim = F.cosine_similarity(src_valid, trg_valid)
    #     print(f"  Cosine similarity (first example): {cos_sim.item():.4f}", file=sys.stderr, flush=True)

    loss = F.mse_loss(preds, labels_all.unsqueeze(1))

    preds = torch.argmax(preds, dim=1)
    acc = torch.mean((preds == labels_all).float())

    if not eval and batch_idx < 3:
        print(f"Pred distribution: 0={torch.sum(preds == 0).item()}, 1={torch.sum(preds == 1).item()}", file=sys.stderr, flush=True)
        print(f"Label distribution: 0={torch.sum(labels_all == 0).item()}, 1={torch.sum(labels_all == 1).item()}", file=sys.stderr, flush=True)

    return loss, acc


# def loss_step(batch, tokenizer, encoder, cond_encoder, config, device, empty_embeds, empty_mask, eval=False,
#               batch_idx=0):
#     src = tokenizer(
#         batch['text_src'],
#         add_special_tokens=True,
#         padding='max_length',
#         truncation=True,
#         max_length=config.cond_encoder.max_sequence_len,
#         return_tensors="pt",
#         return_token_type_ids=False,
#     ).to(device)
#
#     trg = tokenizer(
#         batch['text_trg'],
#         add_special_tokens=True,
#         padding='max_length',
#         truncation=True,
#         max_length=config.cond_encoder.max_sequence_len,
#         return_tensors="pt",
#         return_token_type_ids=False,
#     ).to(device)
#
#     with torch.no_grad():
#         src_embeds = encoder(
#             input_ids=src["input_ids"],
#             attention_mask=src["attention_mask"]
#         )
#         trg_embeds = encoder(
#             input_ids=trg["input_ids"],
#             attention_mask=trg["attention_mask"]
#         )
#
#     batch_size = src_embeds.shape[0]
#
#     # positive pairs
#     src_embeds_pos = src_embeds
#     trg_embeds_pos = trg_embeds
#     src_mask_pos = src["attention_mask"]
#     trg_mask_pos = trg["attention_mask"]
#     labels_pos = torch.ones(batch_size, dtype=torch.long, device=device)
#
#     # negative pairs
#     indices = torch.randperm(batch_size, device=device)
#     while (indices == torch.arange(batch_size, device=device)).any():
#         indices = torch.randperm(batch_size, device=device)
#
#     trg_embeds_neg = trg_embeds[indices].clone()
#     trg_mask_neg = trg["attention_mask"][indices].clone()
#
#     if not eval and config.cond_encoder.empty_trg_prob > 0:
#         empty_mask_bool = torch.rand(batch_size, device=device) < config.cond_encoder.empty_trg_prob
#
#         if empty_mask_bool.any():
#             trg_embeds_neg[empty_mask_bool] = empty_embeds
#             trg_mask_neg[empty_mask_bool] = empty_mask
#
#     src_embeds_neg = src_embeds
#     src_mask_neg = src["attention_mask"]
#     labels_neg = torch.zeros(batch_size, dtype=torch.long, device=device)
#
#     src_embeds_all = torch.cat([src_embeds_pos, src_embeds_neg], dim=0)
#     trg_embeds_all = torch.cat([trg_embeds_pos, trg_embeds_neg], dim=0)
#     src_mask_all = torch.cat([src_mask_pos, src_mask_neg], dim=0)
#     trg_mask_all = torch.cat([trg_mask_pos, trg_mask_neg], dim=0)
#     labels_all = torch.cat([labels_pos, labels_neg], dim=0)
#
#     if not eval and batch_idx < 3:
#         for i in range(min(3, batch_size)):
#             print(f"Pos src: {batch['text_src'][i][:50]}...", file=sys.stderr, flush=True)
#             print(f"Pos trg: {batch['text_trg'][i][:50]}...", file=sys.stderr, flush=True)
#             print(f"Neg trg: {batch['text_trg'][indices[i].item()][:50]}...", file=sys.stderr, flush=True)
#             print("---", file=sys.stderr, flush=True)
#
#         print(f"src_embeds mean: {src_embeds.mean():.4f}, std: {src_embeds.std():.4f}", file=sys.stderr, flush=True)
#         print(f"trg_embeds mean: {trg_embeds.mean():.4f}, std: {trg_embeds.std():.4f}", file=sys.stderr, flush=True)
#         print(f"src_embeds range: [{src_embeds.min():.4f}, {src_embeds.max():.4f}]", file=sys.stderr, flush=True)
#         print(f"trg_embeds range: [{trg_embeds.min():.4f}, {trg_embeds.max():.4f}]", file=sys.stderr, flush=True)
#         print("---", file=sys.stderr, flush=True)
#
#     noisy_trg_embeds = trg_embeds_all
#     combined_embeds = torch.cat([src_embeds_all, noisy_trg_embeds], dim=1)
#     combined_mask = torch.cat([src_mask_all, trg_mask_all], dim=1)
#
#     logits = cond_encoder(combined_embeds, combined_mask)
#
#     # ДОБАВЛЕНО
#     if not eval and batch_idx < 3:
#         print(f"Logits mean per class: {logits.mean(dim=0)}", file=sys.stderr, flush=True)
#         print(f"Logits std per class: {logits.std(dim=0)}", file=sys.stderr, flush=True)
#         print(f"Logits[:5]:\n{logits[:5]}", file=sys.stderr, flush=True)
#         print(f"Probs[:5]:\n{F.softmax(logits[:5], dim=1)}", file=sys.stderr, flush=True)
#
#     loss = F.cross_entropy(logits, labels_all)
#
#     preds = torch.argmax(logits, dim=1)
#     acc = torch.mean((preds == labels_all).float())
#
#     if not eval and batch_idx < 3:
#         print(f"Pred distribution: 0={torch.sum(preds == 0).item()}, 1={torch.sum(preds == 1).item()}", file=sys.stderr,
#               flush=True)
#         print(f"Label distribution: 0={torch.sum(labels_all == 0).item()}, 1={torch.sum(labels_all == 1).item()}",
#               file=sys.stderr, flush=True)
#
#     return loss, acc


# def loss_step(batch, tokenizer, encoder, cond_encoder, config, device, empty_embeds, empty_mask, eval=False):
#     src = tokenizer(
#         batch['text_src'],
#         add_special_tokens=True,
#         padding='max_length',
#         truncation=True,
#         max_length=config.cond_encoder.max_sequence_len,
#         return_tensors="pt",
#         return_token_type_ids=False,
#     ).to(device)
#
#     trg = tokenizer(
#         batch['text_trg'],
#         add_special_tokens=True,
#         padding='max_length',
#         truncation=True,
#         max_length=config.cond_encoder.max_sequence_len,
#         return_tensors="pt",
#         return_token_type_ids=False,
#     ).to(device)
#
#     # with torch.no_grad():
#     #     if device.type == 'cuda':
#     #         with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
#     #             src_embeds = encoder(
#     #                 input_ids=src["input_ids"],
#     #                 attention_mask=src["attention_mask"]
#     #             )
#     #
#     #             trg_embeds = encoder(
#     #                 input_ids=trg["input_ids"],
#     #                 attention_mask=trg["attention_mask"]
#     #             )
#     #     else:
#     #         src_embeds = encoder(
#     #             input_ids=src["input_ids"],
#     #             attention_mask=src["attention_mask"]
#     #         )
#     #
#     #         trg_embeds = encoder(
#     #             input_ids=trg["input_ids"],
#     #             attention_mask=trg["attention_mask"]
#     #         )
#
#     with torch.no_grad():
#         src_embeds = encoder(
#             input_ids=src["input_ids"],
#             attention_mask=src["attention_mask"]
#         )
#         trg_embeds = encoder(
#             input_ids=trg["input_ids"],
#             attention_mask=trg["attention_mask"]
#         )
#
#     batch_size = src_embeds.shape[0]
#
#     # positive pairs
#     src_embeds_pos = src_embeds
#     trg_embeds_pos = trg_embeds
#     src_mask_pos = src["attention_mask"]
#     trg_mask_pos = trg["attention_mask"]
#     labels_pos = torch.ones(batch_size, dtype=torch.long, device=device)
#
#     # negative pairs
#     # indices = torch.roll(torch.arange(batch_size, device=device), shifts=1)
#     indices = torch.randperm(batch_size, device=device)
#     while (indices == torch.arange(batch_size, device=device)).any():
#         indices = torch.randperm(batch_size, device=device)
#
#     trg_embeds_neg = trg_embeds[indices].clone()
#     trg_mask_neg = trg["attention_mask"][indices].clone()
#
#     if not eval and config.cond_encoder.empty_trg_prob > 0:
#         empty_mask_bool = torch.rand(batch_size, device=device) < config.cond_encoder.empty_trg_prob
#
#         if empty_mask_bool.any():
#             trg_embeds_neg[empty_mask_bool] = empty_embeds
#             trg_mask_neg[empty_mask_bool] = empty_mask
#
#     src_embeds_neg = src_embeds
#     src_mask_neg = src["attention_mask"]
#     labels_neg = torch.zeros(batch_size, dtype=torch.long, device=device)
#
#     src_embeds_all = torch.cat([src_embeds_pos, src_embeds_neg], dim=0)
#     trg_embeds_all = torch.cat([trg_embeds_pos, trg_embeds_neg], dim=0)
#     src_mask_all = torch.cat([src_mask_pos, src_mask_neg], dim=0)
#     trg_mask_all = torch.cat([trg_mask_pos, trg_mask_neg], dim=0)
#     labels_all = torch.cat([labels_pos, labels_neg], dim=0)
#
#     total_batch_size = src_embeds_all.shape[0]
#
#     for i in range(min(3, batch_size)):
#         print(f"Pos src: {batch['text_src'][i][:50]}...")
#         print(f"Pos trg: {batch['text_trg'][i][:50]}...")
#         print(f"Neg trg: {batch['text_trg'][indices[i].item()][:50]}...")
#         print("---")
#
#
#     print(f"src_embeds mean: {src_embeds.mean():.4f}, std: {src_embeds.std():.4f}")
#     print(f"trg_embeds mean: {trg_embeds.mean():.4f}, std: {trg_embeds.std():.4f}")
#     print(f"src_embeds range: [{src_embeds.min():.4f}, {src_embeds.max():.4f}]")
#     print(f"trg_embeds range: [{trg_embeds.min():.4f}, {trg_embeds.max():.4f}]")
#     print("---")
#
#     # if not eval:
#     #     print(f"Batch size: {batch_size}")
#     #     print(f"Pos labels: {labels_pos.sum()}, Neg labels: {labels_neg.sum()}")
#     #     print(f"Total: pos={torch.sum(labels_all == 1)}, neg={torch.sum(labels_all == 0)}")
#     #
#     #     dynamic = DynamicSDE(config=config)
#     #     if device.type == 'cuda':
#     #         t = torch.cuda.FloatTensor(total_batch_size).uniform_() * (
#     #                 config.cond_encoder.T - config.cond_encoder.eps) + config.cond_encoder.eps
#     #     else:
#     #         t = torch.FloatTensor(total_batch_size).uniform_() * (
#     #                 config.cond_encoder.T - config.cond_encoder.eps) + config.cond_encoder.eps
#     #         t = t.to(device)
#     #     marg_forward = dynamic.marginal(trg_embeds_all, t)
#     #     noisy_trg_embeds = marg_forward['x_t']
#     # else:
#     #     t = torch.ones(total_batch_size, device=device) * config.cond_encoder.eps
#     #     noisy_trg_embeds = trg_embeds_all
#
#     # with torch.no_grad():
#     #     if not config.emb:
#     #         if hasattr(encoder, 'module'):
#     #             src_embeds_all = encoder.module.enc_normalizer.denormalize(src_embeds_all)
#     #             noisy_trg_embeds = encoder.module.enc_normalizer.denormalize(noisy_trg_embeds)
#     #         else:
#     #             src_embeds_all = encoder.enc_normalizer.denormalize(src_embeds_all)
#     #             noisy_trg_embeds = encoder.enc_normalizer.denormalize(noisy_trg_embeds)
#
#     noisy_trg_embeds = trg_embeds_all
#     combined_embeds = torch.cat([src_embeds_all, noisy_trg_embeds], dim=1)
#     combined_mask = torch.cat([src_mask_all, trg_mask_all], dim=1)
#
#     # if device.type == 'cuda':
#     #     with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
#     #         # no t
#     #         logits = cond_encoder(combined_embeds, combined_mask)
#     # else:
#     #     logits = cond_encoder(combined_embeds, combined_mask)
#     logits = cond_encoder(combined_embeds, combined_mask)
#
#     loss = F.cross_entropy(logits, labels_all)
#
#     # logits_one = logits[:, 1]
#     # loss = torch.nn.functional.binary_cross_entropy_with_logits(logits_one, labels_all.float())
#
#     preds = torch.argmax(logits, dim=1)
#     acc = torch.mean((preds == labels_all).float())
#
#     if not eval:
#         print(f"Pred distribution: 0={torch.sum(preds == 0).item()}, 1={torch.sum(preds == 1).item()}")
#         print(f"Label distribution: 0={torch.sum(labels_all == 0).item()}, 1={torch.sum(labels_all == 1).item()}")
#
#     return loss, acc


def train(config, encoder, cond_encoder, tokenizer, device):
    print("=== START TRAIN ===")
    total_number_params = sum(p.numel() for p in cond_encoder.parameters() if p.requires_grad)
    print(f"Num params: {total_number_params}")

    trainable_params = sum(p.numel() for p in cond_encoder.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in cond_encoder.parameters())
    print(f"Trainable: {trainable_params}, Total: {total_params}")

    batch_size = config.cond_encoder.batch_size
    print(f"Batch size: {batch_size}")

    print("Getting empty embedding...")
    empty_embeds, empty_mask = get_empty_embedding(encoder, tokenizer, config, device)
    print(f"Empty embedding shape: {empty_embeds.shape}")

    print("Getting datasets...")
    train_dataset, valid_dataset = get_datasets(config=config)
    print("Datasets loaded")

    optimizer = torch.optim.AdamW(
        cond_encoder.parameters(),
        lr=config.cond_encoder.lr,
        weight_decay=config.cond_encoder.weight_decay,
        betas=config.cond_encoder.betas,
    )
    print("Optimizer created")

    print("Creating data loaders...")
    train_loader, valid_loader = get_loaders(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        batch_size=batch_size
    )
    print(f"Train loader length: {len(train_loader)}")
    print(f"Valid loader length: {len(valid_loader)}")

    num_training_steps = len(train_loader) * config.cond_encoder.epochs
    num_warmup_steps = num_training_steps // 10
    print(f"Training steps: {num_training_steps}, Warmup steps: {num_warmup_steps}")

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    print("Scheduler created")

    step = 0
    print(f"Starting training for {config.cond_encoder.epochs} epochs...")
    for epoch in range(config.cond_encoder.epochs):
        print(f"\n=== EPOCH {epoch + 1}/{config.cond_encoder.epochs} ===")

        cond_encoder.train()
        print("Model set to train mode")

        print("Creating tqdm...")
        train_bar = tqdm(train_loader)
        print("Starting batch loop...")

        # for batch_idx, batch in enumerate(train_loader):
        for batch_idx, batch in enumerate(train_bar):
            print(f"\n--- Batch {batch_idx} ---")
            print(f"Batch keys: {batch.keys()}")
            print(f"text_src length: {len(batch['text_src'])}")
            print(f"text_trg length: {len(batch['text_trg'])}")

            loss, acc = loss_step(
                batch=batch,
                tokenizer=tokenizer,
                encoder=encoder,
                cond_encoder=cond_encoder,
                config=config,
                device=device,
                empty_embeds=empty_embeds,
                empty_mask=empty_mask,
                batch_idx=batch_idx
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                cond_encoder.parameters(),
                max_norm=config.cond_encoder.max_norm
            )
            optimizer.step()
            scheduler.step()

            wandb.log({'train loss': loss.item()}, step=step)
            wandb.log({'train accuracy': acc.item()}, step=step)

            train_bar.set_postfix({
                'Stage': 'Training',
                'Epoch': f"{epoch + 1}/{config.cond_encoder.epochs}",
                'Loss': loss.item()
            })

            step += 1

        cond_encoder.eval()
        with torch.no_grad():
            total_loss = 0.
            total_acc = 0.
            total_num = 0.

            valid_bar = tqdm(valid_loader)
            # for batch in valid_loader:
            for batch_idx, batch in enumerate(valid_bar):
                loss, acc = loss_step(
                    batch=batch,
                    tokenizer=tokenizer,
                    encoder=encoder,
                    cond_encoder=cond_encoder,
                    config=config,
                    eval=True,
                    device=device,
                    empty_embeds=empty_embeds,
                    empty_mask=empty_mask,
                    batch_idx=batch_idx
                )
                batch_size_cur = len(batch['text_trg'])
                total_loss += loss * batch_size_cur
                total_acc += acc * batch_size_cur
                total_num += batch_size_cur

            total_loss /= total_num
            total_acc /= total_num

            wandb.log({'valid loss': total_loss.item()}, step=step)
            wandb.log({'valid accuracy': total_acc.item()}, step=step)

            valid_bar.set_postfix({
                'Stage': 'Validation',
                'Epoch': f"{epoch + 1}/{config.cond_encoder.epochs}",
                'Loss': loss.item()
            })

        save_checkpoint(cond_encoder, config)


def main():
    args = parse()
    config = create_config(args)
    if not config.emb:
        enc_normalizer = EncNormalizer(
            enc_mean_path=config.data.enc_gen_mean,
            enc_std_path=config.data.enc_gen_std,
        )
    else:
        enc_normalizer = None
    encoder = Encoder(
        config.model.encoder_link,
        enc_normalizer=enc_normalizer,
        is_change_sp_tokens=False,
        emb=config.emb
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(encoder.encoder_link)
    # encoder = AutoModel.from_pretrained(config.model.encoder_link).eval()
    # tokenizer = AutoTokenizer.from_pretrained(config.model.encoder_link)

    cond_encoder = ConditionalEncoder(config.model.encoder_link, tokenizer).train()

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if num_gpus > 0:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if num_gpus > 1:
        encoder = torch.nn.DataParallel(encoder).to(device)
        cond_encoder = cond_encoder.to(device)
        print(f'Training on {num_gpus} GPUs')
    elif num_gpus == 1:
        encoder = encoder.to(device)
        cond_encoder = cond_encoder.to(device)
        print(f'Training on {num_gpus} GPU')
    else:
        encoder = encoder.to(device)
        cond_encoder = cond_encoder.to(device)
        print('Training on CPU')

    print(f"config.emb = {config.emb}")

    print('Training Conditional BERT NSP on rocstories')
    wandb.init(project=config.project_name, name="conditional_bert_nsp", mode="offline")

    train(config, encoder, cond_encoder, tokenizer, device)


if __name__ == '__main__':
    main()
