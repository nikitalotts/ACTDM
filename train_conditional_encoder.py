"""
Обучение классификатора для classifier guidance в текстовой диффузии.

Согласно PDF (Algorithm 2):
1. Выбираем случайный шаг t
2. Зашумляем правильное продолжение: x_t^+ = sqrt(α̅_t) * x_0 + sqrt(1-α̅_t) * ε
3. Зашумляем неправильное продолжение: x_t^- = sqrt(α̅_t) * x_0^- + sqrt(1-α̅_t) * ε'
4. Получаем логиты: f^+ = Classifier(x_t^+, t, y), f^- = Classifier(x_t^-, t, y)
5. Лосс: L_cls = -log σ(f^+) - log(1 - σ(f^-))
"""

import os
import sys
import torch
import wandb
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup

from data.dataset import get_dataset_iter
from model.encoder import Encoder
from create_config import create_config
from model.enc_normalizer import EncNormalizer
from diffusion_utils.dynamic import DynamicSDE
from utils.util import parse
from model.conditional_encoder import ConditionalEncoder


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
    )
    test_dataset = get_dataset_iter(
        config,
        dataset_name=config.cond_encoder.dataset,
        split="test",
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


def loss_step(epoch, batch, tokenizer, encoder, cond_encoder, config, device, eval=False, batch_idx=0):
    """
    Один шаг обучения классификатора.

    Согласно PDF (Algorithm 2, секция 4.2):
    - x_t^+ - зашумлённое правильное продолжение
    - x_t^- - зашумлённый случайный текст
    - L_cls = -log σ(f^+) - log(1 - σ(f^-))

    Это эквивалентно BCE loss:
    L = BCE(f^+, 1) + BCE(f^-, 0)
    """

    # Debug: проверка сырых текстов
    if not eval and batch_idx == 0:
        print(f"\n=== RAW TEXT CHECK ===", file=sys.stderr, flush=True)
        print(f"text_src[0]: '{batch['text_src'][0]}'", file=sys.stderr, flush=True)
        print(f"text_trg[0]: '{batch['text_trg'][0]}'", file=sys.stderr, flush=True)
        print(f"Are texts identical? {batch['text_src'][0] == batch['text_trg'][0]}", file=sys.stderr, flush=True)

    # Токенизация с add_special_tokens=True (консистентно с diffusion_holder)
    # Результат: [CLS] text [SEP] [PAD]...
    src = tokenizer(
        batch['text_src'],
        add_special_tokens=True,
        padding='max_length',
        truncation=True,
        max_length=config.cond_encoder.max_sequence_len,
        return_tensors="pt",
        return_special_tokens_mask=True,
        return_token_type_ids=False
    ).to(device)

    trg = tokenizer(
        batch['text_trg'],
        add_special_tokens=True,
        padding='max_length',
        truncation=True,
        max_length=config.cond_encoder.max_sequence_len,
        return_tensors="pt",
        return_special_tokens_mask=True,
        return_token_type_ids=False
    ).to(device)

    # Debug: проверка токенизации
    if not eval and batch_idx == 0:
        print(f"\n=== TOKENIZATION CHECK ===", file=sys.stderr, flush=True)
        print(f"src input_ids[0]: {src['input_ids'][0]}", file=sys.stderr, flush=True)
        print(f"trg input_ids[0]: {trg['input_ids'][0]}", file=sys.stderr, flush=True)
        print(f"Are input_ids identical? {torch.equal(src['input_ids'][0], trg['input_ids'][0])}", file=sys.stderr, flush=True)

    # Получаем ПОЛНЫЕ последовательности эмбеддингов от encoder
    # Эмбеддинги содержат: [CLS_emb] [tok1_emb] ... [SEP_emb] [PAD_emb]...
    with torch.no_grad():
        # y - префикс (условие)
        src_latent = encoder(
            input_ids=src["input_ids"].long(),
            attention_mask=src["attention_mask"]
        )
        src_latent = src_latent if isinstance(encoder, Encoder) else src_latent.last_hidden_state
        # [batch, seq_len, hidden]

        # x_0 - правильное продолжение (до зашумления)
        trg_latent = encoder(
            input_ids=trg["input_ids"].long(),
            attention_mask=trg["attention_mask"]
        )
        trg_latent = trg_latent if isinstance(encoder, Encoder) else trg_latent.last_hidden_state
        # [batch, seq_len, hidden]

    # Debug: проверка эмбеддингов
    if not eval and batch_idx == 0:
        print(f"\n=== EMBEDDINGS ===", file=sys.stderr, flush=True)
        print(f"src_latent shape: {src_latent.shape}", file=sys.stderr, flush=True)
        print(f"trg_latent shape: {trg_latent.shape}", file=sys.stderr, flush=True)
        print(f"src mean: {src_latent.mean():.4f}, std: {src_latent.std():.4f}", file=sys.stderr, flush=True)
        print(f"trg mean: {trg_latent.mean():.4f}, std: {trg_latent.std():.4f}", file=sys.stderr, flush=True)

    batch_size = src_latent.shape[0]

    # Сохраняем src_mask (trg_mask НЕ нужна для консистентности с инференсом)
    src_mask = src["attention_mask"]  # [batch, seq_len]

    # === POSITIVE PAIRS: (y, x_0^+) ===
    src_embeds_pos = src_latent
    trg_embeds_pos = trg_latent
    src_mask_pos = src_mask
    labels_pos = torch.ones(batch_size, dtype=torch.float32, device=device)

    # === NEGATIVE PAIRS: (y, x_0^-) ===
    # x_0^- - случайный текст (перемешанные trg из батча)
    indices = torch.randperm(batch_size, device=device)
    while (indices == torch.arange(batch_size, device=device)).any():
        indices = torch.randperm(batch_size, device=device)

    src_embeds_neg = src_latent
    trg_embeds_neg = trg_latent[indices]
    src_mask_neg = src_mask
    labels_neg = torch.zeros(batch_size, dtype=torch.float32, device=device)

    # Debug: косинусная близость
    if not eval and batch_idx == 0:
        # Берём CLS токены для анализа
        cls_src = src_latent[:, 0, :]
        cls_trg = trg_latent[:, 0, :]

        cos_sim_pos = F.cosine_similarity(cls_src, cls_trg, dim=-1)
        cos_sim_neg = F.cosine_similarity(cls_src, trg_latent[indices][:, 0, :], dim=-1)

        print(f"\n=== COSINE SIMILARITY (CLS tokens) ===", file=sys.stderr, flush=True)
        print(f"POSITIVE pairs: {cos_sim_pos.mean():.4f} ± {cos_sim_pos.std():.4f}", file=sys.stderr, flush=True)
        print(f"NEGATIVE pairs: {cos_sim_neg.mean():.4f} ± {cos_sim_neg.std():.4f}", file=sys.stderr, flush=True)
        print(f"Difference: {(cos_sim_pos.mean() - cos_sim_neg.mean()):.4f}", file=sys.stderr, flush=True)

    # Debug: примеры пар
    if not eval and batch_idx < 3:
        for i in range(min(3, batch_size)):
            print(f"Pos src: {batch['text_src'][i]}", file=sys.stderr, flush=True)
            print(f"Pos trg: {batch['text_trg'][i]}", file=sys.stderr, flush=True)
            print(f"Neg trg: {batch['text_trg'][indices[i].item()]}", file=sys.stderr, flush=True)
            print("---", file=sys.stderr, flush=True)

    # Объединяем positive и negative
    src_embeds_all = torch.cat([src_embeds_pos, src_embeds_neg], dim=0)
    trg_embeds_all = torch.cat([trg_embeds_pos, trg_embeds_neg], dim=0)
    src_mask_all = torch.cat([src_mask_pos, src_mask_neg], dim=0)
    labels_all = torch.cat([labels_pos, labels_neg], dim=0)

    total_batch_size = src_embeds_all.shape[0]

    # === ЗАШУМЛЕНИЕ (согласно PDF Algorithm 2) ===
    # x_t = sqrt(α̅_t) * x_0 + sqrt(1-α̅_t) * ε
    eps_t = 0.01 # eps сразу как на диффузии
    dynamic = DynamicSDE(config=config)

    current_max_t = dynamic.T

    # if device.type == 'cuda':
    #     t = torch.cuda.FloatTensor(total_batch_size).uniform_() * (current_max_t - eps_t) + eps_t
    # else:
    #     t = torch.FloatTensor(total_batch_size).uniform_() * (current_max_t - eps_t) + eps_t
    #     t = t.to(device)
    #
    # if not eval and batch_idx == 0:
    #     print(f"t ~ Uniform[{eps_t}, {current_max_t}]", file=sys.stderr, flush=True)
    #     print(f"t sample: {t[:5]}", file=sys.stderr, flush=True)

    u = torch.cuda.FloatTensor(total_batch_size).uniform_()
    t = eps_t + (u ** 0.5) * (current_max_t - eps_t)

    # Зашумляем x_0 -> x_t
    marg_forward = dynamic.marginal(trg_embeds_all, t)
    noisy_trg_embeds = marg_forward['x_t']

    # === FORWARD PASS ===
    # f(x_t, t, y) - логит классификатора
    # Передаём только src_mask, без trg_mask
    logits = cond_encoder(
        src_embeds=src_embeds_all,
        noisy_trg_embeds=noisy_trg_embeds,
        t=t,
        src_mask=src_mask_all
    )

    # === LOSS (согласно PDF секция 4.2) ===
    # L_cls = -log σ(f^+) - log(1 - σ(f^-))
    t_normalized = (t - eps_t) / (current_max_t - eps_t)
    weights = torch.clamp(1.0 - t_normalized, min=0.0)
    weights = weights / (weights.mean() + 1e-8)

    loss_per_sample = F.binary_cross_entropy_with_logits(logits, labels_all, reduction='none')
    loss = (loss_per_sample * weights).mean()

    # Метрики
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    acc = torch.mean((preds == labels_all).float())

    # Debug info (как в оригинале)
    if not eval:
        probs_pos = probs[:batch_size]
        probs_neg = probs[batch_size:]

        logits_pos = logits[:batch_size]
        logits_neg = logits[batch_size:]
        loss_pos = F.binary_cross_entropy_with_logits(logits_pos, torch.ones_like(logits_pos))
        loss_neg = F.binary_cross_entropy_with_logits(logits_neg, torch.zeros_like(logits_neg))

        print(f"\n=== LOSS & METRICS ===", file=sys.stderr, flush=True)
        print(f"loss_pos: {loss_pos.item():.4f}", file=sys.stderr, flush=True)
        print(f"loss_neg: {loss_neg.item():.4f}", file=sys.stderr, flush=True)
        print(f"total loss: {loss.item():.4f}", file=sys.stderr, flush=True)

        print(f"\nProbs positive (first 5): {probs_pos[:5]}", file=sys.stderr, flush=True)
        print(f"Probs negative (first 5): {probs_neg[:5]}", file=sys.stderr, flush=True)
        print(f"probs_pos mean: {probs_pos.mean():.4f}", file=sys.stderr, flush=True)
        print(f"probs_neg mean: {probs_neg.mean():.4f}", file=sys.stderr, flush=True)

        print(f"\nPred distribution: 0={torch.sum(preds == 0).item()}, 1={torch.sum(preds == 1).item()}",
              file=sys.stderr, flush=True)
        print(f"Label distribution: 0={torch.sum(labels_all == 0).item()}, 1={torch.sum(labels_all == 1).item()}",
              file=sys.stderr, flush=True)

        correct_pos = (probs_pos > 0.5).float().mean()
        correct_neg = (probs_neg < 0.5).float().mean()
        print(f"Accuracy positive: {correct_pos.item():.4f}", file=sys.stderr, flush=True)
        print(f"Accuracy negative: {correct_neg.item():.4f}", file=sys.stderr, flush=True)

    return loss, acc


def train(config, encoder, cond_encoder, tokenizer, device):
    print("=== START TRAIN ===")
    print("Training classifier for classifier guidance (PDF Algorithm 2)")

    total_number_params = sum(p.numel() for p in cond_encoder.parameters() if p.requires_grad)
    print(f"Num params: {total_number_params}")

    trainable_params = sum(p.numel() for p in cond_encoder.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in cond_encoder.parameters())
    print(f"Trainable: {trainable_params}, Total: {total_params}")

    batch_size = config.cond_encoder.batch_size
    print(f"Batch size: {batch_size}")

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

        for batch_idx, batch in enumerate(train_bar):
            print(f"\n--- Batch {batch_idx} ---")
            print(f"Batch keys: {batch.keys()}")
            print(f"text_src length: {len(batch['text_src'])}")
            print(f"text_trg length: {len(batch['text_trg'])}")

            loss, acc = loss_step(
                epoch=epoch,
                batch=batch,
                tokenizer=tokenizer,
                encoder=encoder,
                cond_encoder=cond_encoder,
                config=config,
                device=device,
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

        # Validation
        cond_encoder.eval()
        with torch.no_grad():
            total_loss = 0.
            total_acc = 0.
            total_num = 0.

            valid_bar = tqdm(valid_loader)
            for batch_idx, batch in enumerate(valid_bar):
                loss, acc = loss_step(
                    epoch=epoch,
                    batch=batch,
                    tokenizer=tokenizer,
                    encoder=encoder,
                    cond_encoder=cond_encoder,
                    config=config,
                    eval=True,
                    device=device,
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

            print(f"Validation - Loss: {total_loss.item():.4f}, Acc: {total_acc.item():.4f}")

        save_checkpoint(cond_encoder, config)


def main():
    args = parse()
    config = create_config(args)

    # Настройки
    # config.cond_encoder.epochs = 15
    config.cond_encoder.lr = 1e-4

    config.cond_encoder.epochs = 15
    # config.cond_encoder.lr = 1e-5
    config.cond_encoder.weight_decay = 0.01

    config.model.encoder_link = "bert-base-cased"
    config.decoder.mode = "transformer"
    config.decoder.decoder_path = "datasets/rocstories/decoder_rocstories_bert_cased_spt_3l_transformer_0_15x_t_noise.pt"
    config.training.checkpoints_folder = "checkpoints"
    config.training.checkpoints_prefix = "tencdm-bert-base-cased-384-0.0002-rocstories-cfg=0.0"
    config.training.checkpoint_name = "rocstory-bert-base-cased-sd-9-spt_100000"

    config.is_conditional = True
    config.cond_encoder.use_conditional_encoder = True

    # Encoder для получения эмбеддингов
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
        is_change_sp_tokens=True,
        emb=config.emb
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(encoder.encoder_link)

    # Conditional Encoder (классификатор)
    cond_encoder = ConditionalEncoder(config.model.encoder_link, tokenizer).train()

    # config.cond_encoder.cond_encoder_path = 'datasets/rocstories/enc_backup/conditional-encoder.pth'

    config.cond_encoder.cond_encoder_path += 'sptokenscratchweights'

    cond_encoder_path = config.cond_encoder.cond_encoder_path
    if os.path.exists(cond_encoder_path):
        print(f"Loading existing model from: {cond_encoder_path}")
        checkpoint = torch.load(cond_encoder_path, map_location='cpu')
        print('CONDINCH', "cond_encoder" in checkpoint, checkpoint.keys())
        cond_encoder_checkpoint = checkpoint["cond_encoder"] if "cond_encoder" in checkpoint else checkpoint
        cond_encoder.load_state_dict(cond_encoder_checkpoint)
        print(f"Successfully loaded model from: {cond_encoder_path}")
    else:
        print(f"No existing model found at: {cond_encoder_path}")
        print("Starting training from scratch")

    # Device setup
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
    print('Training Classifier for Classifier Guidance (PDF Algorithm 2)')

    wandb.init(project=config.project_name, name="classifier_guidance", mode="offline")

    print(config, end="\n\n\n")
    train(config, encoder, cond_encoder, tokenizer, device)


if __name__ == '__main__':
    main()