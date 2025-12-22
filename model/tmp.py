"""
conditional_encoder_debug.py

Debug + overfit helper for ConditionalEncoder training

How to use:
1. Put this file next to your project (it imports your modules: Encoder, ConditionalEncoder, create_config, etc.).
2. Run: python conditional_encoder_debug.py --mode overfit  # to try overfitting 1-2 batches
   or    python conditional_encoder_debug.py --mode debug    # to run one epoch with extra prints

This script adds diagnostic prints inside loss computation, gradient checks, and provides an overfit routine
that trains on 1-2 batches for many steps so you can verify the model can learn.

Set environment variable DEBUG_COND_ENCODER=1 to enable detailed prints.
"""
import argparse
import os
from utils.util import parse
import torch
import time
from tqdm import tqdm

from create_config import create_config
from transformers import AutoTokenizer
from model.encoder import Encoder
from model.conditional_encoder import ConditionalEncoder
from model.enc_normalizer import EncNormalizer
from data.dataset import get_dataset_iter
from diffusion_utils.dynamic import DynamicSDE

# --- helper to load datasets like in your script ---
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


# --- Enhanced loss step with diagnostics ---
def loss_step(batch, tokenizer, encoder, cond_encoder, config, device, empty_embeds, empty_mask, eval=False, debug=False):
    # tokenization
    src = tokenizer(
        batch['text_src'],
        add_special_tokens=True,
        padding='max_length',
        truncation=True,
        max_length=config.cond_encoder.max_sequence_len,
        return_tensors="pt",
        return_token_type_ids=False,
    ).to(device)

    trg = tokenizer(
        batch['text_trg'],
        add_special_tokens=True,
        padding='max_length',
        truncation=True,
        max_length=config.cond_encoder.max_sequence_len,
        return_tensors="pt",
        return_token_type_ids=False,
    ).to(device)

    # get encoder embeddings (encoder is frozen during this step in the original script)
    with torch.no_grad():
        if device.type == 'cuda':
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                src_embeds = encoder(input_ids=src['input_ids'], attention_mask=src['attention_mask'])
                trg_embeds = encoder(input_ids=trg['input_ids'], attention_mask=trg['attention_mask'])
        else:
            src_embeds = encoder(input_ids=src['input_ids'], attention_mask=src['attention_mask'])
            trg_embeds = encoder(input_ids=trg['input_ids'], attention_mask=trg['attention_mask'])

    batch_size = src_embeds.shape[0]

    # positive pairs
    src_embeds_pos = src_embeds
    trg_embeds_pos = trg_embeds
    src_mask_pos = src['attention_mask']
    trg_mask_pos = trg['attention_mask']
    labels_pos = torch.ones(batch_size, dtype=torch.long, device=device)

    # negative pairs (roll)
    indices = torch.roll(torch.arange(batch_size, device=device), shifts=1)
    trg_embeds_neg = trg_embeds[indices].clone()
    trg_mask_neg = trg['attention_mask'][indices].clone()

    if not eval and config.cond_encoder.empty_trg_prob > 0:
        empty_mask_bool = torch.rand(batch_size, device=device) < config.cond_encoder.empty_trg_prob
        if empty_mask_bool.any():
            trg_embeds_neg[empty_mask_bool] = empty_embeds
            trg_mask_neg[empty_mask_bool] = empty_mask

    src_embeds_neg = src_embeds
    src_mask_neg = src['attention_mask']
    labels_neg = torch.zeros(batch_size, dtype=torch.long, device=device)

    src_embeds_all = torch.cat([src_embeds_pos, src_embeds_neg], dim=0)
    trg_embeds_all = torch.cat([trg_embeds_pos, trg_embeds_neg], dim=0)
    src_mask_all = torch.cat([src_mask_pos, src_mask_neg], dim=0)
    trg_mask_all = torch.cat([trg_mask_pos, trg_mask_neg], dim=0)
    labels_all = torch.cat([labels_pos, labels_neg], dim=0)

    total_batch_size = src_embeds_all.shape[0]

    # # optionally add SDE noise
    # if not eval and hasattr(config, 'cond_encoder'):
    #     dynamic = DynamicSDE(config=config)
    #     t = (torch.rand(total_batch_size, device=device) *
    #          (config.cond_encoder.T - config.cond_encoder.eps) + config.cond_encoder.eps)
    #     marg_forward = dynamic.marginal(trg_embeds_all, t)
    #     noisy_trg_embeds = marg_forward['x_t']
    # else:
    #     t = torch.ones(total_batch_size, device=device) * getattr(config.cond_encoder, 'eps', 1e-3)
    #     noisy_trg_embeds = trg_embeds_all
    t = torch.ones(total_batch_size, device=device) * getattr(config.cond_encoder, 'eps', 1e-3)
    noisy_trg_embeds = trg_embeds_all

    # denormalize if encoder used normalizer
    with torch.no_grad():
        if not config.emb:
            if hasattr(encoder, 'module'):
                src_embeds_all = encoder.module.enc_normalizer.denormalize(src_embeds_all)
                noisy_trg_embeds = encoder.module.enc_normalizer.denormalize(noisy_trg_embeds)
            else:
                src_embeds_all = encoder.enc_normalizer.denormalize(src_embeds_all)
                noisy_trg_embeds = encoder.enc_normalizer.denormalize(noisy_trg_embeds)

    combined_embeds = torch.cat([src_embeds_all, noisy_trg_embeds], dim=1)
    combined_mask = torch.cat([src_mask_all, trg_mask_all], dim=1)

    # forward through conditional encoder
    if device.type == 'cuda':
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = cond_encoder(combined_embeds, combined_mask, t)
    else:
        logits = cond_encoder(combined_embeds, combined_mask, t)

    # --- DEBUG: печать форм и статистики (включать через env DEBUG_COND_ENCODER=1) ---
    if os.environ.get("DEBUG_COND_ENCODER") == "1" or debug:
        try:
            with torch.no_grad():
                print("=== DEBUG loss_step ===")
                print("device:", device)
                print("src_embeds_all.shape:", getattr(src_embeds_all, "shape", None))
                print("noisy_trg_embeds.shape:", getattr(noisy_trg_embeds, "shape", None))
                print("combined_embeds.shape:", combined_embeds.shape)
                print("combined_mask.shape:", combined_mask.shape)
                print("t.shape/dtype/device:", getattr(t, "shape", None), getattr(t, "dtype", None), getattr(t, "device", None))
                print("logits.shape:", logits.shape)
                probs = torch.softmax(logits, dim=1)
                print("probs mean class1:", probs[:, 1].mean().item())
                print("labels_all counts:", torch.unique(labels_all, return_counts=True))
        except Exception as e:
            print("DEBUG loss_step error:", e)
    # ------------------------------------------------------------------------------

    loss = torch.nn.functional.cross_entropy(logits, labels_all)
    preds = torch.argmax(logits, dim=1)
    acc = torch.mean((preds == labels_all).float())

    if not eval:
        print(f"Pred distribution: 0={torch.sum(preds == 0).item()}, 1={torch.sum(preds == 1).item()}")
        print(f"Label distribution: 0={torch.sum(labels_all == 0).item()}, 1={torch.sum(labels_all == 1).item()}")

    return loss, acc


# --- gradient checker ---
def print_grad_stats(model):
    total = 0
    none_cnt = 0
    zero_cnt = 0
    small_cnt = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        total += 1
        if p.grad is None:
            none_cnt += 1
            print(f"NO GRAD: {name}")
        else:
            gm = p.grad.abs().mean().item()
            if gm == 0:
                zero_cnt += 1
                print(f"ZERO GRAD: {name}")
            if gm < 1e-8:
                small_cnt += 1
    print(f"Grad stats: total_params_with_grad={total}, no_grad={none_cnt}, zero_grad={zero_cnt}, tiny_grad<{1e-8}={small_cnt})")


# --- overfit routine ---
def overfit_on_few_batches(config, encoder, cond_encoder, tokenizer, device, n_batches=2, iters=300, lr=5e-5):
    print("Preparing overfit run...")
    train_dataset, _ = get_datasets(config)
    loader = torch.utils.data.DataLoader(next(train_dataset), batch_size=config.cond_encoder.batch_size, shuffle=True)
    it = iter(loader)
    small_batches = [next(it) for _ in range(n_batches)]

    optimizer = torch.optim.AdamW(cond_encoder.parameters(), lr=lr)

    # compute empty embeds once
    with torch.no_grad():
        empty_tokens = tokenizer([""], add_special_tokens=True, padding='max_length', truncation=True, max_length=config.cond_encoder.max_sequence_len, return_tensors='pt').to(device)
        empty_embeds = encoder(input_ids=empty_tokens['input_ids'], attention_mask=empty_tokens['attention_mask'])
        empty_mask = empty_tokens['attention_mask']

    cond_encoder.train()
    for step in range(iters):
        total_loss = 0.0
        total_acc = 0.0
        for batch in small_batches:
            loss, acc = loss_step(batch, tokenizer, encoder, cond_encoder, config, device, empty_embeds, empty_mask, eval=False, debug=(step % 50 == 0))
            optimizer.zero_grad()
            loss.backward()

            # --- DEBUG: проверить градиенты и их нормы (включать через env DEBUG_COND_ENCODER=1) ---
            if os.environ.get("DEBUG_COND_ENCODER") == "1":
                try:
                    none_grad = []
                    zero_grad = []
                    small_grad = []
                    total_trainable = 0
                    for name, p in cond_encoder.named_parameters():
                        if not p.requires_grad:
                            continue
                        total_trainable += 1
                        if p.grad is None:
                            none_grad.append(name)
                        else:
                            gabs = p.grad.abs().sum().item()
                            if gabs == 0:
                                zero_grad.append(name)
                            if gabs < 1e-8:
                                small_grad.append(name)
                    print(f"=== DEBUG grads BEFORE CLIP === total_trainable={total_trainable}, no_grad={len(none_grad)}, zero_grad={len(zero_grad)}, tiny_grad<{1e-8}={len(small_grad)}")
                    if len(none_grad) > 0:
                        print(" params with no grad (sample):", none_grad[:10])
                except Exception as e:
                    print("DEBUG grad check error:", e)
            # ------------------------------------------------------------------------------

            torch.nn.utils.clip_grad_norm_(cond_encoder.parameters(), max_norm=getattr(config.cond_encoder, 'max_norm', 1.0))

            # дополнительная индикация общей нормы градиентов
            if os.environ.get("DEBUG_COND_ENCODER") == "1":
                try:
                    total_grad_norm = 0.0
                    cnt = 0
                    for p in cond_encoder.parameters():
                        if p.grad is not None:
                            total_grad_norm += float(p.grad.norm().item())
                            cnt += 1
                    print("DEBUG total_grad_norm (sum of norms across params) =", total_grad_norm, "param_count_with_grad=", cnt)
                except Exception as e:
                    print("DEBUG grad norm error:", e)

            optimizer.step()
            total_loss += loss.item()
            total_acc += acc.item()
        if step % 10 == 0:
            print(f"Step {step}/{iters} avg loss={total_loss/len(small_batches):.4f} acc={total_acc/len(small_batches):.4f}")

    print("Overfit run finished.")


# --- debug run (single epoch with extra prints) ---
def debug_run(config, encoder, cond_encoder, tokenizer, device, debug_batches=5):
    train_dataset, valid_dataset = get_datasets(config)
    train_loader = torch.utils.data.DataLoader(next(train_dataset), batch_size=config.cond_encoder.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(next(valid_dataset), batch_size=config.cond_encoder.batch_size)

    optimizer = torch.optim.AdamW(cond_encoder.parameters(), lr=config.cond_encoder.lr)

    with torch.no_grad():
        empty_tokens = tokenizer([""], add_special_tokens=True, padding='max_length', truncation=True, max_length=config.cond_encoder.max_sequence_len, return_tensors='pt').to(device)
        empty_embeds = encoder(input_ids=empty_tokens['input_ids'], attention_mask=empty_tokens['attention_mask'])
        empty_mask = empty_tokens['attention_mask']

    cond_encoder.train()
    for i, batch in enumerate(tqdm(train_loader, total=debug_batches, desc="Debug training")):
        loss, acc = loss_step(batch, tokenizer, encoder, cond_encoder, config, device, empty_embeds, empty_mask, eval=False, debug=True)
        optimizer.zero_grad()
        loss.backward()
        print_grad_stats(cond_encoder)
        optimizer.step()
        if i >= debug_batches - 1:
            break

    # quick validation pass
    cond_encoder.eval()
    with torch.no_grad():
        total_loss = 0.0
        total_acc = 0.0
        total_n = 0
        for batch in valid_loader:
            loss, acc = loss_step(batch, tokenizer, encoder, cond_encoder, config, device, empty_embeds, empty_mask, eval=True, debug=False)
            total_n += len(batch['text_trg'])
            total_loss += loss.item() * len(batch['text_trg'])
            total_acc += acc.item() * len(batch['text_trg'])
        if total_n:
            print(f"Validation loss: {total_loss/total_n:.4f}, acc: {total_acc/total_n:.4f}")


# --- main harness ---
def main():

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--mode', choices=['overfit', 'debug'], default='overfit')
    # parser.add_argument('--iters', type=int, default=300)
    # parser.add_argument('--n_batches', type=int, default=2)
    # parser.add_argument('--config_args', nargs='*', default=None, help='Args forwarded to create_config')
    args = parse()
    args.mode = 'overfit'
    args.iters = 300
    args.n_batches = 1

    # parse config just like your original script
    config = create_config(args)

    if not config.emb:
        enc_normalizer = EncNormalizer(
            enc_mean_path=config.data.enc_gen_mean,
            enc_std_path=config.data.enc_gen_std,
        )
    else:
        enc_normalizer = None

    encoder = Encoder(config.model.encoder_link, enc_normalizer=enc_normalizer, is_change_sp_tokens=True, emb=config.emb).eval()
    tokenizer = AutoTokenizer.from_pretrained(encoder.encoder_link)
    cond_encoder = ConditionalEncoder(encoder.encoder_link).train()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = encoder.to(device)
    cond_encoder = cond_encoder.to(device)

    print('Device', device)

    if args.mode == 'overfit':
        overfit_on_few_batches(config, encoder, cond_encoder, tokenizer, device, n_batches=args.n_batches, iters=args.iters)
    else:
        debug_run(config, encoder, cond_encoder, tokenizer, device)


if __name__ == '__main__':
    main()
