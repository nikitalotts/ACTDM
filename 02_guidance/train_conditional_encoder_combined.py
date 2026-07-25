
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
from copy import deepcopy

from data.dataset import get_dataset_iter
from model.encoder import Encoder
from create_config import create_config
from model.enc_normalizer import EncNormalizer
from diffusion_utils.dynamic import DynamicSDE
from utils.util import parse
from model.conditional_encoder import ConditionalEncoder
from model.score_estimator import ScoreEstimatorEMB


def get_loaders(train_dataset, valid_dataset, batch_size):
    train_loader = DataLoader(
        next(train_dataset),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=True
    )
    valid_loader = DataLoader(
        next(valid_dataset),
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
        drop_last=True
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
        {"cond_encoder": model.state_dict()},
        config.cond_encoder.cond_encoder_path
    )
    print(f"Save model to: {config.cond_encoder.cond_encoder_path}")


@torch.no_grad()
def predict_x0_from_xt(x_t, t, score_estimator, cond=None, cond_mask=None):
    x_0_self_cond = torch.zeros_like(x_t, dtype=x_t.dtype)
    x_0_pred = score_estimator(
        x_t=x_t,
        time_t=t,
        cond=cond,
        attention_mask=None,
        cond_mask=cond_mask,
        x_0_self_cond=x_0_self_cond
    )
    return x_0_pred


def loss_step(epoch, batch, tokenizer, encoder, cond_encoder, score_estimator,
              config, device, eval=False, batch_idx=0):
    if not eval and batch_idx == 0:
        print(f"\n=== RAW TEXT CHECK ===", file=sys.stderr, flush=True)
        print(f"text_src[0]: '{batch['text_src'][0]}'", file=sys.stderr, flush=True)
        print(f"text_trg[0]: '{batch['text_trg'][0]}'", file=sys.stderr, flush=True)
        print(f"Are texts identical? {batch['text_src'][0] == batch['text_trg'][0]}", file=sys.stderr, flush=True)

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

    with torch.no_grad():
        src_latent = encoder(
            input_ids=src["input_ids"].long(),
            attention_mask=src["attention_mask"]
        )
        src_latent = src_latent if isinstance(encoder, Encoder) else src_latent.last_hidden_state

        trg_latent = encoder(
            input_ids=trg["input_ids"].long(),
            attention_mask=trg["attention_mask"]
        )
        trg_latent = trg_latent if isinstance(encoder, Encoder) else trg_latent.last_hidden_state

    batch_size = src_latent.shape[0]
    src_mask = src["attention_mask"]
    trg_mask = trg["attention_mask"]

    if not eval and batch_idx == 0:
        print(f"\n=== EMBEDDINGS ===", file=sys.stderr, flush=True)
        print(f"src_latent shape: {src_latent.shape}", file=sys.stderr, flush=True)
        print(f"trg_latent shape: {trg_latent.shape}", file=sys.stderr, flush=True)
        print(f"src mean: {src_latent.mean():.4f}, std: {src_latent.std():.4f}", file=sys.stderr, flush=True)
        print(f"trg mean: {trg_latent.mean():.4f}, std: {trg_latent.std():.4f}", file=sys.stderr, flush=True)

    dynamic = DynamicSDE(config=config)

    indices = torch.randperm(batch_size, device=device)
    while (indices == torch.arange(batch_size, device=device)).any():
        indices = torch.randperm(batch_size, device=device)

    trg_latent_neg1 = trg_latent[indices] 
    trg_mask_neg1 = trg_mask[indices]

    if not eval and batch_idx == 0:
        cls_src = src_latent[:, 0, :]
        cos_sim_pos = F.cosine_similarity(cls_src, trg_latent[:, 0, :], dim=-1)
        cos_sim_neg = F.cosine_similarity(cls_src, trg_latent_neg1[:, 0, :], dim=-1)
        print(f"\n=== COSINE SIMILARITY (CLS tokens) ===", file=sys.stderr, flush=True)
        print(f"POSITIVE pairs: {cos_sim_pos.mean():.4f} В± {cos_sim_pos.std():.4f}", file=sys.stderr, flush=True)
        print(f"NEGATIVE1 pairs: {cos_sim_neg.mean():.4f} В± {cos_sim_neg.std():.4f}", file=sys.stderr, flush=True)
        print(f"Difference: {(cos_sim_pos.mean() - cos_sim_neg.mean()):.4f}", file=sys.stderr, flush=True)

    if not eval and batch_idx < 3:
        for i in range(min(3, batch_size)):
            print(f"Pos src: {batch['text_src'][i]}", file=sys.stderr, flush=True)
            print(f"Pos trg: {batch['text_trg'][i]}", file=sys.stderr, flush=True)
            print(f"Neg1 trg (shuffled): {batch['text_trg'][indices[i].item()]}", file=sys.stderr, flush=True)
            print("---", file=sys.stderr, flush=True)

    if device.type == 'cuda':
        t_aug = torch.cuda.FloatTensor(batch_size).uniform_() * 0.4 + 0.3 
    else:
        t_aug = (torch.FloatTensor(batch_size).uniform_() * 0.4 + 0.3).to(device)

    x_t_aug = dynamic.marginal(trg_latent, t_aug)['x_t']

    if not eval and batch_idx == 0:
        print(f"\n=== STEP NEG2: NOISING x_0 ===", file=sys.stderr, flush=True)
        print(f"t_aug range: [{t_aug.min():.4f}, {t_aug.max():.4f}]", file=sys.stderr, flush=True)
        print(f"x_t_aug mean: {x_t_aug.mean():.4f}, std: {x_t_aug.std():.4f}", file=sys.stderr, flush=True)

    with torch.no_grad():
        trg_latent_neg2 = predict_x0_from_xt(
            x_t=x_t_aug,
            t=t_aug,
            score_estimator=score_estimator,
            cond=None,
            cond_mask=None,
        )
    trg_mask_neg2 = trg_mask

    if not eval and batch_idx == 0:
        print(f"\n=== STEP NEG2: DIFFUSION X_0 PREDICTION ===", file=sys.stderr, flush=True)
        print(f"xМ‚_0 mean: {trg_latent_neg2.mean():.4f}, std: {trg_latent_neg2.std():.4f}", file=sys.stderr, flush=True)
        mse = F.mse_loss(trg_latent_neg2, trg_latent)
        cos_sim = F.cosine_similarity(
            trg_latent_neg2.view(batch_size, -1),
            trg_latent.view(batch_size, -1), dim=-1
        ).mean()
        print(f"MSE(xМ‚_0, x_0): {mse:.4f}", file=sys.stderr, flush=True)
        print(f"CosSim(xМ‚_0, x_0): {cos_sim:.4f}", file=sys.stderr, flush=True)

    if eval:
        current_T = dynamic.T
    else:
        warmup_epochs = 10
        if (epoch + 1) < warmup_epochs:
            progress = (epoch + 1) / warmup_epochs
            current_T = dynamic.eps + (dynamic.T - dynamic.eps) * progress
        else:
            current_T = dynamic.T

    if not eval and batch_idx == 0:
        print(f"\n=== CURRICULUM ===", file=sys.stderr, flush=True)
        print(f"Epoch: {epoch}, current_T: {current_T:.4f} (T={dynamic.T:.4f})", file=sys.stderr, flush=True)

    def sample_t_prime(n):
        if device.type == 'cuda':
            return torch.cuda.FloatTensor(n).uniform_() * (current_T - dynamic.eps) + dynamic.eps
        else:
            return (torch.FloatTensor(n).uniform_() * (current_T - dynamic.eps) + dynamic.eps).to(device)

    t_prime_pos = sample_t_prime(batch_size)
    t_prime_neg1 = sample_t_prime(batch_size)
    t_prime_neg2 = sample_t_prime(batch_size)

    noisy_pos = dynamic.marginal(trg_latent, t_prime_pos)['x_t']
    noisy_neg1 = dynamic.marginal(trg_latent_neg1, t_prime_neg1)['x_t']
    noisy_neg2 = dynamic.marginal(trg_latent_neg2, t_prime_neg2)['x_t']

    if batch_idx == 0:
        print(f"\n=== NOISING WITH t' ===", file=sys.stderr, flush=True)
        print(f"t'_pos  range: [{t_prime_pos.min():.4f}, {t_prime_pos.max():.4f}]", file=sys.stderr, flush=True)
        print(f"t'_neg1 range: [{t_prime_neg1.min():.4f}, {t_prime_neg1.max():.4f}]", file=sys.stderr, flush=True)
        print(f"t'_neg2 range: [{t_prime_neg2.min():.4f}, {t_prime_neg2.max():.4f}]", file=sys.stderr, flush=True)
        print(f"noisy_pos  mean: {noisy_pos.mean():.4f}, std: {noisy_pos.std():.4f}", file=sys.stderr, flush=True)
        print(f"noisy_neg1 mean: {noisy_neg1.mean():.4f}, std: {noisy_neg1.std():.4f}", file=sys.stderr, flush=True)
        print(f"noisy_neg2 mean: {noisy_neg2.mean():.4f}, std: {noisy_neg2.std():.4f}", file=sys.stderr, flush=True)

    src_embeds_all = torch.cat([src_latent, src_latent, src_latent], dim=0)
    noisy_trg_all = torch.cat([noisy_pos, noisy_neg1, noisy_neg2], dim=0)
    src_mask_all = torch.cat([src_mask, src_mask, src_mask], dim=0)
    t_all = torch.cat([t_prime_pos, t_prime_neg1, t_prime_neg2], dim=0)
    labels_all = torch.cat([
        torch.ones(batch_size, dtype=torch.float32, device=device), 
        torch.zeros(batch_size, dtype=torch.float32, device=device), 
        torch.zeros(batch_size, dtype=torch.float32, device=device), 
    ], dim=0)

    if not eval and batch_idx == 0:
        print(f"\n=== CLASSIFIER INPUT ===", file=sys.stderr, flush=True)
        print(f"Positive  (noise(x_0,  t')): mean={noisy_pos.mean():.4f},  std={noisy_pos.std():.4f}", file=sys.stderr, flush=True)
        print(f"Negative1 (noise(x_0^-,t')): mean={noisy_neg1.mean():.4f}, std={noisy_neg1.std():.4f}", file=sys.stderr, flush=True)
        print(f"Negative2 (noise(xМ‚_0, t')): mean={noisy_neg2.mean():.4f}, std={noisy_neg2.std():.4f}", file=sys.stderr, flush=True)

    logits = cond_encoder(
        src_embeds=src_embeds_all,
        noisy_trg_embeds=noisy_trg_all,
        src_mask=src_mask_all,
        t=t_all
    )

    loss = F.binary_cross_entropy_with_logits(
        logits.squeeze(-1),
        labels_all,
        pos_weight=torch.tensor(2.0, device=device)
    )

    with torch.no_grad():
        probs = torch.sigmoid(logits.squeeze(-1))
        preds = (probs > 0.5).float()
        acc = (preds == labels_all).float().mean()

    if not eval and batch_idx == 0:
        print(f"\n=== CLASSIFIER OUTPUT ===", file=sys.stderr, flush=True)
        pos_logits = logits[:batch_size]
        neg1_logits = logits[batch_size:2*batch_size]
        neg2_logits = logits[2*batch_size:]
        print(f"Logits - pos: {pos_logits.mean():.4f}, neg1(shuffle): {neg1_logits.mean():.4f}, neg2(diffusion): {neg2_logits.mean():.4f}", file=sys.stderr, flush=True)
        pos_probs = probs[:batch_size]
        neg1_probs = probs[batch_size:2*batch_size]
        neg2_probs = probs[2*batch_size:]
        print(f"Probs  - pos: {pos_probs.mean():.4f}, neg1(shuffle): {neg1_probs.mean():.4f}, neg2(diffusion): {neg2_probs.mean():.4f}", file=sys.stderr, flush=True)
        print(f"Weights mean: n/a (no weighting)", file=sys.stderr, flush=True)
        print(f"Loss: {loss.item():.4f}, Accuracy: {acc.item():.4f}", file=sys.stderr, flush=True)

    return loss, acc


def train(config, encoder, cond_encoder, score_estimator, tokenizer, device):
    print("Starting training...")
    cond_encoder.train()

    trainable_params = sum(p.numel() for p in cond_encoder.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in cond_encoder.parameters())
    print(f"Trainable: {trainable_params}, Total: {total_params}")

    batch_size = config.cond_encoder.batch_size
    print(f"Batch size: {batch_size}")

    train_dataset, valid_dataset = get_datasets(config=config)

    optimizer = torch.optim.AdamW(
        cond_encoder.parameters(),
        lr=config.cond_encoder.lr,
        weight_decay=config.cond_encoder.weight_decay,
        betas=config.cond_encoder.betas,
    )

    train_loader, valid_loader = get_loaders(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        batch_size=batch_size
    )
    print(f"Train loader: {len(train_loader)}, Valid loader: {len(valid_loader)}")

    num_training_steps = len(train_loader) * config.cond_encoder.epochs
    num_warmup_steps = num_training_steps // 10

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    print("\n" + "="*80)
    print("TRAINING LOGIC:")
    print("  POSITIVE  (label=1): src + x_0          в†’ noise(t') в†’ classifier")
    print("  NEGATIVE1 (label=0): src + shuffled x_0 в†’ noise(t') в†’ classifier")
    print("  NEGATIVE2 (label=0): src + xМ‚_0         в†’ noise(t') в†’ classifier")
    print("      РіРґРµ xМ‚_0 = diffusion(noise(x_0, t_aug)), t_aug в€€ [0.3, 0.7]")
    print("  t' вЂ” curriculum: eps в†’ T Р·Р° curriculum_warmup_epochs СЌРїРѕС… (default=11)")
    print("="*80 + "\n")

    step = 0
    for epoch in range(config.cond_encoder.epochs):
        print(f"\n=== EPOCH {epoch + 1}/{config.cond_encoder.epochs} ===")

        cond_encoder.train()
        train_bar = tqdm(train_loader)

        for batch_idx, batch in enumerate(train_bar):
            loss, acc = loss_step(
                epoch=epoch,
                batch=batch,
                tokenizer=tokenizer,
                encoder=encoder,
                cond_encoder=cond_encoder,
                score_estimator=score_estimator,
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

            wandb.log({'train/loss': loss.item(), 'train/accuracy': acc.item()}, step=step)

            train_bar.set_postfix({
                'Epoch': f"{epoch + 1}/{config.cond_encoder.epochs}",
                'Loss': f"{loss.item():.4f}",
                'Acc': f"{acc.item():.4f}",
            })
            step += 1

        cond_encoder.eval()
        total_loss = total_acc = total_num = 0.0

        print(f"\n=== VALIDATION epoch {epoch + 1} (size={len(valid_loader)} batches) ===", file=sys.stderr, flush=True)

        with torch.no_grad():
            valid_bar = tqdm(valid_loader, desc=f"Validation epoch {epoch + 1}")
            for batch_idx, batch in enumerate(valid_bar):

                if batch_idx % 100 == 0:
                    print(f"  valid batch {batch_idx}/{len(valid_loader)}...", file=sys.stderr, flush=True)

                loss, acc = loss_step(
                    epoch=epoch,
                    batch=batch,
                    tokenizer=tokenizer,
                    encoder=encoder,
                    cond_encoder=cond_encoder,
                    score_estimator=score_estimator,
                    config=config,
                    eval=True,
                    device=device,
                    batch_idx=batch_idx
                )
                n = len(batch['text_trg'])
                total_loss += loss.item() * n
                total_acc += acc.item() * n
                total_num += n

                valid_bar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'Acc': f"{acc.item():.4f}",
                })

        total_loss /= total_num
        total_acc /= total_num

        print(f"=== VALIDATION DONE: Loss={total_loss:.4f}, Acc={total_acc:.4f} ===", file=sys.stderr, flush=True)
        wandb.log({'valid/loss': total_loss, 'valid/accuracy': total_acc}, step=step)
        print(f"Validation вЂ” Loss: {total_loss:.4f}, Acc: {total_acc:.4f}")

        save_checkpoint(cond_encoder, config)


def main():
    args = parse()
    # схема негативов задается самим скриптом и попадает в имя чекпоинта
    # классификатора, чтобы три схемы не писали в один файл
    args.augmentation_scheme = "combined"
    config = create_config(args)

    config.cond_encoder.lr = 1e-4
    config.cond_encoder.epochs = 13
    config.cond_encoder.weight_decay = 0.01


    config.model.encoder_link = "bert-base-cased"
    config.decoder.mode = "transformer"
    config.decoder.decoder_path = "datasets/rocstories/3-3-another-decoder-bert-base-cased-80-transformer.pth"
    config.training.checkpoints_folder = "checkpoints"
    config.training.checkpoints_prefix = "actdm-bert-base-cased-512-0.0002-rocstories-cfg=0.0"
    config.training.checkpoint_name = "100000"

    config.is_conditional = True
    config.cond_encoder.use_conditional_encoder = True
    config.cond_encoder.cond_encoder_path = (
        '/home/nklotts/tencdm/datasets/rocstories/'
        'conditional-encoder-bert-base-cased-80-transformer.pth'
    )

    if config.normalize_encodings:
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

    print("="*80)
    print("LOADING DIFFUSION MODEL")
    print("="*80)

    se_config = deepcopy(config.se_config)
    se_config.use_self_cond = config.use_self_cond
    score_estimator = ScoreEstimatorEMB(config=se_config)

    prefix_folder = os.path.join(config.training.checkpoints_folder, config.training.checkpoints_prefix)
    if not os.path.exists(prefix_folder):
        raise FileNotFoundError(f"Checkpoint folder not found: {prefix_folder}")

    checkpoint_names = [
        int(t.replace(".pth", ""))
        for t in os.listdir(prefix_folder)
        if t.replace(".pth", "").isdigit()
    ]
    if not checkpoint_names:
        raise FileNotFoundError(f"No checkpoints in: {prefix_folder}")

    name = config.training.checkpoint_name or max(checkpoint_names)
    checkpoint_name = f"{prefix_folder}/{name}.pth"
    if not os.path.exists(checkpoint_name):
        raise FileNotFoundError(f"Diffusion checkpoint not found: {checkpoint_name}")

    print(f"Loading: {checkpoint_name}")
    load = torch.load(checkpoint_name, map_location='cpu')
    score_estimator.load_state_dict(load["model"])

    from utils.ema_model import ExponentialMovingAverage
    ema = ExponentialMovingAverage(score_estimator.parameters(), config.model.ema_rate)
    ema.load_state_dict(load["ema"])
    ema.store(score_estimator.parameters())
    ema.copy_to(score_estimator.parameters())
    print(f"EMA applied. Step: {load.get('step', 'unknown')}")

    score_estimator.eval()
    for param in score_estimator.parameters():
        param.requires_grad = False
    print("Score estimator frozen.")
    print("="*80 + "\n")

    cond_encoder = ConditionalEncoder(config.model.encoder_link, tokenizer).train()
    cond_encoder_path = config.cond_encoder.cond_encoder_path

    if os.path.exists(cond_encoder_path):
        print(f"Loading existing classifier from: {cond_encoder_path}")
        checkpoint = torch.load(cond_encoder_path, map_location='cpu')
        state = checkpoint["cond_encoder"] if "cond_encoder" in checkpoint else checkpoint
        cond_encoder.load_state_dict(state)
    else:
        print(f"No classifier at {cond_encoder_path}, training from scratch.")

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    device = torch.device('cuda' if num_gpus > 0 else 'cpu')

    encoder = encoder.to(device)
    score_estimator = score_estimator.to(device)
    cond_encoder = cond_encoder.to(device)

    if num_gpus > 1:
        encoder = torch.nn.DataParallel(encoder)
        score_estimator = torch.nn.DataParallel(score_estimator)
        print(f"Training on {num_gpus} GPUs")
    else:
        print(f"Training on {'GPU' if num_gpus == 1 else 'CPU'}")

    wandb.init(
        project=config.project_name,
        name="classifier_3way_curriculum",
        mode="offline"
    )

    train(config, encoder, cond_encoder, score_estimator, tokenizer, device)


if __name__ == '__main__':
    main()
