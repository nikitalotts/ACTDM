
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
    # drop_last обязателен: negative-ы строятся перестановкой внутри батча,
    # и на хвостовом батче из 1 примера подбор перестановки без неподвижных
    # точек зацикливается навсегда
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
    os.makedirs(os.path.dirname(config.cond_encoder.cond_encoder_path), exist_ok=True)
    model.eval()
    torch.save(
        {
            "cond_encoder": model.state_dict(),
            "time_scale": float(config.cond_encoder.time_scale),
        },
        config.cond_encoder.cond_encoder_path
    )
    print(f"Save model to: {config.cond_encoder.cond_encoder_path}")

@torch.no_grad()
def predict_x0_from_xt(x_t, t, score_estimator, cond=None, cond_mask=None, use_autocast=False):
    x_0_self_cond = torch.zeros_like(x_t, dtype=x_t.dtype)

    if use_autocast and x_t.device.type == 'cuda':
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            x_0_pred = score_estimator(
                x_t=x_t,
                time_t=t,
                cond=cond,
                attention_mask=None,
                cond_mask=cond_mask,
                x_0_self_cond=x_0_self_cond
            )
    else:
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

    if not eval and batch_idx == 0:
        print(f"\n=== EMBEDDINGS ===", file=sys.stderr, flush=True)
        print(f"src_latent shape: {src_latent.shape}", file=sys.stderr, flush=True)
        print(f"trg_latent shape: {trg_latent.shape}", file=sys.stderr, flush=True)
        print(f"src mean: {src_latent.mean():.4f}, std: {src_latent.std():.4f}", file=sys.stderr, flush=True)
        print(f"trg mean: {trg_latent.mean():.4f}, std: {trg_latent.std():.4f}", file=sys.stderr, flush=True)

    batch_size = src_latent.shape[0]
    src_mask = src["attention_mask"] 
    trg_mask = trg["attention_mask"] 

    indices = torch.randperm(batch_size, device=device)
    while (indices == torch.arange(batch_size, device=device)).any():
        indices = torch.randperm(batch_size, device=device)

    trg_latent_neg = trg_latent[indices] 
    trg_mask_neg = trg_mask[indices] 

    if not eval and batch_idx == 0:
        cls_src = src_latent[:, 0, :]
        cos_sim_pos = F.cosine_similarity(cls_src, trg_latent[:, 0, :], dim=-1)
        cos_sim_neg = F.cosine_similarity(cls_src, trg_latent_neg[:, 0, :], dim=-1)
        print(f"\n=== COSINE SIMILARITY (CLS tokens) ===", file=sys.stderr, flush=True)
        print(f"POSITIVE pairs: {cos_sim_pos.mean():.4f} В± {cos_sim_pos.std():.4f}", file=sys.stderr, flush=True)
        print(f"NEGATIVE pairs: {cos_sim_neg.mean():.4f} В± {cos_sim_neg.std():.4f}", file=sys.stderr, flush=True)
        print(f"Difference: {(cos_sim_pos.mean() - cos_sim_neg.mean()):.4f}", file=sys.stderr, flush=True)

    if not eval and batch_idx < 3:
        for i in range(min(3, batch_size)):
            print(f"Pos src: {batch['text_src'][i]}", file=sys.stderr, flush=True)
            print(f"Pos trg: {batch['text_trg'][i]}", file=sys.stderr, flush=True)
            print(f"Neg trg: {batch['text_trg'][indices[i].item()]}", file=sys.stderr, flush=True)
            print("---", file=sys.stderr, flush=True)

    trg_embeds_all = torch.cat([trg_latent, trg_latent_neg], dim=0)
    src_embeds_all = torch.cat([src_latent, src_latent], dim=0)
    src_mask_all = torch.cat([src_mask, src_mask], dim=0)
    trg_mask_all = torch.cat([trg_mask, trg_mask_neg], dim=0) 
    labels_all = torch.cat([
        torch.ones(batch_size, dtype=torch.float32, device=device),
        torch.zeros(batch_size, dtype=torch.float32, device=device),
    ], dim=0)

    total_batch_size = trg_embeds_all.shape[0] 

    dynamic = DynamicSDE(config=config)

    eps = dynamic.eps 
    if device.type == 'cuda':
        t_diffusion = torch.cuda.FloatTensor(total_batch_size).uniform_() * (0.5 - eps) + eps
    else:
        t_diffusion = (torch.FloatTensor(total_batch_size).uniform_() * (0.5 - eps) + eps).to(device)

    marg = dynamic.marginal(trg_embeds_all, t_diffusion)
    x_t = marg['x_t']

    if not eval and batch_idx == 0:
        print(f"\n=== STEP 1: DIFFUSION NOISING ===", file=sys.stderr, flush=True)
        print(f"t_diffusion range: [{t_diffusion.min():.4f}, {t_diffusion.max():.4f}]", file=sys.stderr, flush=True)
        print(f"x_t mean: {x_t.mean():.4f}, std: {x_t.std():.4f}", file=sys.stderr, flush=True)

    with torch.no_grad():
        x_0_pred = predict_x0_from_xt(
            x_t=x_t,
            t=t_diffusion,
            score_estimator=score_estimator,
            cond=None,
            cond_mask=None,
            use_autocast=False
        )

    if not eval and batch_idx == 0:
        print(f"\n=== STEP 2: DIFFUSION X_0 PREDICTION ===", file=sys.stderr, flush=True)
        print(f"x_0_pred mean: {x_0_pred.mean():.4f}, std: {x_0_pred.std():.4f}", file=sys.stderr, flush=True)
        mse = F.mse_loss(x_0_pred, trg_embeds_all)
        cos_sim = F.cosine_similarity(
            x_0_pred.view(total_batch_size, -1),
            trg_embeds_all.view(total_batch_size, -1),
            dim=-1
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
        print('current_T', current_T)
        print(f"\n=== STEP 3: CURRICULUM LEARNING ===", file=sys.stderr, flush=True)
        print(f"current_T: {current_T:.4f}", file=sys.stderr, flush=True)

    if device.type == 'cuda':
        t_prime = torch.cuda.FloatTensor(total_batch_size).uniform_() * (current_T - dynamic.eps) + dynamic.eps
    else:
        t_prime = (torch.FloatTensor(total_batch_size).uniform_() * (current_T - dynamic.eps) + dynamic.eps).to(device)

    marg_prime = dynamic.marginal(x_0_pred, t_prime)
    noisy_trg_embeds = marg_prime['x_t']

    if batch_idx == 0:
        print(f"t' range: [{t_prime.min():.4f}, {t_prime.max():.4f}]", file=sys.stderr, flush=True)
        print(f"noisy_trg mean: {noisy_trg_embeds.mean():.4f}, std: {noisy_trg_embeds.std():.4f}", file=sys.stderr, flush=True)

    logits = cond_encoder(
        src_embeds=src_embeds_all,
        noisy_trg_embeds=noisy_trg_embeds,
        src_mask=src_mask_all,
        # trg_mask не передаем: на генерации длина продолжения неизвестна
        # и маска там единичная -- вход классификатора должен совпадать
        t=t_prime
    )

    loss = F.binary_cross_entropy_with_logits(logits.squeeze(-1), labels_all)

    with torch.no_grad():
        probs = torch.sigmoid(logits.squeeze(-1))
        preds = (probs > 0.5).float()
        acc = (preds == labels_all).float().mean()

    if not eval and batch_idx == 0:
        print(f"\n=== CLASSIFIER OUTPUT ===", file=sys.stderr, flush=True)
        print(f"Logits - pos: {logits[:batch_size].mean():.4f}, neg: {logits[batch_size:].mean():.4f}", file=sys.stderr, flush=True)
        print(f"Probs  - pos: {probs[:batch_size].mean():.4f}, neg: {probs[batch_size:].mean():.4f}", file=sys.stderr, flush=True)
        print(f"Loss: {loss.item():.4f}, Accuracy: {acc.item():.4f}", file=sys.stderr, flush=True)

    return loss, acc

def train(config, encoder, cond_encoder, score_estimator, tokenizer, device):
    print("Starting training...")
    print(f"Config.is_conditional: {config.is_conditional}")
    print(f"Config.use_conditional_encoder: {config.cond_encoder.use_conditional_encoder}")

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
    print(f"Train loader length: {len(train_loader)}")
    print(f"Valid loader length: {len(valid_loader)}")

    num_training_steps = len(train_loader) * config.cond_encoder.epochs
    num_warmup_steps = num_training_steps // 10

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    step = 0
    print(f"\nStarting training for {config.cond_encoder.epochs} epochs...")
    print("\n" + "="*80)
    print("TRAINING LOGIC:")
    print("  POSITIVE: x_0  в†’  diffusion(t)  в†’  xМ‚_0  в†’  noise(t')  в†’  classifier = 1")
    print("  NEGATIVE: shuffled x_0^-  в†’  diffusion(t)  в†’  xМ‚_0^-  в†’  noise(t')  в†’  classifier = 0")
    print("  trg_mask РїРµСЂРµРјРµС€РёРІР°РµС‚СЃСЏ РІРјРµСЃС‚Рµ СЃ РЅРµРіР°С‚РёРІР°РјРё")
    print("  t в€€ [eps, 0.5], t' в€€ [eps, T] СЃ curriculum")
    print("="*80 + "\n")

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

            wandb.log({'train loss': loss.item()}, step=step)
            wandb.log({'train accuracy': acc.item()}, step=step)

            train_bar.set_postfix({
                'Stage': 'Training',
                'Epoch': f"{epoch + 1}/{config.cond_encoder.epochs}",
                'Loss': f"{loss.item():.4f}",
                'Acc': f"{acc.item():.4f}",
            })

            step += 1

        print('Starting evaluation')
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
                    score_estimator=score_estimator,
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
                'Loss': f"{total_loss.item():.4f}",
                'Acc': f"{total_acc.item():.4f}",
            })

            print(f"Validation - Loss: {total_loss.item():.4f}, Acc: {total_acc.item():.4f}")

        save_checkpoint(cond_encoder, config)

def main():
    args = parse()
    # схема негативов задается самим скриптом и попадает в имя чекпоинта
    # классификатора, чтобы три схемы не писали в один файл
    args.augmentation_scheme = "augmented"
    # классификатор обучается поверх безусловной диффузии
    args.architecture_type = "unconditional"
    config = create_config(args)

    # диффузия под классификатором безусловная, но самому классификатору
    # нужны пары промпт/продолжение -- иначе препроцессинг выбросит text_src
    config.is_pipeline_conditional = True
    config.cond_encoder.use_conditional_encoder = True

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

    print("\n" + "="*80)
    print("LOADING DIFFUSION MODEL FOR X_0 PREDICTION")
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
        raise FileNotFoundError(f"No checkpoints found in: {prefix_folder}")

    name = config.training.checkpoint_name or max(checkpoint_names)
    checkpoint_name = f"{prefix_folder}/{name}.pth"
    if not os.path.exists(checkpoint_name):
        raise FileNotFoundError(f"Diffusion checkpoint not found: {checkpoint_name}")

    print(f"Loading checkpoint: {checkpoint_name}")
    load = torch.load(checkpoint_name, map_location='cpu')

    score_estimator.load_state_dict(load["model"])
    print("Loaded score_estimator weights")

    from utils.ema_model import ExponentialMovingAverage
    ema = ExponentialMovingAverage(score_estimator.parameters(), config.model.ema_rate)
    ema.load_state_dict(load["ema"])
    ema.store(score_estimator.parameters())
    ema.copy_to(score_estimator.parameters())
    print("Applied EMA weights")

    print(f"Checkpoint step: {load.get('step', 'unknown')}")

    score_estimator.eval()
    for param in score_estimator.parameters():
        param.requires_grad = False
    print("Score estimator frozen")
    print("="*80 + "\n")

    cond_encoder = ConditionalEncoder(
        config.model.encoder_link, tokenizer,
        time_scale=config.cond_encoder.time_scale,
    ).train()

    cond_encoder_path = config.cond_encoder.cond_encoder_path
    if os.path.exists(cond_encoder_path):
        print(f"Loading existing classifier from: {cond_encoder_path}")
        checkpoint = torch.load(cond_encoder_path, map_location='cpu')
        state = checkpoint["cond_encoder"] if "cond_encoder" in checkpoint else checkpoint
        cond_encoder.load_state_dict(state)
        print("Loaded classifier checkpoint")
    else:
        print(f"No classifier found at: {cond_encoder_path}, training from scratch")

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    device = torch.device('cuda' if num_gpus > 0 else 'cpu')

    if num_gpus > 1:
        encoder = torch.nn.DataParallel(encoder).to(device)
        score_estimator = torch.nn.DataParallel(score_estimator).to(device)
        cond_encoder = cond_encoder.to(device)
        print(f'Training on {num_gpus} GPUs')
    else:
        encoder = encoder.to(device)
        score_estimator = score_estimator.to(device)
        cond_encoder = cond_encoder.to(device)
        print(f'Training on {"GPU" if num_gpus == 1 else "CPU"}')

    wandb.init(
        project=config.project_name,
        name="classifier_guidance_with_diffusion_augmentation",
        mode="offline"
    )

    print(config, end="\n\n\n")
    train(config, encoder, cond_encoder, score_estimator, tokenizer, device)

if __name__ == '__main__':
    main()
