# import os
# import torch
# import wandb
# from create_config import create_config
#
# import os
# import torch
# import wandb
# import numpy as np
# import argparse
# from tqdm import tqdm
# from torch.nn.functional import cross_entropy
# from torch.utils.data import DataLoader
# from transformers import AutoTokenizer
# import torch.nn.functional as F
#
# from utils.util import dict_to_cuda
# from model.decoder import BertDecoder
# from data.dataset import get_dataset_iter
# from model.encoder import Encoder
# from create_config import create_config
# from model.enc_normalizer import EncNormalizer
# from diffusion_utils.dynamic import DynamicSDE
# from utils.util import parse
# from model.conditional_encoder import ConditionalEncoder
#
#
# def get_loaders(train_dataset, valid_dataset, batch_size):
#     train_loader = DataLoader(
#         next(train_dataset),
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=30,
#         pin_memory=False,
#     )
#
#     valid_loader = DataLoader(
#         next(valid_dataset),
#         batch_size=batch_size,
#         num_workers=30,
#         pin_memory=False,
#     )
#
#     return train_loader, valid_loader
#
#
# def get_datasets(config):
#     train_dataset = get_dataset_iter(
#         config,
#         dataset_name=config.cond_encoder.dataset,
#         split="train",
#         task='train_coniditonal_encoder'
#     )
#     test_dataset = get_dataset_iter(
#         config,
#         dataset_name=config.cond_encoder.dataset,
#         split="test",
#         task='train_coniditonal_encoder'
#     )
#     return train_dataset, test_dataset
#
#
# def save_checkpoint(model, config):
#     os.makedirs(config.training.checkpoints_folder, exist_ok=True)
#
#     model.eval()
#     torch.save(
#         {
#             "cond_encoder": model.state_dict(),
#         },
#         config.cond_encoder.cond_encoder_path
#     )
#     print(f"Save model to: {config.cond_encoder.cond_encoder_path}")
#
#
# def get_empty_embedding(encoder, tokenizer, config, device):
#     empty_tokens = tokenizer(
#         [""],
#         add_special_tokens=True,
#         padding='max_length',
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
#                     input_ids=empty_tokens["input_ids"],
#                     attention_mask=empty_tokens["attention_mask"]
#                 )
#         else:
#             empty_embeds = encoder(
#                 input_ids=empty_tokens["input_ids"],
#                 attention_mask=empty_tokens["attention_mask"]
#             )
#
#     empty_mask = empty_tokens["attention_mask"]
#
#     return empty_embeds, empty_mask
#
#
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
#     with torch.no_grad():
#         if device.type == 'cuda':
#             with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
#                 src_embeds = encoder(
#                     input_ids=src["input_ids"],
#                     attention_mask=src["attention_mask"]
#                 )
#
#                 trg_embeds = encoder(
#                     input_ids=trg["input_ids"],
#                     attention_mask=trg["attention_mask"]
#                 )
#         else:
#             src_embeds = encoder(
#                 input_ids=src["input_ids"],
#                 attention_mask=src["attention_mask"]
#             )
#
#             trg_embeds = encoder(
#                 input_ids=trg["input_ids"],
#                 attention_mask=trg["attention_mask"]
#             )
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
#     indices = torch.roll(torch.arange(batch_size, device=device), shifts=1)
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
#     if not eval:
#         print(f"Batch size: {batch_size}")
#         print(f"Pos labels: {labels_pos.sum()}, Neg labels: {labels_neg.sum()}")
#         print(f"Total: pos={torch.sum(labels_all == 1)}, neg={torch.sum(labels_all == 0)}")
#
#         dynamic = DynamicSDE(config=config)
#         if device.type == 'cuda':
#             t = torch.cuda.FloatTensor(total_batch_size).uniform_() * (
#                     config.cond_encoder.T - config.cond_encoder.eps) + config.cond_encoder.eps
#         else:
#             t = torch.FloatTensor(total_batch_size).uniform_() * (
#                     config.cond_encoder.T - config.cond_encoder.eps) + config.cond_encoder.eps
#             t = t.to(device)
#         marg_forward = dynamic.marginal(trg_embeds_all, t)
#         noisy_trg_embeds = marg_forward['x_t']
#     else:
#         t = torch.ones(total_batch_size, device=device) * config.cond_encoder.eps
#         noisy_trg_embeds = trg_embeds_all
#
#     with torch.no_grad():
#         if not config.emb:
#             if hasattr(encoder, 'module'):
#                 src_embeds_all = encoder.module.enc_normalizer.denormalize(src_embeds_all)
#                 noisy_trg_embeds = encoder.module.enc_normalizer.denormalize(noisy_trg_embeds)
#             else:
#                 src_embeds_all = encoder.enc_normalizer.denormalize(src_embeds_all)
#                 noisy_trg_embeds = encoder.enc_normalizer.denormalize(noisy_trg_embeds)
#
#     combined_embeds = torch.cat([src_embeds_all, noisy_trg_embeds], dim=1)
#     combined_mask = torch.cat([src_mask_all, trg_mask_all], dim=1)
#
#     if device.type == 'cuda':
#         with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
#             logits = cond_encoder(combined_embeds, combined_mask, t)
#     else:
#         logits = cond_encoder(combined_embeds, combined_mask, t)
#
#     loss = F.cross_entropy(logits, labels_all)
#
#     preds = torch.argmax(logits, dim=1)
#     acc = torch.mean((preds == labels_all).float())
#
#     if not eval:
#         print(f"Pred distribution: 0={torch.sum(preds == 0).item()}, 1={torch.sum(preds == 1).item()}")
#         print(f"Label distribution: 0={torch.sum(labels_all == 0).item()}, 1={torch.sum(labels_all == 1).item()}")
#
#     return loss, acc
#
#
# def train(config, encoder, cond_encoder, tokenizer, device):
#     total_number_params = sum(p.numel() for p in cond_encoder.parameters() if p.requires_grad)
#     print(f"Num params: {total_number_params}")
#
#     trainable_params = sum(p.numel() for p in cond_encoder.parameters() if p.requires_grad)
#     total_params = sum(p.numel() for p in cond_encoder.parameters())
#     print(f"Trainable: {trainable_params}, Total: {total_params}")
#
#     batch_size = config.cond_encoder.batch_size
#
#     empty_embeds, empty_mask = get_empty_embedding(encoder, tokenizer, config, device)
#
#     train_dataset, valid_dataset = get_datasets(config=config)
#
#     optimizer = torch.optim.AdamW(
#         cond_encoder.parameters(),
#         lr=config.cond_encoder.lr,
#         weight_decay=config.cond_encoder.weight_decay,
#         betas=config.cond_encoder.betas,
#     )
#
#     step = 0
#     for epoch in range(config.cond_encoder.epochs):
#         train_loader, valid_loader = get_loaders(
#             train_dataset=train_dataset,
#             valid_dataset=valid_dataset,
#             batch_size=batch_size
#         )
#
#         cond_encoder.train()
#         train_bar = tqdm(train_loader)
#         for batch in train_bar:
#             loss, acc = loss_step(
#                 batch=batch,
#                 tokenizer=tokenizer,
#                 encoder=encoder,
#                 cond_encoder=cond_encoder,
#                 config=config,
#                 device=device,
#                 empty_embeds=empty_embeds,
#                 empty_mask=empty_mask
#             )
#
#             optimizer.zero_grad()
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(
#                 cond_encoder.parameters(),
#                 max_norm=config.cond_encoder.max_norm
#             )
#             optimizer.step()
#
#             wandb.log({'train loss': loss.item()}, step=step)
#             wandb.log({'train accuracy': acc.item()}, step=step)
#
#             train_bar.set_postfix({
#                 'Stage': 'Training',
#                 'Epoch': f"{epoch + 1}/{config.cond_encoder.epochs}",
#                 'Loss': loss.item()
#             })
#
#             step += 1
#
#         cond_encoder.eval()
#         with torch.no_grad():
#             total_loss = 0.
#             total_acc = 0.
#             total_num = 0.
#
#             valid_bar = tqdm(valid_loader)
#             for batch in valid_bar:
#                 loss, acc = loss_step(
#                     batch=batch,
#                     tokenizer=tokenizer,
#                     encoder=encoder,
#                     cond_encoder=cond_encoder,
#                     config=config,
#                     eval=True,
#                     device=device,
#                     empty_embeds=empty_embeds,
#                     empty_mask=empty_mask
#                 )
#                 batch_size_cur = len(batch['text_trg'])
#                 total_loss += loss * batch_size_cur
#                 total_acc += acc * batch_size_cur
#                 total_num += batch_size_cur
#
#             total_loss /= total_num
#             total_acc /= total_num
#
#             wandb.log({'valid loss': total_loss.item()}, step=step)
#             wandb.log({'valid accuracy': total_acc.item()}, step=step)
#
#             valid_bar.set_postfix({
#                 'Stage': 'Validation',
#                 'Epoch': f"{epoch + 1}/{config.cond_encoder.epochs}",
#                 'Loss': loss.item()
#             })
#
#         save_checkpoint(cond_encoder, config)
#
#
# def main():
#     args = parse()
#     config = create_config(args)
#     if not config.emb:
#         enc_normalizer = EncNormalizer(
#             enc_mean_path=config.data.enc_gen_mean,
#             enc_std_path=config.data.enc_gen_std,
#         )
#     else:
#         enc_normalizer = None
#     encoder = Encoder(
#         config.model.encoder_link,
#         enc_normalizer=enc_normalizer,
#         is_change_sp_tokens=True,
#         emb=config.emb
#     ).eval()
#     tokenizer = AutoTokenizer.from_pretrained(encoder.encoder_link)
#     cond_encoder = ConditionalEncoder(encoder.encoder_link).train()
#
#     num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
#     if num_gpus > 0:
#         device = torch.device('cuda')
#     else:
#         device = torch.device('cpu')
#
#     if num_gpus > 1:
#         encoder = torch.nn.DataParallel(encoder).to(device)
#         cond_encoder = cond_encoder.to(device)
#         print(f'Training on {num_gpus} GPUs')
#     elif num_gpus == 1:
#         encoder = encoder.to(device)
#         cond_encoder = cond_encoder.to(device)
#         print(f'Training on {num_gpus} GPU')
#     else:
#         encoder = encoder.to(device)
#         cond_encoder = cond_encoder.to(device)
#         print('Training on CPU')
#
#     print('Training Conditional BERT NSP on rocstories')
#     wandb.init(project=config.project_name, name="conditional_bert_nsp", mode="offline")
#
#     train(config, encoder, cond_encoder, tokenizer, device)
#
#
# if __name__ == '__main__':
#     main()
