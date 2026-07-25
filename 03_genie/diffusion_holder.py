import os
import time

import wandb
import random
import numpy as np
import torch
import ml_collections
import torch.distributed as dist
from torch.utils.data import DataLoader
from ml_collections import ConfigDict
from typing import Optional, Union, Dict, Tuple
from tqdm.auto import trange
from tqdm import tqdm
from transformers import AutoTokenizer
from functools import partial
from timm.scheduler.cosine_lr import CosineLRScheduler
from collections import defaultdict
from torch.cuda.amp import GradScaler
import json
from copy import deepcopy
import heapq

from diffusion_utils.dynamic import DynamicSDE
from diffusion_utils.solvers import create_solver

from utils.ema_model import ExponentialMovingAverage
from utils.util import mse_loss, get_stat, reduce_tensor, set_seed
from data.dataset import DatasetDDP, get_dataset_iter
from data.util import tokenize, BatchEncoding

from model.score_estimator import ScoreEstimatorEMB
from model.encoder import Encoder
from model.enc_normalizer import EncNormalizer
from model.decoder import Decoder

from estimation_utils.util import gather_texts, compute_metric
from estimation_utils.metrics import compute_metric


class DiffusionRunner:
    def __init__(
            self,
            config: ConfigDict,
            eval: bool = False
    ):
        self.config = config

        self.printed_example_estimate = False
        self.printed_initial_examples = False

        gen_cfg = config.model.encoder_link
        self.tokenizer = AutoTokenizer.from_pretrained(gen_cfg)
        if not config.emb:
            self.gen_enc_normalizer = EncNormalizer(
                enc_mean_path=self.config.data.enc_gen_mean,
                enc_std_path=self.config.data.enc_gen_std,
            )
        else:
            self.gen_enc_normalizer = None
        
        self.encoder = Encoder(
            gen_cfg,
            enc_normalizer=self.gen_enc_normalizer,
            is_change_sp_tokens=True,
            emb=config.emb
        ).eval().cuda()

        self.decoder = Decoder(
            decoder_config=config.decoder,
            diffusion_config=config.se_config
        )
        self.restore_decoder()
        self.decoder = self.decoder.cuda().eval()
        
        self.se_config = deepcopy(config.se_config)
        self.se_config.use_self_cond = config.use_self_cond
        self.score_estimator = ScoreEstimatorEMB(
            config=self.se_config
        ).cuda()

        self.ddp_score_estimator = self.score_estimator
        if self.config.ddp:
            self.ddp_score_estimator = torch.nn.parallel.DistributedDataParallel(
                self.score_estimator,
                device_ids=[config.local_rank],
                broadcast_buffers=False,
            )

        self.config.params_number = ml_collections.ConfigDict()
        self.config.params_number.score_estimator = sum(p.numel() for p in self.score_estimator.parameters() if p.requires_grad)
        self.config.params_number.decoder = sum(p.numel() for p in self.decoder.parameters())
        self.config.params_number.generative_encoder = sum(p.numel() for p in self.encoder.parameters())

        self.device = next(self.score_estimator.parameters()).device

        self.dynamic = DynamicSDE(config=config)
        self.diff_eq_solver = create_solver(config)(
            dynamic=self.dynamic,
            score_fn=partial(self.calc_score, model=self.ddp_score_estimator),
            ode_sampling=config.training.ode_sampling
        )
        
        self.train_datasets_iter = DatasetDDP(
            split="train",
            config=config,
        ).get_data()
        self.train_dataset = None
        
        try:
            self.valid_dataset = next(DatasetDDP(split="validation", config=config).get_data())
        except Exception:
            self.valid_dataset = next(DatasetDDP(split="test", config=config).get_data())

        self.ema = ExponentialMovingAverage(self.score_estimator.parameters(), self.config.model.ema_rate)

        self.all_checkpoints = []
        self.tracked_test_metric = dict() 
        
        if self.config.ddp and dist.get_rank() == 0:
            wandb.init(
                project=self.config.project_name,
                name=self.config.training.checkpoints_prefix,
                config=dict(self.config),
                mode="offline"
            )
        
        if eval:
            self.restore_parameters(self.device)
            self.score_estimator.eval()
            num_seeds = int(getattr(self.config, "num_seeds", 1) or 1)
            if num_seeds > 1:
                self.run_statistical_eval("test", num_seeds=num_seeds)
            else:
                self.estimate("test")
        else:
            self.set_optimizer()
            self.set_scheduler()
            self.set_grad_scaler()
            self.step = 0
            
            if self.load_checkpoint():
                if self.config.is_conditional:
                    self.estimate("validation")
                self.estimate("test")
                self.validate()

    def restore_parameters(self, device: Optional[torch.device] = None) -> None:
        prefix_folder = os.path.join(self.config.training.checkpoints_folder, self.config.training.checkpoints_prefix)
        
        checkpoint_names = list(os.listdir(prefix_folder))
        checkpoint_names = [str(t).replace(".pth", "") for t in checkpoint_names]
        checkpoint_names = [int(t) for t in checkpoint_names if t.isdigit()]

        print('CHEKCNAMES', checkpoint_names)

        if not checkpoint_names:
            return False
            
        name = self.config.training.checkpoint_name
        if not name:
            name = max(checkpoint_names)
        checkpoint_name = f"{prefix_folder}/{name}.pth"
        load = torch.load(checkpoint_name)
        self.step = load["step"]
        print('LOADED Diffusion from', checkpoint_name, self.step)
        self.ema.load_state_dict(load["ema"])
        self.switch_to_ema()

        if "encoder" in load:
            self.encoder.load_state_dict(load["encoder"])
            print("Encoder loaded from checkpoint")
        if "decoder" in load:
            self.decoder.load_state_dict(load["decoder"])
            print("Decoder loaded from checkpoint")
        
    def save_checkpoint(self, last: bool = False) -> None:
        if not dist.get_rank() == 0:
            return

        if not os.path.exists(self.config.training.checkpoints_folder):
            os.makedirs(self.config.training.checkpoints_folder)
            
        prefix_folder = os.path.join(self.config.training.checkpoints_folder, self.config.training.checkpoints_prefix)
        if not os.path.exists(prefix_folder):
            os.makedirs(prefix_folder)

        if last:
            prefix = 'last'
        else:
            prefix = str(self.step)

        save_path = os.path.join(prefix_folder, prefix + ".pth")

        if self.config.higher_better:
            item = (self.tracked_test_metric[self.step], save_path)
        else:
            item = (-self.tracked_test_metric[self.step], save_path)

        if self.config.save_top_k is None or self.config.save_top_k > len(self.all_checkpoints):
            self.__save_checkpoint(save_path)
            heapq.heappush(self.all_checkpoints, item)
        else:
            heap_smallest = self.all_checkpoints[0]
            if heap_smallest[0] < item[0]:
                self.__remove_checkpoint(heap_smallest[1])
                heapq.heappop(self.all_checkpoints)
                self.__save_checkpoint(item[1])
                heapq.heappush(self.all_checkpoints, item)

    def __remove_checkpoint(self, save_path):
        os.remove(save_path)
    
    def __save_checkpoint(self, save_path):

        checkpoint = {
                "model": self.score_estimator.state_dict(),
                "ema": self.ema.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "scaler": self.grad_scaler.state_dict(),
                "step": self.step,
                "encoder": self.encoder.state_dict(),
                "decoder": self.decoder.state_dict(),
                "config": self.config,
                "random_states": {
                        "python_random_state": random.getstate(),
                        "numpy_random_state": np.random.get_state(),
                        "torch_random_state": torch.get_rng_state(),
                        "cuda_random_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                        "seed": self.config.seed,
                },
                "tracked_metrics": {
                    "test_metrics": self.tracked_test_metric,
                    "all_checkpoints": [(score, path) for score, path in self.all_checkpoints],
                },
                "train_range_progress": {
                    "current_step": self.step,
                    "training_iters": self.config.training.training_iters,
                },
        }

        torch.save(checkpoint, save_path)

        print(f"Save model to: {save_path}")

        
    def load_checkpoint(self) -> int:
        prefix_folder = os.path.join(self.config.training.checkpoints_folder, self.config.training.checkpoints_prefix)

        if not os.path.exists(prefix_folder):
            return False

        checkpoint_names = list(os.listdir(prefix_folder))
        checkpoint_names = [str(t).replace(".pth", "") for t in checkpoint_names]
        checkpoint_names = [int(t) for t in checkpoint_names if t.isdigit()]

        if not checkpoint_names:
            return False
            
        name = self.config.training.checkpoint_name
        if not name:
            name = max(checkpoint_names)
        checkpoint_name = f"{prefix_folder}/{name}.pth"

        load = torch.load(checkpoint_name, map_location="cpu")

        self.ema.load_state_dict(load["ema"])
        self.ema.cuda()
        self.score_estimator.load_state_dict(load["model"])
        self.optimizer.load_state_dict(load["optimizer"])
        self.scheduler.load_state_dict(load["scheduler"])
        self.grad_scaler.load_state_dict(load["scaler"])
        
        self.step = load["step"]
        if dist.get_rank() == 0:
            print(f"Checkpoint is loaded {checkpoint_name}")
        return True

    def restore_decoder(self):
        decoder_path = self.config.decoder.decoder_path
        checkpoint = torch.load(decoder_path)
        print('DECINCH', "decoder" in checkpoint, checkpoint.keys())
        self.decoder.load_state_dict(checkpoint["decoder"])

    def switch_to_ema(self) -> None:
        ema = self.ema
        score_model = self.score_estimator
        ema.store(score_model.parameters())
        ema.copy_to(score_model.parameters())

    def switch_back_from_ema(self) -> None:
        ema = self.ema
        score_model = self.score_estimator
        ema.restore(score_model.parameters())

    def set_optimizer(self) -> None:
        optimizer = torch.optim.AdamW(
            self.score_estimator.parameters(),
            lr=self.config.optim.lr,
            weight_decay=self.config.optim.weight_decay,
            betas=(self.config.optim.beta_1, self.config.optim.beta_2),
            eps=self.config.optim.eps,
        )
        self.warmup = self.config.optim.linear_warmup
        self.grad_clip_norm = self.config.optim.grad_clip_norm
        self.optimizer = optimizer

    def set_scheduler(self) -> None:
        self.scheduler = CosineLRScheduler(
            self.optimizer,
            t_initial=self.config.training.training_iters,
            lr_min=self.config.optim.min_lr,
            warmup_lr_init=self.config.optim.warmup_lr,
            warmup_t=self.config.optim.linear_warmup,
            cycle_limit=1,
            t_in_epochs=False,
        )
        
    def set_grad_scaler(self) -> None:
        self.grad_scaler = GradScaler()

    def collate_fn(self, batch):
        # diffuseq конкатенирует латенты промпта и продолжения в одну последовательность,
        # поэтому длины должны быть фиксированными, а не подгоняться под самый длинный текст батча
        padding = "max_length" if self.config.architecture_type == "diffuseq" else True

        texts_trg = [t["text_trg"] for t in batch]
        tok_trg = self.tokenizer(
            texts_trg,
            add_special_tokens=True,
            padding=padding,
            truncation=True,
            max_length=self.config.data.max_sequence_len,
            return_tensors="pt",
            return_attention_mask=True,
            return_token_type_ids=False,
        )
        
        if self.config.is_conditional:
            texts_src = [t["text_src"] for t in batch]
            tok_src = self.tokenizer(
                texts_src,
                add_special_tokens=True,
                padding=padding,
                truncation=True,
                max_length=self.config.data.max_context_len,
                return_tensors="pt",
                return_attention_mask=True,
                return_token_type_ids=False,
            )
            
            new_batch = {
                "text_src": texts_src,
                "input_ids_src": tok_src["input_ids"],
                "attention_mask_src": tok_src["attention_mask"],
                "text_trg": texts_trg,
                "input_ids_trg": tok_trg["input_ids"],
                "attention_mask_trg": tok_trg["attention_mask"],
            }
            if "references" in batch[0]:
                new_batch["text_references"] = [t["references"] for t in batch]

            new_batch = BatchEncoding(new_batch)
        else:
            new_batch = BatchEncoding({
                "text_trg": texts_trg,
                "input_ids_trg": tok_trg["input_ids"],
                "attention_mask_trg": tok_trg["attention_mask"],
            })
        return new_batch

    def set_train_data_generator(self) -> None:
        del self.train_dataset
        self.train_dataset = next(self.train_datasets_iter)
        print("Dataset length:", len(self.train_dataset))

        self.train_loader = DataLoader(
            self.train_dataset,
            num_workers=30,
            batch_size=self.config.training.batch_size_per_gpu,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def set_valid_data_generator(self) -> None:
        self.valid_loader = DataLoader(
            self.valid_dataset,
            num_workers=20,
            batch_size=self.config.validation.batch_size,
            collate_fn=self.collate_fn,
            shuffle=False,
        )

    def log_metric(self, metric_name: str, loader_name: str, value: Union[float, torch.Tensor, wandb.Image]):
        if dist.is_initialized() and dist.get_rank() == 0:
            wandb.log({f'{metric_name}/{loader_name}': value}, step=self.step)

    def optimizer_step(self, loss: torch.Tensor):
        self.optimizer.zero_grad()
        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.unscale_(self.optimizer)
        
        grad_norm = torch.sqrt(
            sum([torch.sum(t.grad ** 2) for t in self.score_estimator.parameters() if t.requires_grad])
        )

        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.score_estimator.parameters(),
                max_norm=self.grad_clip_norm
            )

        self.log_metric('lr', 'train', self.optimizer.param_groups[0]['lr'])
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

        scale = self.grad_scaler._scale.item()
        max_scale = 2 ** 30
        min_scale = 1
        scale = np.clip(scale, min_scale, max_scale)
        self.grad_scaler.update(new_scale=scale)

        self.ema.update(self.score_estimator.parameters())
        self.scheduler.step_update(self.step)
        return grad_norm

    def sample_time(self, batch_size: int, eps: float = 1e-5):
        return torch.cuda.FloatTensor(batch_size).uniform_() * (self.dynamic.T - eps) + eps
    
    def train(self) -> None:
        self.set_valid_data_generator()

        self.train_range = trange(self.step + 1, self.config.training.training_iters + 1)
        self.train_range_iter = iter(self.train_range)

        while True:
            self.set_train_data_generator()

            if dist.is_initialized() and dist.get_rank() == 0 and not self.printed_initial_examples:
                print("\n" + "=" * 80)
                print("INITIAL TRAINING EXAMPLES (полные тексты)")
                print("=" * 80)

                for _, batch in enumerate(self.train_loader):
                    print("\nПервые 3 примера из тренировочного набора:")
                    print("-" * 80)

                    for i in range(min(3, len(batch["text_trg"]))):
                        print(f"\nПример {i + 1}:")
                        if self.config.is_conditional and "text_src" in batch:
                            src_text = batch["text_src"][i]
                            print(f"SOURCE (полный текст):")
                            print(src_text)
                            print(f"\nTARGET (полный текст):")
                            print(batch["text_trg"][i])
                        else:
                            print(f"TARGET (полный текст):")
                            print(batch["text_trg"][i])
                        print("-" * 80)

                    break 

                print("=" * 80 + "\n")
                self.printed_initial_examples = True

            self.ddp_score_estimator.train()
            self.train_epoch()

            if self.step >= self.config.training.training_iters:
                break

        self.score_estimator.eval()
        self.save_checkpoint(last=True)
        self.switch_to_ema()

    def train_epoch(self):
        for _, batch in enumerate(self.train_loader):

            if self.step >= self.config.training.training_iters:
                return
            _ = next(self.train_range_iter)

            loss_dict, stat_dict = self.train_step(batch)

            if self.step % self.config.training.eval_freq == 0:
                total_start = time.time()
                print('#INFO enter self.step % self.config.training.eval_freq == 0:')
                if self.config.is_conditional:
                    print('#INFO enter self.config.is_conditional: ')
                    val_start = time.time()
                    self.estimate("validation")
                    val_time = time.time() - val_start
                    print(f'#INFO validation estimation time: {val_time:.2f} seconds')

                test_start = time.time()
                self.estimate("test")
                test_time = time.time() - test_start
                print(f'#INFO test estimation time: {test_time:.2f} seconds')

                validate_start = time.time()
                self.validate()
                validate_time = time.time() - validate_start
                print(f'#INFO validation time: {validate_time:.2f} seconds')

                total_time = time.time() - total_start
                print(f'#INFO finished estimation, total time: {total_time:.2f} seconds')
            
            if self.step % self.config.training.checkpoint_freq == 0:
                self.save_checkpoint()

            if self.step % self.config.training.accum_batch_steps == 0:
                self.train_range.set_description(
                    f"loss_x_0: {loss_dict['loss_x_0'].item():0.4f}, "
                    f"grad_norm: {stat_dict['grad_norm'].item():0.4f}, "
                )

    def train_step(self, batch):
        self.step += 1

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16), torch.no_grad():
            batch = batch.to(f"cuda:{dist.get_rank()}")

            if self.config.is_conditional:
                src_x = self.encoder(**{
                    "input_ids": batch["input_ids_src"],
                    "attention_mask": batch["attention_mask_src"]
                })
            else:
                src_x = None

            trg_x = self.encoder(**{
                "input_ids": batch["input_ids_trg"], 
                "attention_mask": batch["attention_mask_trg"]
            })

        loss_dict, stat_dict = self.calc_loss(clean_x=trg_x, cond_x=src_x, batch=batch)
        
        if self.step % self.config.training.accum_batch_steps == 0:
            stat_dict["grad_norm"] = self.optimizer_step(loss_dict['total_loss'])
            stat_dict["scale_factor"] = torch.Tensor([self.grad_scaler._scale])

        if self.step % 10 == 0:
            stat_dict["weight_norm"] = torch.sqrt(
                sum([torch.sum(t.data ** 2) for t in self.score_estimator.parameters()]))

            for k, v in loss_dict.items():
                self.log_metric(k, 'train', v.item())

            for k, v in stat_dict.items():
                self.log_metric("statistics", k, v.item())

        return loss_dict, stat_dict

    @torch.no_grad()
    def validate(self) -> None:
        self.set_valid_data_generator()
        prev_mode = self.ddp_score_estimator.training

        self.ddp_score_estimator.eval()
        self.switch_to_ema()

        valid_loss: Dict[str, torch.Tensor] = dict()
        valid_count = torch.Tensor([0.0])

        for batch in self.valid_loader:
            batch = batch.to(f"cuda:{dist.get_rank()}")
            if self.config.is_conditional:
                src_x = self.encoder(**{
                    "input_ids": batch["input_ids_src"],
                    "attention_mask": batch["attention_mask_src"]
                })
            else:
                src_x = None

            trg_x = self.encoder(**{
                "input_ids": batch["input_ids_trg"], 
                "attention_mask": batch["attention_mask_trg"]
            })
            
            loss_dict, _ = self.calc_loss(clean_x=trg_x, cond_x=src_x, batch=batch)
            for k, v in loss_dict.items():
                if k in valid_loss:
                    valid_loss[k] += v.item() * trg_x.size(0)
                else:
                    valid_loss[k] = torch.Tensor([v.item() * trg_x.size(0)])
            valid_count += trg_x.size(0)

        valid_count = reduce_tensor(valid_count.cuda())
        for k, v in valid_loss.items():
            valid_loss[k] = reduce_tensor(valid_loss[k].cuda())

        for k, v in valid_loss.items():
            valid_loss[k] = v / valid_count
        for k, v in valid_loss.items():
            self.log_metric(k, 'valid_loader', v)

        self.switch_back_from_ema()
        self.ddp_score_estimator.train(prev_mode)

    def predict_x_0_unconditional(
        self,
        model,
        x_t, t,
        attention_mask=None,
        x_0_self_cond=None
    ) -> torch.Tensor:
        texts_src = ["" for _ in range(x_t.shape[0])]
        tok_src = self.tokenizer(
            texts_src,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=self.config.data.max_context_len,
            return_tensors="pt",
            return_attention_mask=True,
            return_token_type_ids=False,
        ).to(f"cuda:{dist.get_rank()}")
        src_x = self.encoder(
            input_ids=tok_src["input_ids"],
            attention_mask=tok_src["attention_mask"]
        )

        x_0 = model(
            x_t=x_t, 
            time_t=t, 
            cond=src_x,
            attention_mask=attention_mask, 
            cond_mask=tok_src["attention_mask"],
            x_0_self_cond=x_0_self_cond
        )
        return x_0
        
    def calc_score(
            self,
            model,
            x_t, t,
            cond=None,
            attention_mask=None,
            cond_mask=None,
            x_0_self_cond=None
    ) -> Dict[str, torch.Tensor]:
        params = self.dynamic.marginal_params(t)
        x_0 = model(
            x_t=x_t, time_t=t, cond=cond,
            attention_mask=attention_mask, cond_mask=cond_mask,
            x_0_self_cond=x_0_self_cond
        )
        
        if not model.training and self.config.validation.cfg_coef and self.config.is_conditional:
            x_0_null = self.predict_x_0_unconditional(model, x_t=x_t, t=t, attention_mask=attention_mask, x_0_self_cond=x_0_self_cond)
            x_0 = x_0 + self.config.validation.cfg_coef * (x_0 - x_0_null)
        
        eps_theta = (x_t - params["mu"] * x_0) / params["std"]
        score = -eps_theta / params["std"]
        return {
            "score": score,
            "x_0": x_0,
            "eps_theta": eps_theta
        }

    def build_score_estimator_input(self, x_t, cond_x, attention_mask, cond_mask, trg_mask):
        """Собирает вход denoising network в зависимости от config.architecture_type.

        genie    -- условие подается в каждый блок сети через cross-attention,
                    поэтому x_t идет как есть, а cond_x/cond_mask -- отдельными аргументами.
        diffuseq -- условие подается через latent replacement: латенты промпта
                    конкатенируются с зашумленным продолжением в одну последовательность,
                    cross-attention не используется.

        Возвращает (src_len, z_t, attention_mask, cond, cond_mask).
        """
        if self.config.architecture_type == "diffuseq" and self.config.is_conditional and cond_x is not None:
            src_len = cond_x.shape[1]
            z_t = torch.cat([cond_x, x_t], dim=1)
            if cond_mask is not None:
                if trg_mask is None:
                    trg_mask = torch.ones(
                        x_t.shape[0], x_t.shape[1],
                        device=cond_mask.device, dtype=cond_mask.dtype,
                    )
                combined_mask = torch.cat([cond_mask, trg_mask], dim=1)
            else:
                combined_mask = None
            return src_len, z_t, combined_mask, None, None

        return 0, x_t, attention_mask, cond_x, cond_mask

    def calc_loss(
            self,
            clean_x,
            cond_x,
            batch=None,
            eps: float = 1e-5,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        mask = None

        batch_size = clean_x.size(0)

        t = self.sample_time(batch_size, eps=eps)
        marg_forward = self.dynamic.marginal(clean_x, t)
        x_t, noise = marg_forward['x_t'], marg_forward['noise']

        src_len, z_t, se_mask, se_cond, se_cond_mask = self.build_score_estimator_input(
            x_t=x_t,
            cond_x=cond_x,
            attention_mask=mask,
            cond_mask=batch.get("attention_mask_src"),
            trg_mask=batch.get("attention_mask_trg"),
        )

        x_0_self_cond = torch.zeros_like(z_t, dtype=z_t.dtype)
        if self.config.use_self_cond and random.random() > 0.5:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                with torch.no_grad():
                    x_0_self_cond = self.ddp_score_estimator(
                        x_t=z_t, time_t=t, cond=se_cond,
                        attention_mask=se_mask,
                        cond_mask=se_cond_mask,
                        x_0_self_cond=x_0_self_cond
                    ).detach()
                    if src_len > 0:
                        x_0_self_cond[:, :src_len, :] = cond_x

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            z_0 = self.ddp_score_estimator(
                x_t=z_t, time_t=t, cond=se_cond,
                attention_mask=se_mask,
                cond_mask=se_cond_mask,
                x_0_self_cond=x_0_self_cond
            )

        # для diffuseq первые src_len позиций -- это латенты промпта, лосс считается
        # только по продолжению; для genie src_len == 0 и срез ничего не меняет
        x_0 = z_0[:, src_len:, :]

        loss_x_0 = mse_loss(clean_x, x_0, mask)

        loss_dict = {
            'total_loss': loss_x_0,
            'loss_x_0': loss_x_0,
        }

        with torch.no_grad():
            stat_dict = {}
            clean_x_dict = get_stat(clean_x, mask)
            for key in clean_x_dict:
                stat_dict[f"clean_x_{key}"] = clean_x_dict[key]
    
            x_0_dict = get_stat(x_0.detach(), mask)
            for key in x_0_dict:
                stat_dict[f"x_0_{key}"] = x_0_dict[key]
    
            mask = batch["attention_mask_trg"]
            clean_x_dict_SPT = get_stat(clean_x, mask)
            for key in clean_x_dict_SPT:
                stat_dict[f"clean_x_woSPT_{key}"] = clean_x_dict_SPT[key]
    
            x_0_dict_SPT = get_stat(x_0, mask)
            for key in x_0_dict_SPT:
                stat_dict[f"x_0_woSPT_{key}"] = x_0_dict_SPT[key]

        return loss_dict, stat_dict

    @torch.no_grad()
    def generate_text_conditional(self, dataset_name: str, split: str):
        dt = next(get_dataset_iter(self.config, dataset_name, split=split))
        loader = DataLoader(
            dt,
            num_workers=20,
            batch_size=self.config.validation.batch_size,
            collate_fn=self.collate_fn,
        )

        result_dict = {
            "GEN": [],
            "TRG": []
        }
        if self.config.is_conditional:
            result_dict["SRC"] = []

        print('Loader size:', len(loader))
        gen_total_time = 0.0
        gen_total_count = 0
        for batch in loader:
            if dist.is_initialized():
                batch = batch.to(f"cuda:{dist.get_rank()}")
            else:
                batch = batch.to(f"cuda:0")

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _t0 = time.time()

            if self.config.is_conditional:
                src_x = self.encoder(**{
                    "input_ids": batch["input_ids_src"],
                    "attention_mask": batch["attention_mask_src"]
                })
            else:
                src_x = None
           
            gen_text = self.generate_text_batch(
                batch_size=len(batch["text_trg"]),
                cond_x=src_x,
                attention_mask=None,
                cond_mask=batch.get("attention_mask_src"),
            )[0]

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            gen_total_time += time.time() - _t0
            gen_total_count += int(len(batch["text_trg"]))

            if not self.printed_example_estimate and dist.get_rank() == 0:
                for i in range(10):
                    print(f'EXAMPLE #{i+1}')
                    if self.config.is_conditional and "text_src" in batch:
                        print(f"#DEBUG Source: {batch['text_src'][i]}")
                    print(f"#DEBUG Target : {batch['text_trg'][i]}")

                    if gen_text:
                        print(f"#DEBUG Generated: {gen_text[i]}")
                self.printed_example_estimate = True

            if dataset_name not in self.config.data.datasets.downstream_tasks:
                result_dict["TRG"] += self.tokenizer.batch_decode(batch["input_ids_trg"], skip_special_tokens=True)
            else:
                if "text_references" in batch:
                    result_dict["TRG"] += batch["text_references"]
                else:
                    result_dict["TRG"] += batch["text_trg"]
                
            result_dict["GEN"] += gen_text
            if self.config.is_conditional:
                result_dict["SRC"] += batch["text_src"]

            if len(result_dict["TRG"]) >= (self.config.validation.num_gen_texts // dist.get_world_size()):
                break

        self._last_gen_time = gen_total_time
        self._last_gen_count = gen_total_count

        return result_dict

    @torch.no_grad()
    def generate_text_batch(self, batch_size, cond_x=None, attention_mask=None, cond_mask=None):
        if attention_mask is not None:
            attention_mask = attention_mask.cuda()

        pred_embeddings = self.pred_embeddings(
            batch_size=batch_size,
            attention_mask=attention_mask,
            cond_x=cond_x,
            cond_mask=cond_mask,
        )

        output = self.pred_logits(pred_embeddings, cond_x=cond_x, cond_mask=cond_mask)
        tokens = output.argmax(dim=-1)

        end_tokens = []
        if hasattr(self.tokenizer, 'eos_token') and self.tokenizer.eos_token is not None:
            end_tokens.append(self.tokenizer.vocab[self.tokenizer.eos_token])
        if hasattr(self.tokenizer, 'sep_token') and self.tokenizer.sep_token is not None:
            end_tokens.append(self.tokenizer.vocab[self.tokenizer.sep_token])

        tokens = tokens.detach().cpu().tolist()
        tokens_list = []
        for seq in tokens:
            id = 0
            while id < len(seq) and seq[id] not in end_tokens:
                id += 1
            tokens_list.append(seq[0: id])

        text = self.tokenizer.batch_decode(tokens_list, skip_special_tokens=True)
        return text, pred_embeddings

    @torch.no_grad()
    def pred_logits(self, pred_embeddings, cond_x=None, cond_mask=None):
        if not self.config.emb:
            pred_embeddings = self.gen_enc_normalizer.denormalize(pred_embeddings)
            if self.config.decoder.is_conditional and cond_x is not None:
                cond_x = self.gen_enc_normalizer.denormalize(cond_x)
        else:
            cond_x = None
            cond_mask = None
        output = self.decoder(pred_embeddings, cond_x=cond_x, cond_mask=cond_mask)
        return output

    @torch.no_grad()
    def pred_embeddings(
            self,
            batch_size,
            cond_x=None,
            cond_mask=None,
            attention_mask=None,
    ) -> torch.Tensor:
        self.score_estimator.eval()
        shape = (
            batch_size,
            self.config.data.max_sequence_len,
            self.encoder.encoder.config.hidden_size
        )

        with torch.no_grad():
            x = self.dynamic.prior_sampling(shape).to(self.device)

            src_len, z, se_mask, se_cond, se_cond_mask = self.build_score_estimator_input(
                x_t=x,
                cond_x=cond_x,
                attention_mask=attention_mask,
                cond_mask=cond_mask,
                trg_mask=None,
            )

            x_0_self_cond = torch.zeros_like(z, dtype=z.dtype)
            eps_t = 0.01

            if self.config.timesteps == "linear":
                timesteps = torch.linspace(self.dynamic.T, eps_t, self.dynamic.N + 1, device=self.device)
            elif self.config.timesteps == "quad":
                deg = 2
                timesteps = torch.linspace(1, 0, self.dynamic.N + 1, device=self.device) ** deg * (self.dynamic.T - eps_t) + eps_t

            for idx in tqdm(range(self.dynamic.N)):
                t = timesteps[idx]
                next_t = timesteps[idx + 1]

                input_t = t * torch.ones(shape[0], device=self.device)
                next_input_t = next_t * torch.ones(shape[0], device=self.device)

                output = self.diff_eq_solver.step(
                    x_t=z, t=input_t, next_t=next_input_t,
                    cond=se_cond,
                    cond_mask=se_cond_mask,
                    attention_mask=se_mask,
                    x_0_self_cond=x_0_self_cond,
                )

                z, z_mean = output["x"], output["x_mean"]
                x_0_self_cond = output["x_0"]

                # diffuseq: латенты промпта фиксируются на каждом шаге обратного процесса
                if src_len > 0:
                    z[:, :src_len, :] = cond_x
                    z_mean[:, :src_len, :] = cond_x
                    x_0_self_cond[:, :src_len, :] = cond_x

            pred_embeddings = z_mean[:, src_len:, :]

        return pred_embeddings

    @torch.no_grad()
    def estimate(self, split: str):
        self.score_estimator.eval()
        self.ddp_score_estimator.eval()
        self.switch_to_ema()
        
        result_dict = dict()
        timing_per_dataset = dict()
        
        for dataset_name in self.config.data.datasets.datasets_list:
            if dist.is_initialized():
                seed = self.config.seed + self.step + dist.get_rank()
            else:
                seed = self.config.seed + self.step
            set_seed(seed)
            result_dict[dataset_name] = self.generate_text_conditional(dataset_name, split=split)

            local_time = float(getattr(self, "_last_gen_time", 0.0))
            local_cnt = int(getattr(self, "_last_gen_count", 0))
            if dist.is_initialized():
                t = torch.tensor(local_time, device=self.device)
                c = torch.tensor(local_cnt, device=self.device, dtype=torch.float32)
                t = reduce_tensor(t) * dist.get_world_size()
                c = reduce_tensor(c) * dist.get_world_size()
                total_time = float(t.item())
                total_cnt = int(c.item())
            else:
                total_time = local_time
                total_cnt = local_cnt
            per_ex_ms = (total_time / total_cnt * 1000.0) if total_cnt > 0 else 0.0
            timing_per_dataset[dataset_name] = {
                "total_sec": total_time,
                "per_example_ms": per_ex_ms,
                "n_examples": total_cnt,
            }
            ws = dist.get_world_size() if dist.is_initialized() else 1
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"\n[GEN TIMING] {dataset_name}: total {total_time:.2f}s "
                      f"for {total_cnt} examples → {per_ex_ms:.2f} ms/example "
                      f"(world_size={ws})")
                self.log_metric("gen_time", "total_sec", total_time)
                self.log_metric("gen_time", "per_example_ms", per_ex_ms)

            if not dist.is_initialized() or dist.get_rank() == 0:
                print("\n" + "=" * 80)
                print(f"GENERATED EXAMPLES - Step {self.step} - {split.upper()} (полные тексты)")
                print("=" * 80)

                for dataset_name in result_dict:
                    print(f"\n{'=' * 40}")
                    print(f"Dataset: {dataset_name}")
                    print('=' * 40)

                    keys = list(result_dict[dataset_name].keys())
                    for i in range(min(10, len(result_dict[dataset_name][keys[0]]))):
                        print(f"\nПример {i + 1}:")

                        if "SRC" in result_dict[dataset_name]:
                            src_text = result_dict[dataset_name]["SRC"][i]
                            print(f"\nSOURCE:")
                            print(src_text)

                        trg_text = result_dict[dataset_name]["TRG"][i]
                        print(f"\nTARGET:")
                        print(trg_text)

                        gen_text = result_dict[dataset_name]["GEN"][i]
                        print(f"\nGENERATED:")
                        print(gen_text)

                        print("-" * 40)

                print("=" * 80 + "\n")

        if dist.is_initialized():
            for dataset_name in result_dict:
                for key in result_dict[dataset_name]:
                    result_dict[dataset_name][key] = gather_texts(result_dict[dataset_name][key])
                if dataset_name not in self.config.data.datasets.downstream_tasks:
                    for key in result_dict[dataset_name]:
                        result_dict[dataset_name][key] = result_dict[dataset_name][key][:self.config.validation.num_gen_texts]

        result_list = dict()

        for dataset_name in result_dict:
            keys = list(result_dict[dataset_name].keys())
            result_list[dataset_name] = []
            for ind in range(len(result_dict[dataset_name][keys[0]])):
                result_list[dataset_name].append(
                    {key: result_dict[dataset_name][key][ind] for key in keys}
                )
                
        if not dist.is_initialized() or dist.get_rank() == 0:
            if not os.path.exists(self.config.validation.texts_path):
                os.makedirs(self.config.validation.texts_path)

            prefix_folder = os.path.join(self.config.validation.texts_path, self.config.training.checkpoints_prefix)
            if not os.path.exists(prefix_folder):
                os.makedirs(prefix_folder)

            file_name = f"{self.step}-N={self.config.dynamic.N}-seed={self.config.seed}-len={len(result_list)}.json"
            save_path = os.path.join(prefix_folder, file_name)
            json.dump(result_list, open(save_path, "w"), indent=4)
            print(f"Texts are saved in {save_path}")

        metrics_dict = dict()
        for dataset_name in self.config.data.datasets.datasets_list:
            texts_src = result_dict[dataset_name].get("SRC")
            texts_trg = result_dict[dataset_name]["TRG"]
            texts_gen = result_dict[dataset_name]["GEN"]

            metrics_dict[dataset_name] = dict()

            for metric_name in self.config.data.datasets.metrics[dataset_name]["metrics"]:
                metrics_dict[dataset_name][metric_name] = compute_metric(
                    metric_name, 
                    predictions=texts_gen, 
                    references=texts_trg, 
                    sources=texts_src
                )

        if not dist.is_initialized() or dist.get_rank() == 0:
            for dataset_name in self.config.data.datasets.datasets_list:
                print("-----", f"{dataset_name}-{split}", "-----")
                for metric_name in self.config.data.datasets.metrics[dataset_name]["metrics"]:
                    value = metrics_dict[dataset_name][metric_name]
                    if isinstance(value, dict):
                        for key in value:
                            print(f"{key}: {value[key]:0.5f}")
                            self.log_metric(metric_name=f"{dataset_name}-{split}", loader_name=key, value=value[key])
                    else:
                        print(f"{metric_name}: {value:0.5f}")
                        self.log_metric(metric_name=f"{dataset_name}-{split}", loader_name=metric_name, value=value)

        if split == "test":
            self.tracked_test_metric[self.step] = metrics_dict[self.config.tracked_dataset][self.config.tracked_metric]
        
        if dist.is_initialized():
            seed = self.config.seed + self.step + dist.get_rank()
        else:
            seed = self.config.seed + self.step
        set_seed(seed)

        self.switch_back_from_ema()
        self.ddp_score_estimator.train()
        self.score_estimator.train()

        metrics_dict["__timing__"] = timing_per_dataset

        return metrics_dict

    @torch.no_grad()
    def run_statistical_eval(self, split: str, num_seeds: int):
        base_seed = int(getattr(self.config, "seed", 0))
        seed_step = int(getattr(self.config, "seed_step", 1000))
        rank = dist.get_rank() if dist.is_initialized() else 0

        if rank == 0:
            print("\n" + "#" * 80)
            print(f"# STATISTICAL EVALUATION (DIFFUSION)")
            print(f"# num_seeds = {num_seeds}")
            print(f"# base_seed = {base_seed}")
            print(f"# seed_step = {seed_step}")
            print(f"# N (steps) = {self.config.dynamic.N}")
            print("#" * 80 + "\n")

        all_runs = []
        for i in range(num_seeds):
            run_seed = base_seed + i * seed_step
            if rank == 0:
                print("\n" + "=" * 80)
                print(f"=== RUN {i + 1}/{num_seeds}   (base seed = {run_seed}) ===")
                print("=" * 80 + "\n")

            self.config.seed = run_seed
            set_seed(run_seed + rank)

            metrics = self.estimate(split)
            all_runs.append(metrics)

        self.config.seed = base_seed

        if rank == 0:
            self._aggregate_and_save(all_runs, split, num_seeds, base_seed)

    def _aggregate_and_save(self, all_runs, split, num_seeds, base_seed):
        import numpy as _np

        t_crit_95 = {
            1: float("inf"), 2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776,
            6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262,
            15: 2.131, 20: 2.086, 25: 2.060, 30: 2.042,
        }

        def t_crit(n):
            df = n - 1
            if df <= 0:
                return float("inf")
            if df in t_crit_95:
                return t_crit_95[df]
            if df > 30:
                return 1.96
            keys = sorted(k for k in t_crit_95.keys() if k <= df)
            return t_crit_95[keys[-1]] if keys else float("inf")

        def stats_from_values(values):
            arr = _np.array(values, dtype=float)
            n = len(arr)
            mean = float(arr.mean())
            std = float(arr.std(ddof=1)) if n > 1 else 0.0
            stderr = std / _np.sqrt(n) if n > 1 else 0.0
            ci_half = t_crit(n) * stderr if n > 1 else 0.0
            return {
                "mean": mean,
                "std": std,
                "stderr": float(stderr),
                "ci95_half_width": float(ci_half),
                "n": n,
                "values": [float(x) for x in arr],
            }

        aggregated = {}
        for dataset_name in self.config.data.datasets.datasets_list:
            aggregated[dataset_name] = {}
            for metric_name in self.config.data.datasets.metrics[dataset_name]["metrics"]:
                per_run = [run[dataset_name][metric_name] for run in all_runs]
                is_dict = isinstance(per_run[0], dict)
                if is_dict:
                    sub_keys = list(per_run[0].keys())
                    aggregated[dataset_name][metric_name] = {}
                    for sk in sub_keys:
                        aggregated[dataset_name][metric_name][sk] = \
                            stats_from_values([v[sk] for v in per_run])
                else:
                    aggregated[dataset_name][metric_name] = stats_from_values(per_run)

        timing_aggregated = {}
        if all("__timing__" in run for run in all_runs):
            ds_names_with_timing = list(all_runs[0]["__timing__"].keys())
            for ds in ds_names_with_timing:
                total_vals = [run["__timing__"][ds]["total_sec"] for run in all_runs]
                per_ex_vals = [run["__timing__"][ds]["per_example_ms"] for run in all_runs]
                n_ex_vals = [run["__timing__"][ds]["n_examples"] for run in all_runs]
                timing_aggregated[ds] = {
                    "total_sec": stats_from_values(total_vals),
                    "per_example_ms": stats_from_values(per_ex_vals),
                    "n_examples": int(n_ex_vals[0]),
                }

        print("\n" + "=" * 80)
        print(f"STATISTICAL EVAL SUMMARY  —  {num_seeds} runs, base_seed={base_seed}")
        print(f"  checkpoint : {self.config.training.checkpoints_prefix}"
              f" / {self.config.training.checkpoint_name}")
        print(f"  split      : {split}")
        print(f"  N (steps)  : {self.config.dynamic.N}")
        print("=" * 80)
        for dataset_name in aggregated:
            print(f"\n[{dataset_name}]")
            print("-" * 80)
            for metric_name, val in aggregated[dataset_name].items():
                if isinstance(val, dict) and "mean" in val:
                    vals_fmt = ", ".join(f"{v:.5f}" for v in val["values"])
                    print(f"  {metric_name:15s}: "
                          f"{val['mean']:.5f} ± {val['std']:.5f}  "
                          f"(95% CI ±{val['ci95_half_width']:.5f}, n={val['n']})")
                    print(f"  {'':15s}  values: [{vals_fmt}]")
                else:
                    for sk, ss in val.items():
                        vals_fmt = ", ".join(f"{v:.5f}" for v in ss["values"])
                        print(f"  {metric_name}/{sk:10s}: "
                              f"{ss['mean']:.5f} ± {ss['std']:.5f}  "
                              f"(95% CI ±{ss['ci95_half_width']:.5f}, n={ss['n']})")
                        print(f"  {'':15s}  values: [{vals_fmt}]")
        if timing_aggregated:
            print("\n[__timing__]")
            print("-" * 80)
            for ds, t in timing_aggregated.items():
                tt, pe, ne = t["total_sec"], t["per_example_ms"], t["n_examples"]
                print(f"  {ds}: n_examples={ne}")
                print(f"    total_sec      : {tt['mean']:.3f} ± {tt['std']:.3f}  "
                      f"(95% CI ±{tt['ci95_half_width']:.3f}, n={tt['n']})")
                print(f"    per_example_ms : {pe['mean']:.3f} ± {pe['std']:.3f}  "
                      f"(95% CI ±{pe['ci95_half_width']:.3f}, n={pe['n']})")
        print("=" * 80 + "\n")

        if not os.path.exists(self.config.validation.texts_path):
            os.makedirs(self.config.validation.texts_path)
        prefix_folder = os.path.join(self.config.validation.texts_path,
                                     self.config.training.checkpoints_prefix)
        if not os.path.exists(prefix_folder):
            os.makedirs(prefix_folder)

        payload = {
            "checkpoint_prefix": self.config.training.checkpoints_prefix,
            "checkpoint_name": str(self.config.training.checkpoint_name),
            "split": split,
            "num_seeds": num_seeds,
            "base_seed": base_seed,
            "seed_step": int(getattr(self.config, "seed_step", 1000)),
            "N_steps": int(self.config.dynamic.N),
            "num_gen_texts": int(self.config.validation.num_gen_texts),
            "step": self.step,
            "metrics": aggregated,
            "timing": timing_aggregated,
        }
        file_name = (f"statistical_eval-{split}-num_seeds={num_seeds}"
                     f"-base_seed={base_seed}-step={self.step}-N={self.config.dynamic.N}.json")
        save_path = os.path.join(prefix_folder, file_name)
        with open(save_path, "w") as f:
            json.dump(payload, f, indent=4)
        print(f"Aggregated statistical metrics saved to:\n  {save_path}\n")
