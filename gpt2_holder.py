import os
import contextlib
import time
import json
import random
import numpy as np
import torch
import torch.distributed as dist
import wandb
import heapq

from torch.utils.data import DataLoader
from tqdm.auto import trange
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from torch.cuda.amp import GradScaler
from timm.scheduler.cosine_lr import CosineLRScheduler
from typing import Optional, Union, Dict

from data.dataset import DatasetDDP, get_dataset_iter
from data.util import BatchEncoding, available_cpus
from estimation_utils.metrics import compute_metric
from estimation_utils.util import gather_texts
from utils.util import set_seed, reduce_tensor, gpu_stats, resume_checkpoint_path



# Сколько позиций считать за раз в чанкованном лоссе. 512 позиций x 50257 слов
# -- это 103 МБ логитов в fp32 плюс столько же на log_softmax, порядка 0.2 ГБ
# пика независимо от батча. Больше брать нельзя: чанк, превышающий число позиций
# в лоссе, означает, что чанкования нет вовсе.
LM_LOSS_CHUNK_TOKENS = 512


def _chunk_loss(hidden, labels, lm_head):
    logits = lm_head(hidden)
    return torch.nn.functional.cross_entropy(
        logits.float(), labels, reduction="sum")


def chunked_lm_loss(hidden, lm_head, labels, chunk_tokens=LM_LOSS_CHUNK_TOKENS):
    """Кросс-энтропия без материализации полного тензора логитов.

    GPT2LMHeadModel считает логиты сразу для всей последовательности:
    [батч x 128 позиций x 50257 слов]. При батче 32 это 0.8 ГБ в fp32, плюс
    столько же на копию внутри cross_entropy (log_softmax сохраняется для
    backward), плюс bf16-копия -- около 2 ГБ, и все это растет линейно с батчем.

    Здесь два сокращения:
      * логиты считаются ТОЛЬКО для позиций, попадающих в лосс. Промпт
        помечен -100 и раньше прогонялся через выходной слой впустую -- это
        половина позиций;
      * оставшиеся позиции обрабатываются чанками под checkpoint: в памяти
        живут логиты одного чанка, остальные пересчитываются на backward.

    Лосс идентичен исходному: сумма по токенам, деленная на их число.
    """
    hidden = hidden[:, :-1, :].reshape(-1, hidden.size(-1))
    labels = labels[:, 1:].reshape(-1)

    keep = labels != -100
    hidden = hidden[keep]
    labels = labels[keep]

    total = labels.numel()
    if total == 0:
        return hidden.sum() * 0.0

    loss = None
    for i in range(0, total, chunk_tokens):
        part = torch.utils.checkpoint.checkpoint(
            _chunk_loss, hidden[i:i + chunk_tokens], labels[i:i + chunk_tokens],
            lm_head, use_reentrant=False)
        loss = part if loss is None else loss + part
    return loss / total


class GPT2WithChunkedLoss(torch.nn.Module):
    """Обертка, считающая лосс внутри forward.

    Нужна из-за двух ограничений сразу. GPT2LMHeadModel материализует логиты
    всегда, даже когда labels не переданы, поэтому звать его нельзя -- надо
    идти в .transformer напрямую. Но вызывать .transformer в обход DDP тоже
    нельзя: DDP вешает хуки на forward обернутого модуля, и без его вызова
    градиенты по картам не синхронизируются. Поэтому оборачиваем в DDP вот
    этот модуль: он ходит в transformer и сам считает чанкованный лосс.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, position_ids, labels):
        hidden = self.model.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        ).last_hidden_state
        return chunked_lm_loss(hidden, self.model.lm_head, labels)


def _loader_workers() -> int:
    """Воркеров загрузчика -- по числу выделенных ядер, но не больше 8.

    Раньше стояло 30 у train-лоадера и 20 у valid при 12-20 ядрах на
    задание. Лишние воркеры не ускоряют чтение, а конкурируют за ядра
    с препроцессингом и грузят диск.
    """
    return max(1, min(available_cpus() // 2, 8))


def is_ddp() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_ddp() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_ddp() else 1



def _should_resume(config) -> bool:
    """Короткий проверочный прогон обязан начинаться с нуля.

    Иначе получается тихая ловушка: smoke сохраняет чекпоинт на шаге 200, а при
    следующем запуске training_iters тоже 200, load_checkpoint поднимает step=200,
    и цикл обучения выходит, не сделав ни шага. В логе при этом все выглядит
    успешно -- отрабатывает eval на СТАРЫХ весах и печатаются метрики прошлого
    прогона. Именно так и произошло с заданием 4267591.
    """
    return os.environ.get("SMOKE", "0") != "1"


class GPT2Runner:
    def __init__(self, config, eval: bool = False):
        self.config = config

        self.printed_example_estimate = False
        self.printed_initial_examples = False

        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        gpt2_config = GPT2Config.from_pretrained("gpt2-medium")
        gpt2_config.bos_token_id = self.tokenizer.bos_token_id
        gpt2_config.eos_token_id = self.tokenizer.eos_token_id
        self.device = torch.device(f"cuda:{config.local_rank}" if torch.cuda.is_available() else "cpu")
        self.model = GPT2LMHeadModel(gpt2_config).to(self.device)
        print(f"[GPT2] Model initialized from SCRATCH (random weights)")
        print(f"[GPT2] Params: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

        # Пересчет активаций в блоках трансформера вместо их хранения. Именно
        # активации 24 блоков -- основной расход памяти (около 20 ГБ из 24.7 при
        # батче 64), а не выходной слой. Без этого 128 примеров на карту не
        # влезают: нужно 0.18 ГБ на пример, а без пересчета выходит 0.31.
        # Платим примерно 30% скорости, получаем возможность набирать батч 512
        # за один шаг, без накопления.
        if getattr(self.config.training, "gradient_checkpointing", True):
            try:
                self.model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:      # transformers постарше не знает kwargs
                self.model.gradient_checkpointing_enable()
            self.model.config.use_cache = False   # с пересчетом кэш не нужен
            print("[GPT2] gradient checkpointing включен")

        # DDP оборачивает не саму GPT2LMHeadModel, а обертку с чанкованным
        # лоссом -- иначе логиты материализуются на всю последовательность
        self.loss_model = GPT2WithChunkedLoss(self.model).to(self.device)
        self.ddp_model = self.loss_model
        if self.config.ddp and torch.cuda.is_available():
            self.ddp_model = torch.nn.parallel.DistributedDataParallel(
                self.loss_model,
                device_ids=[config.local_rank],
                broadcast_buffers=False,
                # градиенты живут прямо в буферах обмена, без второй копии:
                # для 355M параметров это экономит около 1.3 ГБ
                gradient_as_bucket_view=True,
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

        self.all_checkpoints = []
        self.tracked_test_metric = dict()

        if get_rank() == 0:
            wandb.init(
                project=self.config.project_name,
                name=self.config.training.checkpoints_prefix,
                config=dict(self.config),
                mode="offline"
            )

        if eval:
            self.restore_parameters(self.device)
            self.model.eval()
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

            if _should_resume(self.config) and self.load_checkpoint():
                self.estimate("validation")
                self.estimate("test")
                self.validate()

    def collate_fn(self, batch):
        texts_trg = [t["text_trg"] for t in batch]
        texts_src = [t["text_src"] for t in batch]

        max_src_len = self.config.data.max_context_len
        max_trg_len = self.config.data.max_sequence_len
        max_total_len = max_src_len + max_trg_len
        pad_id = self.tokenizer.pad_token_id

        all_input_ids = []
        all_attention_mask = []
        all_labels = []

        for src_text, trg_text in zip(texts_src, texts_trg):
            src_ids = self.tokenizer.encode(
                src_text, add_special_tokens=False
            )[:max_src_len]
            trg_ids = self.tokenizer.encode(
                trg_text, add_special_tokens=False
            )[:max_trg_len]

            ids = src_ids + trg_ids
            labels = [-100] * len(src_ids) + trg_ids

            if len(ids) > max_total_len:
                trg_ids = trg_ids[:max_total_len - len(src_ids)]
                ids = src_ids + trg_ids
                labels = [-100] * len(src_ids) + trg_ids

            pad_len = max_total_len - len(ids)
            attn = [0] * pad_len + [1] * len(ids)
            ids = [pad_id] * pad_len + ids
            labels = [-100] * pad_len + labels

            all_input_ids.append(ids)
            all_attention_mask.append(attn)
            all_labels.append(labels)

        attention_mask = torch.tensor(all_attention_mask, dtype=torch.long)
        # При left padding GPT2 без явных position_ids нумерует позиции с нуля
        # от края батча, то есть реальные токены получают позиции со сдвигом на
        # длину паддинга. generate() же строит позиции из attention_mask
        # (cumsum - 1), где первый реальный токен получает позицию 0.
        # Считаем на обучении так же, иначе позиции train/inference расходятся.
        position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp(min=0)

        new_batch = BatchEncoding({
            "text_src": texts_src,
            "text_trg": texts_trg,
            "input_ids": torch.tensor(all_input_ids, dtype=torch.long),
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "labels": torch.tensor(all_labels, dtype=torch.long),
        })
        return new_batch

    def set_train_data_generator(self):
        del self.train_dataset
        self.train_dataset = next(self.train_datasets_iter)
        print("Dataset length:", len(self.train_dataset))

        self.train_loader = DataLoader(
            self.train_dataset,
            num_workers=_loader_workers(),
            batch_size=self.config.training.batch_size_per_gpu,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def set_valid_data_generator(self):
        self.valid_loader = DataLoader(
            self.valid_dataset,
            num_workers=_loader_workers(),
            batch_size=self.config.validation.batch_size,
            collate_fn=self.collate_fn,
            shuffle=False,
        )

    def set_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.optim.lr,
            weight_decay=self.config.optim.weight_decay,
            betas=(self.config.optim.beta_1, self.config.optim.beta_2),
            eps=self.config.optim.eps,
        )
        self.warmup = self.config.optim.linear_warmup
        self.grad_clip_norm = self.config.optim.grad_clip_norm

    def set_scheduler(self):
        self.scheduler = CosineLRScheduler(
            self.optimizer,
            # тикает шедулер оптимизаторными шагами (step_update ниже делит
            # self.step на accum), поэтому и длина цикла -- в них же.
            # training_iters считает МИКРОшаги, их в accum раз больше
            t_initial=self.config.training.training_iters // self.config.training.accum_batch_steps,
            lr_min=self.config.optim.min_lr,
            warmup_lr_init=self.config.optim.warmup_lr,
            warmup_t=self.config.optim.linear_warmup,
            cycle_limit=1,
            t_in_epochs=False,
        )

    def set_grad_scaler(self):
        self.grad_scaler = GradScaler() if torch.cuda.is_available() else None

    def log_metric(self, metric_name, loader_name, value):
        if get_rank() == 0:
            wandb.log({f'{metric_name}/{loader_name}': value}, step=self.step)

    def optimizer_step(self):
        if self.grad_scaler is not None:
            self.grad_scaler.unscale_(self.optimizer)

        grad_norm = torch.sqrt(
            sum([torch.sum(t.grad ** 2) for t in self.model.parameters()
                 if t.requires_grad and t.grad is not None])
        )

        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)

        self.log_metric('lr', 'train', self.optimizer.param_groups[0]['lr'])

        if self.grad_scaler is not None:
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            self.optimizer.step()

        self.optimizer.zero_grad()
        self.scheduler.step_update(self.step // self.config.training.accum_batch_steps)
        return grad_norm

    def train(self):
        self.set_valid_data_generator()
        # Прогон, поднятый из чекпоинта, у которого шаг уже равен бюджету,
        # не сделает ни одного шага обучения, но отработает eval на старых
        # весах и напишет метрики в лог -- выглядит как успешный прогон.
        # Кричим об этом явно.
        if self.step >= self.config.training.training_iters:
            print(f"[WARNING] шаг {self.step} >= бюджета "
                  f"{self.config.training.training_iters}: обучения НЕ БУДЕТ, "
                  f"чекпоинт уже доучен. Удалите каталог чекпоинтов или "
                  f"увеличьте training_iters.")

        self.train_range = trange(self.step + 1, self.config.training.training_iters + 1)
        self.train_range_iter = iter(self.train_range)

        while True:
            self.set_train_data_generator()

            if get_rank() == 0 and not self.printed_initial_examples:
                print("\n" + "=" * 80)
                print("INITIAL TRAINING EXAMPLES (полные тексты)")
                print("=" * 80)
                for _, batch in enumerate(self.train_loader):
                    print("\nПервые 3 примера из тренировочного набора:")
                    print("-" * 80)
                    for i in range(min(3, len(batch["text_trg"]))):
                        print(f"\nПример {i + 1}:")
                        print(f"SOURCE: {batch['text_src'][i]}")
                        print(f"TARGET: {batch['text_trg'][i]}")
                        print("-" * 80)
                    break
                print("=" * 80 + "\n")
                self.printed_initial_examples = True

            self.ddp_model.train()
            self.train_epoch()

            if self.step >= self.config.training.training_iters:
                break

        self.model.eval()
        self.save_checkpoint(last=True)

    def train_epoch(self):
        for _, batch in enumerate(self.train_loader):
            if self.step >= self.config.training.training_iters:
                return
            _ = next(self.train_range_iter)

            loss_dict, stat_dict = self.train_step(batch)

            # Чекпоинт для дозапуска пишется ДО eval, а не после. На eval идет
            # генерация батчами по validation.batch_size -- самая прожорливая
            # по памяти часть прогона, и падение там (например, по OOM) до
            # сохранения стоило бы всех шагов с прошлого чекпоинта. Нумерованный
            # чекпоинт по-прежнему пишется после eval: ему нужна метрика,
            # которая появляется только там.
            if self.step % self.config.training.checkpoint_freq == 0:
                self.save_checkpoint(last=True)

            if self.step % self.config.training.eval_freq == 0:
                total_start = time.time()
                print('#INFO enter eval step')
                val_start = time.time()
                self.estimate("validation")
                print(f'#INFO validation estimation time: {time.time() - val_start:.2f} seconds')
                test_start = time.time()
                self.estimate("test")
                print(f'#INFO test estimation time: {time.time() - test_start:.2f} seconds')
                validate_start = time.time()
                self.validate()
                print(f'#INFO validation time: {time.time() - validate_start:.2f} seconds')
                print(f'#INFO finished estimation, total time: {time.time() - total_start:.2f} seconds')

            if self.step % self.config.training.checkpoint_freq == 0:
                self.save_checkpoint()

            if self.step % self.config.training.accum_batch_steps == 0:
                self.train_range.set_description(
                    f"loss: {loss_dict['loss'].item():0.4f}, "
                    f"grad_norm: {stat_dict.get('grad_norm', torch.tensor(0.0)).item():0.4f}, "
                )

    def train_step(self, batch):
        self.step += 1

        batch = batch.to(self.device)

        accum = self.config.training.accum_batch_steps
        # DDP синхронизирует градиенты на каждом backward. При накоплении это
        # лишнее: пока шаг оптимизатора не наступил, градиенты копятся локально.
        # Для GPT2-medium (355M параметров) каждый обмен -- порядка 1.4 ГБ, и
        # при accum=16 пятнадцать из шестнадцати обменов не нужны. Отключаем их
        # на промежуточных микрошагах: результат математически тот же,
        # синхронизация происходит на последнем микрошаге накопления.
        is_accumulating = accum > 1 and self.step % accum != 0
        sync_ctx = (self.ddp_model.no_sync()
                    if is_accumulating and isinstance(
                        self.ddp_model, torch.nn.parallel.DistributedDataParallel)
                    else contextlib.nullcontext())

        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        with sync_ctx:
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                raw_loss = self.ddp_model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    position_ids=batch["position_ids"],
                    labels=batch["labels"],
                )
                loss = raw_loss / accum

            if self.grad_scaler is not None:
                self.grad_scaler.scale(loss).backward()
            else:
                loss.backward()

        loss_dict = {'loss': raw_loss.detach(), 'total_loss': raw_loss.detach()}
        stat_dict = {}

        if self.step % self.config.training.accum_batch_steps == 0:
            stat_dict["grad_norm"] = self.optimizer_step()
            if self.grad_scaler is not None:
                stat_dict["scale_factor"] = torch.tensor(self.grad_scaler.get_scale())

        if self.step % 10 == 0:
            for k, v in loss_dict.items():
                self.log_metric(k, 'train', v.item())
            for k, v in stat_dict.items():
                if isinstance(v, torch.Tensor):
                    self.log_metric("statistics", k, v.item())
            # загрузка карты и память: чтение счетчиков torch бесплатно,
            # NVML опрашивается не чаще раза в минуту (см. gpu_stats)
            for k, v in gpu_stats().items():
                self.log_metric("gpu", k, v)

        return loss_dict, stat_dict

    @torch.no_grad()
    def validate(self):
        self.set_valid_data_generator()
        prev_mode = self.ddp_model.training

        self.ddp_model.eval()

        valid_loss_sum = 0.0
        valid_count = 0.0

        for batch in self.valid_loader:
            batch = batch.to(self.device)
            device_type = "cuda" if torch.cuda.is_available() else "cpu"
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                loss = self.ddp_model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    position_ids=batch["position_ids"],
                    labels=batch["labels"],
                )

            bs = batch["input_ids"].size(0)
            valid_loss_sum += loss.item() * bs
            valid_count += bs

        if is_ddp():
            loss_tensor = torch.tensor(valid_loss_sum, device=self.device)
            count_tensor = torch.tensor(valid_count, device=self.device)
            loss_tensor = reduce_tensor(loss_tensor)
            count_tensor = reduce_tensor(count_tensor)
            mean_loss = (loss_tensor / count_tensor).item()
        else:
            mean_loss = valid_loss_sum / valid_count

        self.log_metric('loss', 'valid_loader', mean_loss)

        self.ddp_model.train(prev_mode)

    @torch.no_grad()
    def generate_text_conditional(self, dataset_name: str, split: str):
        dt = next(get_dataset_iter(self.config, dataset_name, split=split))
        loader = DataLoader(
            dt,
            num_workers=_loader_workers(),
            batch_size=self.config.validation.batch_size,
            collate_fn=self.collate_fn,
        )

        result_dict = {"GEN": [], "TRG": [], "SRC": []}

        decoding = getattr(self.config, "decoding", "greedy")
        sampling_cfg = getattr(self.config, "sampling", {})

        if decoding == "greedy":
            gen_kwargs = dict(do_sample=False)
        else: 
            gen_kwargs = dict(
                do_sample=True,
                temperature=sampling_cfg.get("temperature", 1.0),
                top_p=sampling_cfg.get("top_p", 0.95),
                top_k=sampling_cfg.get("top_k", 0),
            )

        print(f"Loader size: {len(loader)}  |  decoding: {decoding}  |  kwargs: {gen_kwargs}")

        gen_total_time = 0.0
        gen_total_count = 0

        for batch in loader:
            batch = batch.to(self.device)

            tok_src = self.tokenizer(
                batch["text_src"],
                add_special_tokens=False,
                padding=True,
                truncation=True,
                max_length=self.config.data.max_context_len,
                return_tensors="pt",
                return_attention_mask=True,
            ).to(self.device)

            device_type = "cuda" if torch.cuda.is_available() else "cpu"
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _t0 = time.time()
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16,
                                enabled=torch.cuda.is_available()):
                generated = self.model.generate(
                    input_ids=tok_src["input_ids"],
                    attention_mask=tok_src["attention_mask"],
                    max_new_tokens=self.config.data.max_sequence_len,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    **gen_kwargs,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            gen_total_time += time.time() - _t0
            gen_total_count += int(tok_src["input_ids"].shape[0])

            src_len = tok_src["input_ids"].shape[1]
            new_tokens = generated[:, src_len:]
            gen_texts = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

            if not self.printed_example_estimate and get_rank() == 0:
                for i in range(min(10, len(gen_texts))):
                    print(f"EXAMPLE #{i+1}")
                    print(f"  Source   : {batch['text_src'][i]}")
                    print(f"  Target   : {batch['text_trg'][i]}")
                    print(f"  Generated: {gen_texts[i]}")
                self.printed_example_estimate = True

            result_dict["TRG"] += batch["text_trg"]
            result_dict["GEN"] += gen_texts
            result_dict["SRC"] += batch["text_src"]

            if len(result_dict["TRG"]) >= (self.config.validation.num_gen_texts // get_world_size()):
                break

        self._last_gen_time = gen_total_time
        self._last_gen_count = gen_total_count

        return result_dict

    @torch.no_grad()
    def estimate(self, split: str):
        self.printed_example_estimate = False
        self.model.eval()
        self.ddp_model.eval()

        result_dict = dict()
        timing_per_dataset = dict() 

        for dataset_name in self.config.data.datasets.datasets_list:
            seed = self.config.seed + self.step + get_rank()
            set_seed(seed)
            result_dict[dataset_name] = self.generate_text_conditional(dataset_name, split=split)

            local_time = float(getattr(self, "_last_gen_time", 0.0))
            local_cnt = int(getattr(self, "_last_gen_count", 0))
            if is_ddp():
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
            if get_rank() == 0:
                print(f"\n[GEN TIMING] {dataset_name}: total {total_time:.2f}s "
                      f"for {total_cnt} examples → {per_ex_ms:.2f} ms/example "
                      f"(world_size={get_world_size()})")
                self.log_metric("gen_time", "total_sec", total_time)
                self.log_metric("gen_time", "per_example_ms", per_ex_ms)

            if get_rank() == 0:
                print("\n" + "=" * 80)
                print(f"GENERATED EXAMPLES - Step {self.step} - {split.upper()} (полные тексты)")
                print("=" * 80)
                for ds_name in result_dict:
                    print(f"\n{'=' * 40}")
                    print(f"Dataset: {ds_name}")
                    print('=' * 40)
                    keys = list(result_dict[ds_name].keys())
                    for i in range(min(10, len(result_dict[ds_name][keys[0]]))):
                        print(f"\nПример {i + 1}:")
                        print(f"\nSOURCE:\n{result_dict[ds_name]['SRC'][i]}")
                        print(f"\nTARGET:\n{result_dict[ds_name]['TRG'][i]}")
                        print(f"\nGENERATED:\n{result_dict[ds_name]['GEN'][i]}")
                        print("-" * 40)
                print("=" * 80 + "\n")

        if is_ddp():
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

        if get_rank() == 0:
            if not os.path.exists(self.config.validation.texts_path):
                os.makedirs(self.config.validation.texts_path)
            prefix_folder = os.path.join(self.config.validation.texts_path, self.config.training.checkpoints_prefix)
            if not os.path.exists(prefix_folder):
                os.makedirs(prefix_folder)
            file_name = f"{self.step}-seed={self.config.seed}-len={len(result_list)}.json"
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

        if get_rank() == 0:
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

        seed = self.config.seed + self.step + get_rank()
        set_seed(seed)

        self.ddp_model.train()

        metrics_dict["__timing__"] = timing_per_dataset

        return metrics_dict

    @torch.no_grad()
    def run_statistical_eval(self, split: str, num_seeds: int):
        base_seed = int(getattr(self.config, "seed", 0))
        seed_step = int(getattr(self.config, "seed_step", 1000))
        decoding = getattr(self.config, "decoding", "greedy")

        if get_rank() == 0:
            print("\n" + "#" * 80)
            print(f"# STATISTICAL EVALUATION")
            print(f"# num_seeds  = {num_seeds}")
            print(f"# base_seed  = {base_seed}")
            print(f"# seed_step  = {seed_step}")
            print(f"# decoding   = {decoding}")
            if decoding == "greedy":
                print(f"# WARNING: decoding=greedy — разные seed дадут ОДИНАКОВЫЕ")
                print(f"#          результаты. Для статистики используйте sampling.")
            print("#" * 80 + "\n")

        all_runs = [] 
        for i in range(num_seeds):
            run_seed = base_seed + i * seed_step
            if get_rank() == 0:
                print("\n" + "=" * 80)
                print(f"=== RUN {i + 1}/{num_seeds}   (base seed = {run_seed}) ===")
                print("=" * 80 + "\n")

            self.config.seed = run_seed
            set_seed(run_seed + get_rank())

            metrics = self.estimate(split)
            all_runs.append(metrics)

        self.config.seed = base_seed

        if get_rank() == 0:
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
        print(f"  decoding   : {getattr(self.config, 'decoding', 'greedy')}")
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
            "decoding": getattr(self.config, "decoding", "greedy"),
            "sampling": dict(getattr(self.config, "sampling", {}) or {}),
            "num_gen_texts": self.config.validation.num_gen_texts,
            "step": self.step,
            "metrics": aggregated,
            "timing": timing_aggregated,
        }
        file_name = (f"statistical_eval-{split}-num_seeds={num_seeds}"
                     f"-base_seed={base_seed}-step={self.step}.json")
        save_path = os.path.join(prefix_folder, file_name)
        with open(save_path, "w") as f:
            json.dump(payload, f, indent=4)
        print(f"Aggregated statistical metrics saved to:\n  {save_path}\n")

    def save_checkpoint(self, last: bool = False):
        if get_rank() != 0:
            return

        if not os.path.exists(self.config.training.checkpoints_folder):
            os.makedirs(self.config.training.checkpoints_folder)

        prefix_folder = os.path.join(self.config.training.checkpoints_folder, self.config.training.checkpoints_prefix)
        if not os.path.exists(prefix_folder):
            os.makedirs(prefix_folder)

        # Чекпоинт для ДОзапуска пишется всегда, независимо от метрики.
        # Нумерованные файлы отбираются по top-k (у gpt их всего 2): шаг, не
        # попавший в top-k, на диск не ложится. Прогон, снятый по лимиту
        # времени, откатывался бы тогда не к последнему шагу, а к лучшему по
        # метрике -- при 224 ч обучения и лимите 75 ч это давало бы
        # бесконечный цикл дозапусков, ведь mauve не растет монотонно.
        self.__save_checkpoint(os.path.join(prefix_folder, "last.pth"))
        if last:
            return

        save_path = os.path.join(prefix_folder, str(self.step) + ".pth")

        if self.step not in self.tracked_test_metric:
            self.__save_checkpoint(save_path)
            return

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
                # Учет top-k поднимается из чекпоинта, поэтому в куче могут
                # оказаться пути прошлого прогона -- каталог мог быть почищен
                # руками. Отсутствие файла не повод ронять обучение.
                if os.path.exists(heap_smallest[1]):
                    os.remove(heap_smallest[1])
                else:
                    print(f"[WARNING] нечего удалять, файла уже нет: {heap_smallest[1]}")
                heapq.heappop(self.all_checkpoints)
                self.__save_checkpoint(item[1])
                heapq.heappush(self.all_checkpoints, item)

    def __save_checkpoint(self, save_path):
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.grad_scaler.state_dict() if self.grad_scaler else None,
            "step": self.step,
            "config": self.config,
            "tracked_metrics": {
                "test_metrics": self.tracked_test_metric,
                "all_checkpoints": [(score, path) for score, path in self.all_checkpoints],
            },
        }
        # Пишем через временный файл: задание может быть снято по лимиту прямо
        # во время записи, а недописанный last.pth сломал бы дозапуск.
        # os.replace атомарен в пределах одной ФС.
        tmp_path = save_path + ".tmp"
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, save_path)
        print(f"Save model to: {save_path}")

    def load_checkpoint(self):
        prefix_folder = os.path.join(self.config.training.checkpoints_folder, self.config.training.checkpoints_prefix)
        if not os.path.exists(prefix_folder):
            return False

        checkpoint_name = resume_checkpoint_path(
            prefix_folder, self.config.training.checkpoint_name)
        if checkpoint_name is None:
            return False

        load = torch.load(checkpoint_name, map_location="cpu")

        # без этого top-k после дозапуска считается с нуля: файлы прошлого
        # прогона остаются вне учета и никогда не удаляются
        if load.get("tracked_metrics"):
            self.tracked_test_metric = dict(load["tracked_metrics"]["test_metrics"])
            self.all_checkpoints = [tuple(t) for t in load["tracked_metrics"]["all_checkpoints"]]
            heapq.heapify(self.all_checkpoints)

        self.model.load_state_dict(load["model"])
        self.optimizer.load_state_dict(load["optimizer"])
        self.scheduler.load_state_dict(load["scheduler"])
        if self.grad_scaler and load["scaler"]:
            self.grad_scaler.load_state_dict(load["scaler"])
        self.step = load["step"]

        if get_rank() == 0:
            print(f"Checkpoint is loaded {checkpoint_name}")
        return True

    def restore_parameters(self, device=None):
        prefix_folder = os.path.join(self.config.training.checkpoints_folder, self.config.training.checkpoints_prefix)
        if not os.path.exists(prefix_folder):
            raise FileNotFoundError(
                f"Не найден каталог чекпоинтов: {prefix_folder}. "
                f"Имя вычисляется из конфига (architecture_type={self.config.architecture_type}, "
                f"split_scheme={self.config.data.split_scheme}). "
                f"Сначала обучите модель в этом же режиме."
            )

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
        self.step = load["step"]
        print(f'LOADED GPT2 from {checkpoint_name}, step={self.step}')
        self.model.load_state_dict(load["model"])
        if "tracked_metrics" in load:
            self.tracked_test_metric = load["tracked_metrics"]["test_metrics"]
            self.all_checkpoints = load["tracked_metrics"]["all_checkpoints"]
