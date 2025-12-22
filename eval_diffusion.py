"""
Скрипт для eval/генерации через DiffusionRunner.
Запуск: torchrun --nproc_per_node=1 eval_diffusion.py [args]

Или через shell скрипт: ./run_eval.sh
"""

import os
import sys
import time
import torch
import torch.distributed as dist

from diffusion_holder import DiffusionRunner
from utils.util import set_seed, parse
from create_config import create_config

if __name__ == '__main__':
    # Парсим стандартные аргументы как в train_diffusion.py
    args = parse()

    # Создаем конфиг
    config = create_config(args)

    # === ПЕРЕОПРЕДЕЛЯЕМ ПУТИ К ЧЕКПОИНТАМ ===
    # Раскомментируй и измени если нужно:

    # Путь к декодеру
    config.decoder.decoder_path = "datasets/rocstories/decoder_rocstories_bert_cased_spt_3l_transformer_0_15x_t_noise.pth"

    # Имя чекпоинта (без .pth)
    config.training.checkpoint_name = "rocstory-bert-base-cased-sd-9-spt_100000"

    # Префикс папки чекпоинтов (если отличается от сгенерированного)
    # config.training.checkpoints_prefix = "tencdm-bert-base-cased-384-0.0002-rocstories-cfg=0.0"

    # === ПАРАМЕТРЫ ГЕНЕРАЦИИ ===
    config.validation.num_gen_texts = 98161  # Сколько текстов генерировать
    # config.validation.batch_size = 32
    # config.dynamic.N = 50  # Шаги диффузии

    # === EVAL MODE ===
    config.eval = True

    # DDP инициализация (нужна даже для 1 GPU)
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = 0
        world_size = 1

    config.local_rank = rank
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    config.training.batch_size_per_gpu = config.training.batch_size // dist.get_world_size()
    # config.is_conditional = True

    if dist.get_rank() == 0:
        print("=" * 60)
        print("EVAL DIFFUSION MODEL")
        print("=" * 60)
        print(f"\nКонфиг:")
        print(f"  Checkpoint prefix: {config.training.checkpoints_prefix}")
        print(f"  Checkpoint name: {config.training.checkpoint_name}")
        print(f"  Decoder: {config.decoder.decoder_path}")
        print(f"  Scheduler: {config.dynamic.scheduler}, coef_d: {config.dynamic.coef_d}")
        print(f"  Emb: {config.emb}")
        print(f"  Num gen texts: {config.validation.num_gen_texts}")
        print(f"  Diffusion steps: {config.dynamic.N}")
        print()

    seed = config.seed + dist.get_rank()
    set_seed(seed)

    # Запуск - DiffusionRunner при eval=True сам вызовет estimate("test")
    start_time = time.time()

    diffusion = DiffusionRunner(config, eval=True)

    elapsed = time.time() - start_time

    if dist.get_rank() == 0:
        print(f"\n{'=' * 60}")
        print(f"ГОТОВО! Время: {elapsed:.1f} сек ({elapsed / 60:.1f} мин)")
        print(f"{'=' * 60}")