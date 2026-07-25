import os
import sys
import time
import torch
import argparse
import torch.distributed as dist

from gpt2_holder import GPT2Runner
from create_config import create_config
from utils.util import set_seed, parse


if __name__ == '__main__':
    total_start_time = time.time()

    args = parse()
    # эти entrypoint-ы работают только с авторегрессионной моделью
    args.architecture_type = "gpt"
    config = create_config(args)

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1

    config.local_rank = rank
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        torch.distributed.init_process_group(
            backend='nccl', init_method='env://',
            world_size=world_size, rank=rank
        )
    else:
        config.ddp = False
        config.local_rank = 0
        torch.distributed.init_process_group(
            backend='gloo', init_method='env://',
            world_size=max(world_size, 1), rank=max(rank, 0)
        )
    torch.distributed.barrier()

    config.training.batch_size_per_gpu = config.training.batch_size // dist.get_world_size()

    if dist.get_rank() == 0:
        print(config)

    seed = config.seed + dist.get_rank()
    set_seed(seed)

    runner = GPT2Runner(config, config.eval)

    if dist.get_rank() == 0:
        print("=" * 60)
        print("TRAIN GPT2 FROM SCRATCH")
        print("=" * 60)
        print(f"  Checkpoint prefix: {config.training.checkpoints_prefix}")
        print(f"  Dataset: {config.data.datasets.datasets_list[0]}")
        print(f"  Training iters: {config.training.training_iters}")
        print(f"  Batch size: {config.training.batch_size} ({config.training.batch_size_per_gpu} per GPU)")

    if not config.eval:
        train_start_time = time.time()
        runner.train()
        train_time = time.time() - train_start_time

    total_time = time.time() - total_start_time

    if dist.get_rank() == 0:
        print(f'train_gpt2.py finished')
        print(f'Total time: {total_time:.2f}s ({total_time / 60:.2f} min)')
        if not config.eval:
            print(f'Training time: {train_time:.2f}s ({train_time / 60:.2f} min)')

    print('train_gpt2.py finished (from script)')
