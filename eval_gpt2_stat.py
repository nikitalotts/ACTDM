
import os
import time
import torch
import torch.distributed as dist

from gpt2_holder import GPT2Runner
from utils.util import set_seed, parse
from create_config import create_config

if __name__ == '__main__':
    args = parse()
    # эти entrypoint-ы работают только с авторегрессионной моделью
    args.architecture_type = "gpt"
    config = create_config(args)


    decoding = getattr(args, "decoding", "greedy")
    config.decoding = decoding
    config.sampling = getattr(config, "sampling", {}) or {}
    config.sampling["temperature"] = getattr(args, "temperature", 1.0)
    config.sampling["top_p"] = getattr(args, "top_p", 0.95)
    config.sampling["top_k"] = getattr(args, "top_k", 0)

    config.seed = int(getattr(args, "seed", 0))
    config.num_seeds = int(getattr(args, "num_seeds", 5))
    config.seed_step = int(getattr(args, "seed_step", 1000))

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = 0
        world_size = 1

    config.local_rank = rank
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(
        backend='nccl', init_method='env://',
        world_size=world_size, rank=rank,
    )
    torch.distributed.barrier()
    config.training.batch_size_per_gpu = config.training.batch_size // dist.get_world_size()

    if dist.get_rank() == 0:
        print("=" * 60)
        print("STATISTICAL EVAL — GPT-2 MODEL")
        print("=" * 60)
        print(f"  Checkpoint prefix : {config.training.checkpoints_prefix}")
        print(f"  Checkpoint name   : {config.training.checkpoint_name}")
        print(f"  Num gen texts     : {config.validation.num_gen_texts}")
        print(f"  Decoding          : {config.decoding}")
        if config.decoding == "sampling":
            print(f"  Temperature       : {config.sampling['temperature']}")
            print(f"  top_p             : {config.sampling['top_p']}")
            print(f"  top_k             : {config.sampling['top_k']}")
        print(f"  BASE SEED         : {config.seed}")
        print(f"  NUM SEEDS (runs)  : {config.num_seeds}")
        print(f"  SEED STEP         : {config.seed_step}")
        if config.decoding == "greedy" and config.num_seeds > 1:
            print("  !! WARNING: greedy + num_seeds>1 → std=0 (генерация")
            print("     детерминирована). Используйте --decoding sampling.")
        print("###################")
        print(config)

    set_seed(config.seed + dist.get_rank())

    start_time = time.time()
    runner = GPT2Runner(config, eval=True)
    elapsed = time.time() - start_time

    if dist.get_rank() == 0:
        print(f"\n{'=' * 60}")
        print(f"ГОТОВО! Время: {elapsed:.1f} сек ({elapsed / 60:.1f} мин)")
        print(f"{'=' * 60}")
