
import os
import time
import torch
import torch.distributed as dist

from diffusion_holder import DiffusionRunner
from utils.util import set_seed, parse
from create_config import create_config

if __name__ == '__main__':
    args = parse()
    config = create_config(args)

    config.eval = True

    config.seed = int(getattr(args, "seed", 0))

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank, world_size = 0, 1

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
        print("EVAL DIFFUSION MODEL  (single run)")
        print("=" * 60)
        print(f"  Checkpoint prefix : {config.training.checkpoints_prefix}")
        print(f"  Architecture      : {config.architecture_type} "
              f"(cross-attn={config.use_cross_attention}, latent_repl={config.use_latent_replacement}, "
              f"guidance={config.classifier_guidance}, scale={config.guidance_scale})")
        print(f"  Checkpoint name   : {config.training.checkpoint_name or '<latest>'}")
        print(f"  Scheduler         : {config.dynamic.scheduler}, coef_d: {config.dynamic.coef_d}")
        print(f"  emb               : {config.emb}")
        print(f"  is_conditional    : {config.is_conditional}")
        print(f"  Diffusion N steps : {config.dynamic.N}")
        print(f"  Num gen texts     : {config.validation.num_gen_texts}")
        print(f"  Seed              : {config.seed}")
        print("###################")
        print(config)

    set_seed(config.seed + dist.get_rank())

    start_time = time.time()
    DiffusionRunner(config, eval=config.eval)
    elapsed = time.time() - start_time

    if dist.get_rank() == 0:
        print(f"\n{'=' * 60}")
        print(f"ГОТОВО! Время: {elapsed:.1f} сек ({elapsed / 60:.1f} мин)")
        print(f"{'=' * 60}")
