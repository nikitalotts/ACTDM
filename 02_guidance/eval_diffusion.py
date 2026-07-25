import os
import sys
import time
import torch
import torch.distributed as dist

from diffusion_holder import DiffusionRunner
from utils.util import set_seed, parse
from create_config import create_config, checkpoints_prefix_suffix

if __name__ == '__main__':
    args = parse()
    config = create_config(args)


    config.model.encoder_link = "bert-base-cased"
    config.decoder.mode = "transformer"
    config.decoder.decoder_path = "datasets/rocstories/3-3decoder-bert-base-cased-80-transformer.pth"
    config.training.checkpoints_folder = "checkpoints"
    config.training.checkpoints_prefix = "actdm-bert-base-cased-512-0.0002-rocstories-cfg=0.0" + checkpoints_prefix_suffix(config)
    config.training.checkpoint_name = "100000"

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

    config.cond_encoder.cond_encoder_path = '/home/nklotts/tencdm/datasets/rocstories/conditional-encoder-bert-base-cased-80-transformer-v10.pth'

    if dist.get_rank() == 0:
        print("=" * 60)
        print("EVAL DIFFUSION MODEL")
        print("=" * 60)
        print(f"\nКонфиг:")
        print(f"  Mode: {config.generation_mode} "
              f"(is_conditional={config.is_conditional}, "
              f"classifier_guidance={config.classifier_guidance}, "
              f"scale={config.guidance_scale})")
        print(f"  Checkpoint prefix: {config.training.checkpoints_prefix}")
        print(f"  Checkpoint name: {config.training.checkpoint_name}")
        print(f"  Decoder: {config.decoder.decoder_path}")
        print(f"  Scheduler: {config.dynamic.scheduler}, coef_d: {config.dynamic.coef_d}")
        print(f"  Emb: {config.emb}")
        print(f"  Num gen texts: {config.validation.num_gen_texts}")
        print(f"  Diffusion steps: {config.dynamic.N}")
        print('###################')
        print(config)

    seed = config.seed + dist.get_rank()
    set_seed(seed)

    start_time = time.time()

    diffusion = DiffusionRunner(config, eval=config.eval)

    elapsed = time.time() - start_time

    if dist.get_rank() == 0:
        print(f"\n{'=' * 60}")
        print(f"ГОТОВО! Время: {elapsed:.1f} сек ({elapsed / 60:.1f} мин)")
        print(f"{'=' * 60}")
