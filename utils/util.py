import torch
import random
import argparse
import time as _time
import numpy as np
from copy import deepcopy
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.functional import cross_entropy

from utils.schemes import (
    ARCHITECTURE_TYPES, ARCHITECTURE_TYPE_HELP,
    SPLIT_SCHEMES, SPLIT_SCHEME_HELP,
    AUGMENTATION_SCHEMES, AUGMENTATION_SCHEME_HELP,
)


def set_seed(seed: int = 0):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.benchmark = True


def dict_to_cuda(d):
    for key in d:
        d[key] = d[key].cuda(non_blocking=True)
    return d


def dict_to_tensor_cuda(d):
    for key in ["input_ids", "attention_mask", "token_type_ids"]:
        if key not in d:
            continue
        d[key] = torch.Tensor(d[key]).cuda(non_blocking=True)
    return d


def dict_to_tensors(d):
    for key in ["input_ids", "attention_mask", "token_type_ids"]:
        d[key] = torch.tensor(d[key])
    return d


def dict_to_device(d, device):
    return {k: v.to(device) for k, v in d.items()}


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def reduce_sum_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt


def masked_mean(tensor, mask):
    return torch.sum(tensor * mask[:, :, None], dim=[0, 1]) / torch.sum(mask)


def masked_std(tensor, mask):
    mean = masked_mean(tensor, mask)
    return torch.sqrt(torch.sum(tensor ** 2 * mask[:, :, None], dim=[0, 1]) / torch.sum(mask) - mean ** 2)


def parse_checkpoint_name(checkpoint_name):
    items = checkpoint_name.split("-")
    params = dict()
    for item in items:
        key, value = item.split("=")
        params[key] = value
    return params


def make_mask_wo_SEP_CLS(mask):
    mask = deepcopy(mask)
    mask.scatter_(dim=1, index=(mask.sum(dim=1) - 1).reshape(-1, 1), src=torch.zeros_like(mask))
    mask[:, 0] = 0
    return mask


def get_ravel_weights(model):
    ww = []
    for par in model.parameters():
        ww.append(par.detach().cpu().data.numpy().ravel())
    return np.concatenate(ww)


def get_ravel_grad(model):
    ww = []
    for par in model.parameters():
        ww.append(par.grad.detach().cpu().data.numpy().ravel())
    return np.concatenate(ww)


def bert_acc(targets, outputs, mask):
    if mask is None:
        mask = torch.ones(
            (targets.shape[0], targets.shape[1]),
            device=f"cuda:{dist.get_rank()}" if dist.is_initialized() else "cuda:0",
            requires_grad=False,
            dtype=torch.int64,
        )
    pred_tokens = outputs.argmax(dim=-1)

    mask = deepcopy(mask)
    mask.scatter_(dim=1, index=(mask.sum(dim=1) - 1).reshape(-1, 1), src=torch.zeros_like(mask))
    mask[:, 0] = 0
    return torch.sum(mask * (targets == pred_tokens)) / torch.sum(mask)


def mse_loss(inputs, targets, mask):
    if mask is None:
        mask = torch.ones(
            (targets.shape[0], targets.shape[1]),
            device=f"cuda:{dist.get_rank()}" if dist.is_initialized() else "cuda:0",
            requires_grad=False,
            dtype=torch.int64,
        )
    losses = torch.mean(torch.square(inputs - targets), dim=-1)
    losses = losses * mask
    loss = torch.sum(losses) / torch.sum(mask)
    return loss


def recon_loss(inputs, outputs, mask):
    if mask is None:
        mask = torch.ones(
            (inputs.shape[0], inputs.shape[1]),
            device=f"cuda:{dist.get_rank()}" if dist.is_initialized() else "cuda:0",
            requires_grad=False,
            dtype=torch.int64,
        )
    losses = cross_entropy(
        input=inputs.reshape(-1, inputs.shape[-1]),
        target=outputs.reshape(-1),
        reduce=False,
    )
    losses = losses * mask.reshape(-1)
    loss = torch.sum(losses) / torch.sum(mask)
    return loss


def get_stat(z, mask):
    if mask is None:
        mask = torch.ones(
            (z.shape[0], z.shape[1]),
            device=f"cuda:{dist.get_rank()}" if dist.is_initialized() else "cuda:0",
            requires_grad=False,
            dtype=torch.int64,
        )
    else:
        mask = make_mask_wo_SEP_CLS(mask)
    mean = masked_mean(z, mask)
    std = masked_std(z, mask)
    norm = torch.sum(torch.norm(z, dim=2) * mask) / torch.sum(mask)
    stat_dict = {
        "mean": torch.mean(mean),
        "std": torch.mean(std),
        "norm": norm
    }
    return stat_dict


def parse():
    parser = argparse.ArgumentParser(description="Dataset arguments")
    parser.add_argument(
        "--dataset_name", type=str, default=None, 
        choices=[
            "rocstories", 
            "wikipedia", 
            "qqp", "xsum", "wiki_auto", 
        ],
        required=False,
    )
    parser.add_argument(
        "--architecture_type", type=str, default=ARCHITECTURE_TYPES[0],
        choices=ARCHITECTURE_TYPES,
        help="Модель и способ подачи условия: "
             + "; ".join(f"{k} -- {v}" for k, v in ARCHITECTURE_TYPE_HELP.items()),
    )
    parser.add_argument("--local-rank", type=int, default=None)
    parser.add_argument("--swap_cfg_coef", type=float, default=0.)
    parser.add_argument("--scheduler", type=str, default='sd')
    parser.add_argument("--coef_d", type=float, default=9)
    # type=bool здесь работал бы неправильно: argparse вызывает bool("False") -> True,
    # то есть выключить флаг было невозможно
    parser.add_argument("--emb", action='store_true',
                        help="Диффундировать word embeddings вместо выхода энкодера")
    parser.add_argument(
        "--no_normalize_encodings", action='store_true',
        help="Не нормализовать энкодинги статистиками датасета (EncNormalizer). "
             "По умолчанию нормализация включена. Влияет только на режим без --emb: "
             "при --emb эмбеддинги всегда нормируются по статистикам словаря",
    )
    parser.add_argument(
        "--split_scheme", type=str, default=None, choices=SPLIT_SCHEMES,
        help="Схема разбиения текста на промпт и продолжение. По умолчанию "
             "выбирается по датасету (rocstories -- half, wikipedia -- prefix_lm): "
             + ", ".join(f"{k} -- {v}" for k, v in SPLIT_SCHEME_HELP.items()),
    )
    parser.add_argument(
        "--augmentation_scheme", type=str, default=AUGMENTATION_SCHEMES[0],
        choices=AUGMENTATION_SCHEMES,
        help="Схема генерации негативных примеров для классификатора "
             "(только для architecture_type=guidance): "
             + ", ".join(f"{k} -- {v}" for k, v in AUGMENTATION_SCHEME_HELP.items()),
    )
    parser.add_argument(
        "--classifier_guidance_scale", type=float, default=0.0,
        help="Сила classifier guidance. Имеет смысл только при architecture_type=guidance",
    )
    parser.add_argument(
        "--time_scale", type=float, default=1.0,
        help="Масштаб непрерывного t перед синусоидальным эмбеддингом внутри "
             "классификатора. При 1.0 (по умолчанию) эмбеддинги соседних t почти "
             "совпадают и уровень шума слабо информативен; 1000.0 дает разрешение "
             "как в DDPM с T=1000. Влияет только на architecture_type=guidance",
    )
    parser.add_argument("--mode", type=str, default="transformer",
                        help="Архитектура декодера")
    # --- параметры декодирования, только для architecture_type=gpt ---------------
    parser.add_argument("--decoding", type=str, default="greedy", choices=["greedy", "sampling"])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--eval", action='store_true')
    parser.add_argument(
        "--encoder_name", type=str, default='bert-base-cased',
        choices=[
            "bert-base-cased",
            "t5-base",
            "roberta-base",
            "bart-base"
        ])
    parser.add_argument('--project_name', type=str, default='test')

    parser.add_argument("--seed", type=int, default=0,
                        help="Базовый random seed для генерации")
    parser.add_argument("--num_seeds", type=int, default=1,
                        help="Сколько независимых прогонов делать с разными "
                             "сидами для оценки std и 95%% CI метрик")
    parser.add_argument("--seed_step", type=int, default=1000,
                        help="Шаг между базовыми сидами соседних прогонов")

    return parser.parse_args()


# --- статистика GPU для логов обучения -------------------------------------------
# Мониторинг кластера показывал 20% загрузки карт при 22% занятой памяти, но
# понять это можно было только постфактум из отчета HPC TaskMaster. Логируем те
# же величины прямо в обучении, чтобы видеть их на кривых wandb.
_UTIL_STATE = {"last": 0.0, "value": None}


def gpu_stats(util_every_sec: float = 60.0):
    """Память и загрузка текущей карты. None, если GPU нет.

    Память -- это чтение счетчиков torch, стоит наносекунды и не синхронизирует
    поток. Загрузка идет через NVML и стоит дороже, поэтому опрашивается не
    чаще раза в минуту, а между опросами возвращается последнее значение.
    """
    import torch as _torch

    if not _torch.cuda.is_available():
        return {}

    dev = _torch.cuda.current_device()
    total = _torch.cuda.get_device_properties(dev).total_memory
    stats = {
        "gpu_mem_reserved_gb": _torch.cuda.memory_reserved(dev) / 2 ** 30,
        "gpu_mem_peak_gb": _torch.cuda.max_memory_allocated(dev) / 2 ** 30,
        "gpu_mem_percent": 100.0 * _torch.cuda.memory_reserved(dev) / total,
    }

    now = _time.time()
    if now - _UTIL_STATE["last"] >= util_every_sec:
        _UTIL_STATE["last"] = now
        try:
            _UTIL_STATE["value"] = _torch.cuda.utilization(dev)
        except Exception:
            _UTIL_STATE["value"] = None
    if _UTIL_STATE["value"] is not None:
        stats["gpu_util_percent"] = float(_UTIL_STATE["value"])
    return stats


# --- выбор чекпоинта для дозапуска ----------------------------------------------
def diffusion_checkpoint_folder(config):
    """Каталог чекпоинтов диффузии для того, кто их только ЧИТАЕТ.

    Классификаторы augmented/combined реконструируют x_0 замороженной
    безусловной диффузией: score_estimator переводится в eval и в
    requires_grad=False, в этот каталог не пишется ничего.

    Поэтому в SMOKE-режиме, где к prefix добавляется -smoke, разрешаем откат на
    боевой каталог, когда smoke-каталога нет. Иначе короткая проверка требовала
    бы сначала обучить безусловную диффузию отдельным smoke-заданием -- часы
    ради 200 шагов проверки классификатора, при том что проверяться на том
    самом файле, который возьмет боевой прогон, даже честнее. Ровно та же
    логика, что у smoke-декодера в create_config.apply_smoke_overrides.

    Обратный откат невозможен по построению: суффикс -smoke появляется только
    в smoke-режиме, так что боевой прогон никогда не прочитает smoke-веса.
    """
    import os as _os

    prefix = config.training.checkpoints_prefix
    folder = _os.path.join(config.training.checkpoints_folder, prefix)
    if _os.path.exists(folder):
        return folder

    suffix = "-smoke"
    if prefix.endswith(suffix):
        real = _os.path.join(config.training.checkpoints_folder, prefix[:-len(suffix)])
        if _os.path.exists(real):
            print(f"SMOKE: smoke-чекпоинта диффузии нет, читаем боевой {real}",
                  flush=True)
            return real

    raise FileNotFoundError(
        f"Checkpoint folder not found: {folder}\n"
        f"Классификаторы augmented/combined реконструируют x_0 безусловной "
        f"диффузией -- ее нужно обучить раньше: ./run_wikipedia.sh unconditional"
    )


def resume_checkpoint_path(prefix_folder: str, checkpoint_name=None):
    """Путь к чекпоинту, с которого продолжается обучение. None -- продолжать нечего.

    Нумерованные файлы <шаг>.pth отбираются по top-k лучших по метрике: шаг, не
    попавший в top-k, на диск вообще не ложится. Если брать max(<шаг>), то
    прогон, снятый по лимиту времени, откатывался бы не к последнему шагу, а к
    лучшему по метрике. У gpt (224 ч при лимите 75 ч) это давало бы
    бесконечный цикл дозапусков: метрика не обязана расти монотонно, и лучший
    шаг мог остаться далеко позади.

    Поэтому дозапуск идет с last.pth -- его пишет каждый checkpoint_freq
    независимо от метрики, и его шаг всегда не меньше любого нумерованного.
    Нумерованные остаются запасным вариантом (каталоги прошлых прогонов, где
    last.pth еще не было).
    """
    import os as _os

    if not _os.path.exists(prefix_folder):
        return None

    if checkpoint_name:
        path = _os.path.join(prefix_folder, f"{checkpoint_name}.pth")
        return path if _os.path.exists(path) else None

    last_path = _os.path.join(prefix_folder, "last.pth")
    if _os.path.exists(last_path):
        return last_path

    names = [str(t)[:-4] for t in _os.listdir(prefix_folder) if str(t).endswith(".pth")]
    steps = [int(t) for t in names if t.isdigit()]
    if not steps:
        return None
    return _os.path.join(prefix_folder, f"{max(steps)}.pth")
