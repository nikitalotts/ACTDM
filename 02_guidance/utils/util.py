import torch
import random
import argparse
import numpy as np
from copy import deepcopy
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.functional import cross_entropy

from utils.schemes import (
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
        "--split_scheme", type=str, default=SPLIT_SCHEMES[0], choices=SPLIT_SCHEMES,
        help="Схема разбиения истории rocstories на промпт и продолжение "
             "(должна совпадать с той, с которой скачивался датасет в data/load.py): "
             + ", ".join(f"{k} -- {v}" for k, v in SPLIT_SCHEME_HELP.items()),
    )
    parser.add_argument(
        "--augmentation_scheme", type=str, default=AUGMENTATION_SCHEMES[0],
        choices=AUGMENTATION_SCHEMES,
        help="Схема генерации негативных примеров для классификатора: "
             + ", ".join(f"{k} -- {v}" for k, v in AUGMENTATION_SCHEME_HELP.items()),
    )
    parser.add_argument("--mode", type=str, default="transformer")
    parser.add_argument(
        "--encoder_name", type=str, default='bert-base-cased',
        choices=[
            "bert-base-cased",
            "t5-base",
            "roberta-base",
            "bart-base"
        ])
    parser.add_argument('--project_name', type=str, default='test')
    # --- режим работы подпроекта -------------------------------------------------
    # 1) без флагов                    -- безусловная диффузия
    # 2) --is_conditional              -- условная диффузия (промпт через cross-attention)
    # 3) --classifier_guidance         -- безусловная диффузия + classifier guidance при генерации
    # Режимы 2 и 3 взаимоисключающие: в режиме 3 сама диффузия обязана быть безусловной.
    parser.add_argument(
        "--is_conditional", action='store_true',
        help="Условная диффузия: промпт подается в denoising network через cross-attention",
    )
    parser.add_argument(
        "--classifier_guidance", action='store_true',
        help="Безусловная диффузия + classifier guidance на этапе генерации. "
             "Требует обученный ConditionalEncoder и --classifier_guidance_scale > 0. "
             "Несовместим с --is_conditional",
    )
    parser.add_argument(
        "--classifier_guidance_scale", type=float, default=0.0,
        help="Сила classifier guidance. Имеет смысл только вместе с --classifier_guidance",
    )
    parser.add_argument("--use_conditional_encoder", action='store_true',
                        help="DEPRECATED: не используется, режим определяется флагами выше")
    parser.add_argument("--eval", action='store_true')
    parser.add_argument("--num_diffusion_steps", type=int, default=None)
    parser.add_argument("--num_generated_texts", type=int, default=None)

    parser.add_argument("--seed", type=int, default=0,
                        help="Базовый random seed для генерации")
    parser.add_argument("--num_seeds", type=int, default=1,
                        help="Сколько независимых прогонов делать с разными "
                             "сидами для оценки std и 95%% CI метрик")
    parser.add_argument("--seed_step", type=int, default=1000,
                        help="Шаг между базовыми сидами соседних прогонов")

    return parser.parse_args()
