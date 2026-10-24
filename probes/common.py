"""Общая обвязка диагностических прогонов (probe_*.py).

Зачем отдельный модуль
----------------------
Диагностике нужен обученный раннер БЕЗ побочных эффектов. DiffusionRunner при
eval=True сразу зовет estimate() на 5000 текстов, а при eval=False готовит
оптимизатор и может дозапуститься из чекпоинта. Ни то, ни другое диагностике не
нужно: нужен собранный и восстановленный из чекпоинта набор модулей, после чего
прогон сам управляет обратным процессом. Для этого в DiffusionRunner.__init__
добавлен флаг build_only.

Почему обратный процесс продублирован
------------------------------------
Диагностикам нужны перехваты, которых нет у боевого pred_embeddings: ветвление
траектории с другого сида, подмена градиента guidance, снимки промежуточных
состояний. Встраивать это в горячий путь генерации нельзя -- он общий для всех
четырех подходов и уже оттестирован.

Цена дублирования -- риск расхождения. Он закрыт тестом
tests/test_probes.py::test_reverse_process_reproduces_pred_embeddings: при
выключенных перехватах reverse_process обязан дать ПОБИТОВО тот же результат,
что pred_embeddings, при одном сиде. Это страж, а не комментарий: порядок
обращений к генератору случайных чисел здесь часть контракта (сначала
prior_sampling на CPU, затем по одному randn_like на CUDA на каждом шаге).
"""
import os
from functools import partial
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from data.dataset import get_dataset_iter
from diffusion_holder import DiffusionRunner, _loader_workers
from diffusion_utils.solvers import create_solver

EPS_T = 0.01  # то же значение, что в DiffusionRunner.pred_embeddings


def build_probe_runner(config) -> DiffusionRunner:
    """Собрать раннер и восстановить веса, ничего не считая.

    restore_parameters сам переключает модель на EMA-веса -- те же, на которых
    считались все метрики в таблицах. Диагностика обязана мерить именно их,
    иначе ее выводы не относятся к опубликованным числам.
    """
    runner = DiffusionRunner(config, build_only=True)
    runner.restore_parameters(runner.device)
    # restore_parameters возвращает False, если нумерованных чекпоинтов нет, и
    # НЕ выставляет loaded_checkpoint_path. Падать надо здесь, а не через
    # AttributeError в середине прогона: иначе диагностика поехала бы на
    # случайно инициализированных весах и выдала бы правдоподобные числа.
    if not getattr(runner, "loaded_checkpoint_path", None):
        raise FileNotFoundError(
            f"В каталоге чекпоинтов для префикса "
            f"{config.training.checkpoints_prefix} нет ни одного нумерованного "
            f"чекпоинта. Диагностика на необученных весах бессмысленна.")
    runner.score_estimator.eval()
    runner.ddp_score_estimator.eval()
    runner.decoder.eval()
    runner.encoder.eval()
    return runner


def probe_loader(runner: DiffusionRunner, split: str = "test",
                 batch_size: Optional[int] = None) -> DataLoader:
    """Тестовый загрузчик с тем же collate_fn, что у боевого eval.

    shuffle=False: диагностики сравниваются между собой по примерам, и порядок
    обязан совпадать от прогона к прогону.
    """
    dataset_name = runner.config.data.datasets.datasets_list[0]
    dt = next(get_dataset_iter(runner.config, dataset_name, split=split))
    return DataLoader(
        dt,
        num_workers=_loader_workers(),
        batch_size=batch_size or runner.config.validation.batch_size,
        collate_fn=runner.collate_fn,
        shuffle=False,
    )


def pad_batch_to_max(runner: DiffusionRunner, batch):
    """Дополнить input_ids/attention_mask до длин из конфига.

    ЗАЧЕМ. collate_fn подгоняет паддинг под самый длинный текст батча (кроме
    diffuseq, где длина фиксирована, потому что промпт конкатенируется с
    таргетом). Для диагностик этого мало по двум причинам:

      * генерация ВСЕГДА идет на длине data.max_sequence_len, поэтому цель
        reconstruction guidance обязана иметь ровно эту длину -- иначе
        вычитание target - x_0 либо упадет, либо (при удачном совпадении
        размеров) сравнит разные позиции;
      * банк латентов собирается из нескольких батчей, а батчи с разной длиной
        не склеиваются в один тензор;
      * на инференсе классификатор видит таргет-спан фиксированной длины, и
        мерить его точность на обрезанных по батчу таргетах значило бы мерить
        не тот режим, в котором он применяется.

    Паддинг-позиции безопасны: Encoder кладет в них zero_emb, ровно как на
    обучении, а content_mask их исключает из всех статистик.
    """
    pad_id = runner.tokenizer.pad_token_id
    spec = (("input_ids_trg", "attention_mask_trg", runner.config.data.max_sequence_len),
            ("input_ids_src", "attention_mask_src", runner.config.data.max_context_len))
    for ids_key, mask_key, target_len in spec:
        if ids_key not in batch:
            continue
        ids, mask = batch[ids_key], batch[mask_key]
        cur = ids.shape[1]
        if cur == target_len:
            continue
        if cur > target_len:
            batch[ids_key] = ids[:, :target_len]
            batch[mask_key] = mask[:, :target_len]
            continue
        pad = target_len - cur
        batch[ids_key] = torch.cat(
            [ids, torch.full((ids.shape[0], pad), pad_id,
                             dtype=ids.dtype, device=ids.device)], dim=1)
        batch[mask_key] = torch.cat(
            [mask, torch.zeros((mask.shape[0], pad),
                               dtype=mask.dtype, device=mask.device)], dim=1)
    return batch


@torch.no_grad()
def encode_batch(runner: DiffusionRunner, batch):
    """Энкодинги промпта и продолжения в том виде, в каком их видит диффузия.

    Это важно: Encoder применяет enc_normalizer.normalize и подменяет
    спецтокены. Классификатор guidance обучался на нормализованных латентах, и
    зашумление идет уже поверх -- мерить надо в этом же пространстве.

    Длины приводятся к конфигурационным (см. pad_batch_to_max): геометрия
    должна совпасть с той, на которой идет генерация.
    """
    batch = pad_batch_to_max(runner, batch.to(runner.device))
    src_x = None
    if runner.config.is_pipeline_conditional:
        src_x = runner.encoder(
            input_ids=batch["input_ids_src"],
            attention_mask=batch["attention_mask_src"],
        )
    trg_x = runner.encoder(
        input_ids=batch["input_ids_trg"],
        attention_mask=batch["attention_mask_trg"],
    )
    return batch, src_x, trg_x


def make_timesteps(runner: DiffusionRunner) -> torch.Tensor:
    """Сетка t обратного процесса -- копия расчета из pred_embeddings."""
    if runner.config.timesteps == "linear":
        return torch.linspace(
            runner.dynamic.T, EPS_T, runner.dynamic.N + 1, device=runner.device)
    if runner.config.timesteps == "quad":
        deg = 2
        return torch.linspace(1, 0, runner.dynamic.N + 1, device=runner.device) ** deg \
            * (runner.dynamic.T - EPS_T) + EPS_T
    raise Exception(f"Неизвестный config.timesteps={runner.config.timesteps}")


def _wrap_score_fn(runner: DiffusionRunner, guidance_fn):
    """score_fn солвера с внешним градиентом поверх безусловного score.

    Пересчет x_0 из скорректированного score повторяет ветку classifier
    guidance в calc_score один в один. Это обязательно: солвер и
    self-conditioning должны видеть согласованную тройку score/x_0/eps_theta,
    иначе подмена градиента молча меняет еще и вход сети на следующем шаге.
    """
    base = partial(runner.calc_score, model=runner.ddp_score_estimator)
    if guidance_fn is None:
        return base

    def wrapped(x_t, t, **kwargs):
        out = base(x_t=x_t, t=t, **kwargs)
        grad = guidance_fn(x_t=x_t, t=t, out=out, **kwargs)
        if grad is None:
            return out
        params = runner.dynamic.marginal_params(t)
        score = out["score"] + grad
        x_0 = (x_t + params["std"] ** 2 * score) / params["mu"]
        eps_theta = (x_t - params["mu"] * x_0) / params["std"]
        return {"score": score, "x_0": x_0, "eps_theta": eps_theta}

    return wrapped


def reverse_process(
        runner: DiffusionRunner,
        batch_size: int,
        cond_x: Optional[torch.Tensor] = None,
        cond_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        guidance_fn=None,
        resume: Optional[Dict] = None,
        record_at=None,
        record_out: Optional[List[Dict]] = None,
):
    """Обратный процесс с перехватами. Шаг в шаг повторяет pred_embeddings.

    guidance_fn -- вызывается на каждом шаге, возвращает добавку к score
                   (или None). Сигнатура: (x_t, t, out, **kwargs) -> Tensor.
    resume      -- {"z": полное состояние, "x_0_self_cond": ..., "start_idx": i}
                   продолжить траекторию с шага i вместо старта из шума.
                   prior_sampling при этом НЕ вызывается, то есть генератор
                   случайных чисел не трогается до первого шага солвера.
    record_at   -- множество индексов шагов, на которых снять снимок состояния.
    record_out  -- список, куда класть снимки.

    Возвращает латенты продолжения (без src-части) -- как pred_embeddings.
    """
    if attention_mask is not None:
        attention_mask = attention_mask.cuda()

    shape = (
        batch_size,
        runner.config.data.max_sequence_len,
        runner.encoder.encoder.config.hidden_size,
    )
    record_at = set() if record_at is None else set(int(i) for i in record_at)

    solver = create_solver(runner.config)(
        dynamic=runner.dynamic,
        score_fn=_wrap_score_fn(runner, guidance_fn),
        ode_sampling=runner.config.training.ode_sampling,
    )

    with torch.no_grad():
        if resume is None:
            x_init = runner.dynamic.prior_sampling(shape).to(runner.device)
            start_idx = 0
        else:
            # Геометрию входа пересобираем тем же кодом, что и боевой путь, а
            # потом подставляем сохраненное состояние: так ветвление не зависит
            # от того, конкатенирует подход промпт с таргетом или нет.
            # Нули вместо шума -- чтобы не тронуть генератор случайных чисел.
            x_init = torch.zeros(shape, device=runner.device)
            start_idx = int(resume["start_idx"])

        src_len, z, se_mask, se_cond, se_cond_mask = runner.build_score_estimator_input(
            x_t=x_init,
            cond_x=cond_x,
            attention_mask=attention_mask,
            cond_mask=cond_mask,
        )

        if resume is None:
            x_0_self_cond = torch.zeros_like(z, dtype=z.dtype)
        else:
            z = resume["z"].clone()
            x_0_self_cond = resume["x_0_self_cond"].clone()

        timesteps = make_timesteps(runner)
        z_mean = z

        for idx in range(start_idx, runner.dynamic.N):
            t = timesteps[idx]
            next_t = timesteps[idx + 1]

            input_t = t * torch.ones(shape[0], device=runner.device)
            next_input_t = next_t * torch.ones(shape[0], device=runner.device)

            output = solver.step(
                x_t=z, t=input_t, next_t=next_input_t,
                cond=se_cond,
                cond_mask=se_cond_mask,
                attention_mask=se_mask,
                x_0_self_cond=x_0_self_cond,
            )

            z, z_mean = output["x"], output["x_mean"]
            x_0_self_cond = output["x_0"]

            # diffuseq: латенты промпта фиксируются на каждом шаге
            if src_len > 0:
                z[:, :src_len, :] = cond_x
                z_mean[:, :src_len, :] = cond_x
                x_0_self_cond[:, :src_len, :] = cond_x

            if idx in record_at and record_out is not None:
                record_out.append({
                    "idx": idx,
                    "t": float(t.item()),
                    "z": z.detach().clone(),
                    "x_0_self_cond": x_0_self_cond.detach().clone(),
                    "start_idx": idx + 1,
                })

        return z_mean[:, src_len:, :]


@torch.no_grad()
def latents_to_tokens(runner: DiffusionRunner, pred_embeddings,
                      cond_x=None, cond_mask=None):
    """Латенты -> (тексты, списки токенов). Обрезка по end-токену как в боевом пути."""
    output = runner.pred_logits(pred_embeddings, cond_x=cond_x, cond_mask=cond_mask)
    tokens = output.argmax(dim=-1)

    end_tokens = []
    tk = runner.tokenizer
    if getattr(tk, "eos_token", None) is not None:
        end_tokens.append(tk.vocab[tk.eos_token])
    if getattr(tk, "sep_token", None) is not None:
        end_tokens.append(tk.vocab[tk.sep_token])

    tokens = tokens.detach().cpu().tolist()
    tokens_list = []
    for seq in tokens:
        i = 0
        while i < len(seq) and seq[i] not in end_tokens:
            i += 1
        tokens_list.append(seq[0:i])

    texts = tk.batch_decode(tokens_list, skip_special_tokens=True)
    return texts, tokens_list


def content_mask(runner: DiffusionRunner, input_ids: torch.Tensor) -> torch.Tensor:
    """Позиции с настоящими токенами: без паддинга и без [CLS]/[SEP].

    Спецтокены Encoder подменяет на фиксированные векторы, одинаковые у всех
    примеров. Если их не выбросить, любая косинусная статистика окажется
    завышена за счет позиций, которые модель не предсказывает.
    """
    special = set(runner.tokenizer.all_special_ids)
    mask = torch.ones_like(input_ids, dtype=torch.bool)
    for sp in special:
        mask &= input_ids != sp
    return mask


def masked_cosine(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Косинус между a и b по каждому примеру, только по значимым позициям.

    a, b -- [B, L, D]; mask -- [B, L] булева. Возвращает [B].
    """
    m = mask.unsqueeze(-1).to(a.dtype)
    a = a * m
    b = b * m
    num = (a * b).flatten(1).sum(dim=1)
    den = a.flatten(1).norm(dim=1) * b.flatten(1).norm(dim=1)
    return num / den.clamp_min(1e-12)


def masked_norm(a: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).to(a.dtype)
    return (a * m).flatten(1).norm(dim=1)


def t_grid(n_points: int, t_max: float = 1.0, t_min: float = EPS_T) -> List[float]:
    """Равномерная сетка t для диагностик уровня шума."""
    if n_points < 2:
        raise ValueError("нужно минимум две точки сетки")
    step = (t_max - t_min) / (n_points - 1)
    return [t_min + step * i for i in range(n_points)]


def unigram_f1(a_tokens: List[int], b_tokens: List[int]) -> float:
    """F1 по совпадению токенов с кратностью.

    Для кривой предрешенности нужна мера «насколько это тот же самый текст».
    BLEU здесь плох: на коротких продолжениях он шумит и обнуляется из-за
    отсутствия 4-грамм. Униграммный F1 с кратностью ведет себя гладко и равен
    1.0 ровно при совпадении мультимножеств.
    """
    if not a_tokens and not b_tokens:
        return 1.0
    if not a_tokens or not b_tokens:
        return 0.0
    from collections import Counter
    ca, cb = Counter(a_tokens), Counter(b_tokens)
    overlap = sum((ca & cb).values())
    if overlap == 0:
        return 0.0
    p = overlap / len(a_tokens)
    r = overlap / len(b_tokens)
    return 2 * p * r / (p + r)


def dump_json(obj, path: str) -> None:
    import json
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print(f"[PROBE] результат сохранен: {path}")
