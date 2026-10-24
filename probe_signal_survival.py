"""Время жизни условного сигнала: accuracy(t) и информативность градиента.

ЗАЧЕМ ЭТОТ ПРОГОН
-----------------
Главное возражение к отрицательному результату по classifier guidance звучит
так: «у вас классификатор с valid accuracy 0.95 и ровно нулевой эффект на
генерации -- значит, вы что-то сломали». Ответить на него одной цифрой нельзя,
потому что 0.95 -- это усреднение по всему диапазону шума, который классификатор
видел на обучении по curriculum. Вопрос не в средней точности, а в том, НА КАКИХ
УРОВНЯХ ШУМА она набрана.

Прогон раскладывает точность по t и заодно меряет то, чем guidance пользуется на
самом деле, -- направление градиента. Точность -- это про то, может ли
классификатор отличить правильное продолжение от чужого. Guidance же двигает
латент по градиенту логита, и даже точный классификатор может давать градиент,
который никуда не указывает. Поэтому считаются обе величины:

  rank_acc(t)   -- доля примеров, где логит правильного продолжения выше, чем у
                   чужого при том же промпте и том же уровне шума. Мера
                   свободна от калибровки: порог не нужен.
  bal_acc(t)    -- сбалансированная точность по порогу 0 (как на валидации).
  auc(t)        -- площадь под ROC по всем накопленным логитам.
  cos_grad(t)   -- косинус между градиентом guidance и направлением, которое
                   реально ведет к истинному продолжению: mu_t * x_0 - x_t.
  cos_random(t) -- тот же косинус для случайного вектора. Это БАЗОВАЯ ЛИНИЯ, без
                   нее cos_grad не читается: в пространстве 64 x 768 случайный
                   косинус имеет масштаб 1/sqrt(49152) ~ 0.0045, и значение
                   0.01 само по себе ничего не значит.
  cos_score(t)  -- тот же косинус для безусловного score. Это ПОТОЛОК: показывает,
                   что измерение вообще работает и что сама диффузия в это
                   направление смотрит.
  rel_norm(t)   -- ||lambda * grad|| / ||score||: какую долю шага дает guidance.
  x0_shift(t)   -- ||sigma^2 * lambda * grad / mu|| / ||x_0||: на сколько
                   guidance сдвигает предсказание чистого состояния. Отличает
                   «guidance численно невидим» от «двигает, но не туда».

ЧТО ЭТО ДАЕТ СТАТЬЕ
-------------------
Величина t, на которой rank_acc падает до 0.5, -- это tau_signal, верхний край
окна управления. Вместе с кривой предрешенности из probe_commitment.py она дает
главный рисунок: интервал, где сигнал еще есть, и интервал, где траекторию еще
можно изменить, не пересекаются.

ЗАПУСК
------
    ARCH_TYPE=guidance AUG_SCHEME=augmented DATASET=wikipedia \
    CHECKPOINT_STEP=87500 ./probe_signal_survival.sh

Настройки через окружение (значения по умолчанию в скобках):
    PROBE_BATCHES     (8)    сколько батчей теста усреднять
    PROBE_BATCH_SIZE  (64)   размер батча; градиент идет через BERT-классификатор
    PROBE_T_POINTS    (21)   число точек сетки t
    PROBE_OUT         (...)  путь к json с результатом
"""
import os
import time

import torch
import torch.distributed as dist

from create_config import create_config
from probes.common import (
    build_probe_runner, content_mask, dump_json, encode_batch, masked_cosine,
    masked_norm, probe_loader, t_grid,
)
from utils.util import parse, set_seed


def _auc(pos_scores, neg_scores):
    """ROC AUC через ранги. Равен доле пар (pos, neg), где pos выше."""
    import numpy as np
    pos = np.asarray(pos_scores, dtype=np.float64)
    neg = np.asarray(neg_scores, dtype=np.float64)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, allv.size + 1)
    # средний ранг для связок, иначе AUC смещается на вырожденных логитах
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(counts.size)
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    r_pos = ranks[:pos.size].sum()
    return float((r_pos - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def main():
    args = parse()
    config = create_config(args)

    if not config.classifier_guidance:
        raise SystemExit(
            "probe_signal_survival измеряет классификатор guidance -- "
            "запускайте с ARCH_TYPE=guidance (и AUG_SCHEME=...)")

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    config.local_rank = rank
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", init_method="env://",
                            world_size=world_size, rank=rank)
    dist.barrier()
    config.training.batch_size_per_gpu = config.training.batch_size // dist.get_world_size()

    n_batches = int(os.environ.get("PROBE_BATCHES", "8"))
    batch_size = int(os.environ.get("PROBE_BATCH_SIZE", "64"))
    n_points = int(os.environ.get("PROBE_T_POINTS", "21"))
    out_path = os.environ.get(
        "PROBE_OUT",
        f"probe_results/signal_survival-{config.data.datasets.datasets_list[0]}"
        f"-{config.cond_encoder.augmentation_scheme}.json")

    set_seed(config.seed)
    runner = build_probe_runner(config)
    lam = float(runner.guidance_scale)

    if dist.get_rank() == 0:
        print("=" * 72)
        print("PROBE: время жизни условного сигнала")
        print("=" * 72)
        print(f"  чекпоинт диффузии : {runner.loaded_checkpoint_path}")
        print(f"  шаг               : {runner.step}")
        print(f"  классификатор     : {config.cond_encoder.cond_encoder_path}")
        print(f"  схема негативов   : {config.cond_encoder.augmentation_scheme}")
        print(f"  guidance scale    : {lam}")
        print(f"  батчей x размер   : {n_batches} x {batch_size}")
        print(f"  точек сетки t     : {n_points}")
        print("=" * 72)

    grid = t_grid(n_points)
    acc = {tv: {"rank": [], "pos_hit": [], "neg_hit": [], "pos": [], "neg": [],
                "cos_grad": [], "cos_rand": [], "cos_score": [],
                "rel_norm": [], "x0_shift": []} for tv in grid}

    loader = probe_loader(runner, split="test", batch_size=batch_size)
    started = time.time()

    for b_idx, batch in enumerate(loader):
        if b_idx >= n_batches:
            break
        batch, src_x, trg_x = encode_batch(runner, batch)
        mask = content_mask(runner, batch["input_ids_trg"])
        src_mask = batch["attention_mask_src"]
        bsz = trg_x.shape[0]

        # Негатив -- продолжение СОСЕДНЕГО примера при том же промпте и том же
        # уровне шума. Так различие между парами не смешивается с различием в
        # зашумленности: классификатор обязан сказать «это продолжение не к
        # этому промпту», а не «здесь больше шума».
        neg_trg = torch.roll(trg_x, shifts=1, dims=0)

        for tv in grid:
            t = tv * torch.ones(bsz, device=runner.device)
            x_t = runner.dynamic.marginal(trg_x, t)["x_t"]
            x_t_neg = runner.dynamic.marginal(neg_trg, t)["x_t"]

            with torch.no_grad():
                logit_pos = runner.cond_encoder(
                    src_embeds=src_x, noisy_trg_embeds=x_t, t=t, src_mask=src_mask)
                logit_neg = runner.cond_encoder(
                    src_embeds=src_x, noisy_trg_embeds=x_t_neg, t=t, src_mask=src_mask)

            acc[tv]["rank"] += (logit_pos > logit_neg).float().cpu().tolist()
            acc[tv]["pos_hit"] += (logit_pos > 0).float().cpu().tolist()
            acc[tv]["neg_hit"] += (logit_neg < 0).float().cpu().tolist()
            acc[tv]["pos"] += logit_pos.detach().cpu().tolist()
            acc[tv]["neg"] += logit_neg.detach().cpu().tolist()

            # --- что видит сам guidance -------------------------------------
            with torch.enable_grad():
                grad = runner.compute_classifier_guidance(
                    x_t=x_t, cond_x=src_x, t=t, cond_mask=src_mask)

            params = runner.dynamic.marginal_params(t)
            mu = params["mu"].view(-1, 1, 1)
            std = params["std"].view(-1, 1, 1)
            dir_true = mu * trg_x - x_t

            # Безусловный score той же сети. use_guidance гасится на время
            # вызова: иначе calc_score подмешал бы в него градиент
            # классификатора, и «потолок» оказался бы не безусловным.
            was = runner.use_guidance
            runner.use_guidance = False
            try:
                with torch.no_grad():
                    out = runner.calc_score(
                        model=runner.ddp_score_estimator, x_t=x_t, t=t,
                        cond=None, attention_mask=None, cond_mask=None,
                        x_0_self_cond=torch.zeros_like(x_t))
            finally:
                runner.use_guidance = was
            score = out["score"]

            rnd = torch.randn_like(grad)
            acc[tv]["cos_grad"] += masked_cosine(grad, dir_true, mask).cpu().tolist()
            acc[tv]["cos_rand"] += masked_cosine(rnd, dir_true, mask).cpu().tolist()
            acc[tv]["cos_score"] += masked_cosine(score, dir_true, mask).cpu().tolist()

            n_guid = masked_norm(lam * grad, mask)
            n_score = masked_norm(score, mask).clamp_min(1e-12)
            acc[tv]["rel_norm"] += (n_guid / n_score).cpu().tolist()

            shift = (std ** 2) * lam * grad / mu
            n_shift = masked_norm(shift, mask)
            n_x0 = masked_norm(trg_x, mask).clamp_min(1e-12)
            acc[tv]["x0_shift"] += (n_shift / n_x0).cpu().tolist()

        if dist.get_rank() == 0:
            print(f"  батч {b_idx + 1}/{n_batches} готов "
                  f"({time.time() - started:.0f} c)")

    def mean(xs):
        return float(sum(xs) / len(xs)) if xs else float("nan")

    rows = []
    for tv in grid:
        a = acc[tv]
        rows.append({
            "t": round(tv, 4),
            "rank_acc": mean(a["rank"]),
            "bal_acc": 0.5 * (mean(a["pos_hit"]) + mean(a["neg_hit"])),
            "auc": _auc(a["pos"], a["neg"]),
            "cos_grad": mean(a["cos_grad"]),
            "cos_random": mean(a["cos_rand"]),
            "cos_score": mean(a["cos_score"]),
            "rel_norm": mean(a["rel_norm"]),
            "x0_shift": mean(a["x0_shift"]),
            "n": len(a["rank"]),
        })

    if dist.get_rank() == 0:
        print("\n" + "=" * 108)
        print(f"{'t':>6} {'rank_acc':>9} {'bal_acc':>8} {'auc':>7} "
              f"{'cos_grad':>10} {'cos_rand':>10} {'cos_score':>10} "
              f"{'rel_norm':>9} {'x0_shift':>9}")
        print("-" * 108)
        for r in rows:
            print(f"{r['t']:>6.3f} {r['rank_acc']:>9.4f} {r['bal_acc']:>8.4f} "
                  f"{r['auc']:>7.4f} {r['cos_grad']:>10.5f} "
                  f"{r['cos_random']:>10.5f} {r['cos_score']:>10.5f} "
                  f"{r['rel_norm']:>9.4f} {r['x0_shift']:>9.5f}")
        print("=" * 108)

        # tau_signal -- наибольшее t, при котором rank_acc еще выше порога.
        # Порог 0.55, а не 0.5: при n ~ 500 стандартная ошибка доли ~0.022,
        # и 0.55 это примерно +2 сигмы от случайного угадывания.
        above = [r["t"] for r in rows if r["rank_acc"] >= 0.55]
        tau_signal = max(above) if above else None
        print(f"\ntau_signal (rank_acc >= 0.55): {tau_signal}")
        print("Выше этого уровня шума классификатор не отличает правильное "
              "продолжение от чужого,\nто есть guidance там не несет информации "
              "ни при каком lambda.\n")

        dump_json({
            "probe": "signal_survival",
            "dataset": config.data.datasets.datasets_list[0],
            "architecture_type": config.architecture_type,
            "augmentation_scheme": config.cond_encoder.augmentation_scheme,
            "classifier_path": config.cond_encoder.cond_encoder_path,
            "diffusion_checkpoint": runner.loaded_checkpoint_path,
            "step": runner.step,
            "guidance_scale": lam,
            "time_scale": float(config.cond_encoder.time_scale),
            "n_batches": n_batches,
            "batch_size": batch_size,
            "tau_signal_rank055": tau_signal,
            "rows": rows,
        }, out_path)


if __name__ == "__main__":
    main()
