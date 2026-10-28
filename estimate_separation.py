"""Разнесённость мод против разброса внутри моды, по кэшу банка латентов.

ЗАЧЕМ
-----
Выкладка в THEORY.md даёт условие, при котором центроид обязан проигрывать
моде:

    Delta^2 / (4 sigma^2)  >~  ln K

где Delta -- расстояние между центрами мод, sigma -- ПОКООРДИНАТНАЯ ширина
моды. Выполняется ли оно в нашем латенте, мы не знали: Delta и sigma никто не
мерил. Без этих чисел §7 статьи остаётся моделью без привязки к данным, и
первый же рецензент спросит.

Считается по уже сохранённому банку: ни GPU, ни генерации не нужно.

ЧТО ИМЕННО МЕРЯЕТСЯ И ПОЧЕМУ ТАК
--------------------------------
1. Кластеры строятся ТЕМ ЖЕ кодом и тем же сидом, что в probe (pool_sequences
   + kmeans), иначе числа не будут относиться к измеренной оси.

2. Delta и разброс считаются в ПОЛНОМ пространстве последовательностей
   (L x 768 развёрнуто), потому что ось z(beta) живёт именно там, а
   кластеризация по усреднённым векторам -- лишь способ назначить метку.

3. sigma НЕ равна среднему расстоянию до центра. Для N(mu, sigma^2 I_d)
   выполнено E||x-mu||^2 = d sigma^2, то есть sigma = RMS / sqrt(d). Брать
   RMS за sigma -- ошибка в sqrt(d) раз, а d здесь десятки тысяч.

4. Вместо номинального d берётся ЭФФЕКТИВНАЯ размерность разброса:

       d_eff = (sum lambda_i)^2 / sum lambda_i^2

   по собственным числам внутрикластерной ковариации. Латенты не изотропны:
   почти вся дисперсия сидит в небольшом числе направлений, и подстановка
   номинальных 49152 завысила бы разделённость на порядки. Собственные числа
   берутся из матрицы Грама n_k x n_k, а не из ковариации d x d.

ЗАПУСК
------
    python estimate_separation.py probe_results/latent_bank-wikipedia-*.pt
    python estimate_separation.py <файл> --levels 16 64
"""
import argparse
import glob
import math
import sys
from typing import List

import torch

sys.path.insert(0, ".")
from probes.degrade import kmeans, pool_sequences


def spectrum_stats(centered: torch.Tensor, dim: int, max_samples: int = 3000,
                   seed: int = 0) -> dict:
    """Спектр внутрикластерного разброса: d_eff и доли дисперсии.

    centered: [n, d] -- остатки «член минус центроид своего кластера»,
    собранные по ВСЕМ кластерам сразу. Так и надо: модель смеси в THEORY.md
    предполагает общую ширину sigma у всех мод, а объединение остатков даёт
    на порядок больше точек, чем один кластер.

    ВАЖНО ПРО СМЕЩЕНИЕ. participation ratio
        PR = (sum lambda)^2 / sum lambda^2
    при n <~ d меряет не данные, а нехватку точек: для ИЗОТРОПНОГО шума
    спектр Марченко--Пастура даёт E[lambda]=s^2, E[lambda^2]=s^4(1+gamma)
    при gamma = d/n, то есть PR -> d/(1+gamma). На синтетике с d=128 и n=120
    это проверено: истинное d_eff=128, PR=61.1, d/(1+gamma)=61.9.

    Поэтому печатаем ОБА числа: сырой PR (заниженная оценка) и поправленный
    PR*(1+gamma) -- он точен в изотропном случае и служит оценкой сверху.
    Плюс доли дисперсии в ведущих направлениях: они от числа точек почти не
    зависят и говорят об анизотропии напрямую.
    """
    n = centered.shape[0]
    if n < 2:
        return {"pr": float("nan"), "pr_corrected": float("nan"), "top": {}}
    if n > max_samples:
        g = torch.Generator().manual_seed(seed)
        centered = centered[torch.randperm(n, generator=g)[:max_samples]]
        n = max_samples

    gram = (centered @ centered.T).double()
    ev = torch.linalg.eigvalsh(gram).clamp_min(0.0).flip(0)  # по убыванию
    s1, s2 = ev.sum(), (ev ** 2).sum()
    if s2 <= 0:
        return {"pr": float("nan"), "pr_corrected": float("nan"), "top": {}}

    pr = float(s1 * s1 / s2)
    gamma = dim / n
    top = {k: float(ev[:k].sum() / s1) for k in (10, 50, 200) if k < len(ev)}
    return {"pr": pr, "pr_corrected": pr * (1.0 + gamma), "top": top,
            "n_used": n, "gamma": gamma}


def analyse(seq: torch.Tensor, level: int, seed: int = 0) -> dict:
    """seq: [N, L, D]. Возвращает сводку по уровню level."""
    n, L, D = seq.shape
    flat = seq.reshape(n, L * D)

    pooled = pool_sequences(seq)
    centers = kmeans(pooled, level, seed=seed)
    assign = torch.cdist(pooled, centers).argmin(dim=1)

    cents = []            # центроиды в ПОЛНОМ пространстве
    within_sq = []        # ||x - c||^2 по всем членам всех кластеров
    resid = []            # остатки для спектра, по всем кластерам сразу
    axis_len = []         # ||центроид - медоид||, длина нашей оси
    sizes = []

    for k in range(level):
        idx = (assign == k).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        members = flat[idx].double()
        c = members.mean(dim=0)
        cents.append(c)
        sizes.append(int(idx.numel()))

        diff = members - c
        within_sq.append((diff ** 2).sum(dim=1))

        if idx.numel() >= 2:
            resid.append(diff.float())
            # медоид -- ближайший к центроиду РЕАЛЬНЫЙ член, как в probe
            medoid = members[(diff ** 2).sum(dim=1).argmin()]
            axis_len.append(float((medoid - c).norm()))

    C = torch.stack(cents)
    within_sq = torch.cat(within_sq)
    rms_within = float(within_sq.mean().sqrt())

    pd = torch.cdist(C, C)
    iu = torch.triu_indices(len(C), len(C), offset=1)
    between = pd[iu[0], iu[1]]
    delta_min = float(between.min())
    delta_med = float(between.median())

    spec = spectrum_stats(torch.cat(resid), L * D) if resid else {}
    d_eff = spec.get("pr_corrected", float("nan"))
    sigma = rms_within / math.sqrt(d_eff) if d_eff == d_eff and d_eff > 0 else float("nan")

    return {
        "level": level,
        "clusters": len(cents),
        "min_size": min(sizes), "max_size": max(sizes),
        "nominal_dim": L * D,
        "d_eff": d_eff,
        "d_eff_raw": spec.get("pr", float("nan")),
        "top": spec.get("top", {}),
        "spec_n": spec.get("n_used"),
        "rms_within": rms_within,
        "sigma": sigma,
        "delta_min": delta_min,
        "delta_median": delta_med,
        "axis_len": float(torch.tensor(axis_len).mean()) if axis_len else float("nan"),
        "ratio_min": delta_min / sigma if sigma else float("nan"),
        "ratio_median": delta_med / sigma if sigma else float("nan"),
        "lhs_min": (delta_min ** 2) / (4 * sigma ** 2) if sigma else float("nan"),
        "lhs_median": (delta_med ** 2) / (4 * sigma ** 2) if sigma else float("nan"),
        "rhs": math.log(level),
    }


def report(path: str, levels: List[int]) -> None:
    blob = torch.load(path, map_location="cpu")
    seq = blob.get("sequences")
    if seq is None:
        raise SystemExit(f"{path}: в кэше нет последовательностей")
    meta = blob.get("meta", {})
    seq = seq.float()

    print("")
    print("=" * 72)
    print(f"банк: {path}")
    print(f"  последовательностей {seq.shape[0]}, длина {seq.shape[1]}, "
          f"измерений {seq.shape[2]}")
    if meta:
        print(f"  meta: {meta}")

    for level in levels:
        if level >= seq.shape[0]:
            print(f"\nК = {level}: пропущен, точек в банке всего {seq.shape[0]}")
            continue
        r = analyse(seq, level)
        ok_min = r["lhs_min"] > r["rhs"]
        ok_med = r["lhs_median"] > r["rhs"]
        print("")
        print(f"--- K = {level} ---")
        print(f"  кластеров {r['clusters']}, размеры {r['min_size']}..{r['max_size']}")
        print(f"  номинальная размерность      {r['nominal_dim']}")
        print(f"  d_eff сырой (PR)             {r['d_eff_raw']:.1f}   "
              f"<- занижен при n<~d, см. докстринг")
        print(f"  d_eff с поправкой МП         {r['d_eff']:.1f}   "
              f"<- используется ниже")
        if r["top"]:
            tops = "  ".join(f"top{k}={v:.3f}" for k, v in sorted(r["top"].items()))
            print(f"  доля дисперсии в ведущих:    {tops}  (точек {r['spec_n']})")
        print(f"  RMS расстояние внутри моды   {r['rms_within']:.4f}")
        print(f"  sigma (покоординатная)       {r['sigma']:.6f}")
        print(f"  Delta между центрами: мин    {r['delta_min']:.4f}")
        print(f"                        медиана {r['delta_median']:.4f}")
        print(f"  длина оси медоид->центроид   {r['axis_len']:.4f}")
        print(f"  Delta/sigma: мин {r['ratio_min']:.1f}, медиана {r['ratio_median']:.1f}")
        print("")
        print(f"  УСЛОВИЕ Delta^2/(4 sigma^2) > ln K = {r['rhs']:.2f}")
        print(f"    по минимальному Delta: {r['lhs_min']:.1f}  "
              f"-> {'ВЫПОЛНЕНО' if ok_min else 'НЕ ВЫПОЛНЕНО'}")
        print(f"    по медианному  Delta: {r['lhs_median']:.1f}  "
              f"-> {'ВЫПОЛНЕНО' if ok_med else 'НЕ ВЫПОЛНЕНО'}")
        print("")
        print(f"  запас по экспоненте: exp(-Delta^2(K-1)/(4K sigma^2)) * K = "
              f"{level * math.exp(-r['lhs_min'] * (level - 1) / level):.3e}")
        print("    (это оценка p(центроид)/p(мода) из THEORY.md, формула (2))")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="*", help="кэши банка; по умолчанию probe_results/latent_bank-*")
    ap.add_argument("--levels", type=int, nargs="+", default=[16, 64],
                    help="уровни K, те же что в свипах")
    args = ap.parse_args()
    paths = args.paths or sorted(glob.glob("probe_results/latent_bank-*"))
    if not paths:
        raise SystemExit("не найдено ни одного кэша банка латентов")
    for p in paths:
        report(p, args.levels)


if __name__ == "__main__":
    main()
