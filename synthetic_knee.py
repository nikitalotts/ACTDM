"""Контролируемая проверка: что задаёт положение излома. Без GPU.

ЗАЧЕМ
-----
В реальном латенте мы не управляем ни числом мод, ни их разнесённостью,
поэтому два объяснения излома (плотность против границы решения декодера)
там не различить: обе кривые качества выглядят одинаково. Здесь мы строим
игрушечное пространство, где Delta, sigma и K заданы руками, и смотрим, от
чего положение излома ЗАВИСИТ, а от чего нет.

РАЗЛИЧАЮЩЕЕ ПРЕДСКАЗАНИЕ
------------------------
Ось та же: z(beta) = медоид + beta * (центроид - медоид).

  * Если излом задаёт ПЛОТНОСТЬ, его положение определяется отношением
    длины оси к ширине моды: beta* ~ sigma / D. Уменьшили sigma втрое --
    излом обязан сдвинуться к нулю втрое.

  * Если излом задаёт ГРАНИЦА РЕШЕНИЯ декодера, он стоит там, где
    усреднённый вектор перестаёт быть ближе к своему кодовому слову, чем к
    чужому. Это середина отрезка между кодовыми словами, то есть beta*
    близко к постоянной и от sigma почти НЕ ЗАВИСИТ.

Устройство игрушки намеренно повторяет наш случай: «последовательность» это
L позиций, в каждой -- кодовое слово из словаря, а «декодер» это argmax по
ближайшему кодовому слову, то есть классификатор по позициям.

ЗАПУСК
------
    python synthetic_knee.py
    python synthetic_knee.py --sigmas 0.02 0.05 0.1 0.2 --modes 2 4 8
"""
import argparse
import math
from typing import List

import torch


def build(vocab: int, dim: int, length: int, modes: int, per_mode: int,
          sigma: float, seed: int) -> dict:
    """Кластер из `modes` подмод; каждая подмода -- свой набор кодовых слов."""
    g = torch.Generator().manual_seed(seed)
    # кодовые слова на сфере: попарное расстояние примерно одинаково
    code = torch.randn(vocab, dim, generator=g)
    code = code / code.norm(dim=1, keepdim=True)

    seqs, ids = [], []
    for m in range(modes):
        # у каждой подмоды свой «сюжет» -- фиксированная цепочка слов
        plot = torch.randint(vocab, (length,), generator=g)
        for _ in range(per_mode):
            ids.append(plot)
            seqs.append(code[plot] + sigma * torch.randn(length, dim, generator=g))
    return {"code": code, "seq": torch.stack(seqs), "ids": torch.stack(ids)}


def decode(x: torch.Tensor, code: torch.Tensor) -> torch.Tensor:
    """Ближайшее кодовое слово в каждой позиции. [L,D] -> [L]."""
    return torch.cdist(x, code).argmin(dim=1)


def knee(betas: List[float], agree: List[float], level: float = 0.5) -> float:
    """Первое beta, где согласие падает ниже level (линейная интерполяция)."""
    for i in range(1, len(agree)):
        if agree[i] < level <= agree[i - 1]:
            a, b = agree[i - 1], agree[i]
            t = (a - level) / max(a - b, 1e-12)
            return betas[i - 1] + t * (betas[i] - betas[i - 1])
    return float("nan")


def run(sigma: float, modes: int, args, seed: int = 0) -> dict:
    d = build(args.vocab, args.dim, args.length, modes, args.per_mode, sigma, seed)
    seq, code = d["seq"], d["code"]
    n = seq.shape[0]
    flat = seq.reshape(n, -1)

    centroid = flat.mean(dim=0)
    medoid = flat[(flat - centroid).pow(2).sum(dim=1).argmin()]
    ref = decode(medoid.view(args.length, args.dim), code)

    betas = [i / 20 for i in range(21)]
    agree = []
    for b in betas:
        z = (medoid + b * (centroid - medoid)).view(args.length, args.dim)
        agree.append(float((decode(z, code) == ref).float().mean()))

    D = float((centroid - medoid).norm())
    return {"sigma": sigma, "modes": modes, "D": D,
            "D_over_sigma": D / sigma, "betas": betas, "agree": agree,
            "knee": knee(betas, agree)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.02, 0.05, 0.1, 0.2])
    ap.add_argument("--modes", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--vocab", type=int, default=512)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--length", type=int, default=32)
    ap.add_argument("--per-mode", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    print("")
    print("=" * 72)
    print("ЧТО ЗАДАЁТ ПОЛОЖЕНИЕ ИЗЛОМА: контролируемая проверка")
    print("=" * 72)
    print(f"  словарь {args.vocab}, измерений {args.dim}, длина {args.length}, "
          f"по {args.per_mode} на подмоду, сидов {args.seeds}")
    print("")
    print("  подмод  sigma   D/sigma    излом beta*   (разброс по сидам)")
    rows = []
    for modes in args.modes:
        for sigma in args.sigmas:
            ks, dos = [], []
            for s in range(args.seeds):
                r = run(sigma, modes, args, seed=s)
                if r["knee"] == r["knee"]:
                    ks.append(r["knee"])
                dos.append(r["D_over_sigma"])
            if not ks:
                print(f"  {modes:<7} {sigma:<7.3f} {sum(dos)/len(dos):<10.1f} "
                      f"излома нет (согласие не падает ниже 0.5)")
                continue
            t = torch.tensor(ks)
            rows.append((modes, sigma, sum(dos) / len(dos), float(t.mean())))
            print(f"  {modes:<7} {sigma:<7.3f} {sum(dos)/len(dos):<10.1f} "
                  f"{t.mean():.3f}         +-{t.std():.3f}")

    print("")
    print("  КАК ЧИТАТЬ")
    for modes in args.modes:
        sub = [r for r in rows if r[0] == modes]
        if len(sub) < 2:
            continue
        lo, hi = sub[0], sub[-1]
        span_dos = hi[2] / lo[2] if lo[2] else float("nan")
        span_knee = hi[3] / lo[3] if lo[3] else float("nan")
        print(f"  подмод {modes}: D/sigma изменилось в {span_dos:.1f} раз, "
              f"излом -- в {span_knee:.2f} раз")
    print("")
    print("  Плотность предсказывает beta* ~ sigma/D, то есть излом обязан")
    print("  двигаться вместе с D/sigma. Граница решения декодера предсказывает")
    print("  beta* почти постоянным. Смотрите, какое из двух отношений ближе к 1.")


if __name__ == "__main__":
    main()
