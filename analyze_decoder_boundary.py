"""Когда цель перестаёт декодироваться в тот же текст. По дампам, без GPU.

ЗАЧЕМ
-----
Замер разнесённости мод (estimate_separation.py) вскрыл расхождение, которое
нельзя оставить в статье необъяснённым: сферическая гауссова модель
предсказывает обвал качества при beta около 0.03, а наблюдается он при
0.75-0.85, то есть на порядок позже.

Проверка показала, что дело не в масштабе коэффициента, а в самом механизме.
Ось z(beta) = медоид + beta * (центроид - медоид) устроена так, что медоид --
ЧЛЕН кластера, значит (центроид - медоид) есть обычный внутрикластерный
остаток, и его длина равна типичной: 187.6 против RMS 211.8. То есть по beta
цель едет от типичного члена К ЦЕНТРУ СОБСТВЕННОГО кластера.

Отсюда две вещи, обе против прежнего объяснения:

  * по ПЛОТНОСТИ центр кластера -- не минимум, а максимум. Прежняя формула (5)
    считала центроид точкой «между модами», но эмпирически кластер один, и
    его центроид это его же вершина;
  * по ТИПИЧНОСТИ выход из тонкого слоя (относительная ширина 1/sqrt(2 d_eff)
    = 0.017) происходит сразу, при beta порядка 0.02.

Ни то, ни другое не даёт излома на 0.8. Остаётся третий кандидат: ДЕКОДЕР.
Он работает как классификатор по позициям над словарём; усреднение гасит
различающую компоненту вектора и оставляет общую, и предсказанный токен
меняется не плавно, а при ПЕРЕСЕЧЕНИИ границы решения. Порог даёт излом при
beta порядка единицы, а не 0.02, и объясняет, почему после излома всё
схлопывается в одно частотное слово: это ответ классификатора по умолчанию.

ЧТО СЧИТАЕТСЯ
-------------
Для каждого beta -- насколько декодированная ЦЕЛЬ (STEER) отличается от цели
при beta = 0. Это прямая кривая пересечения границы, и она не зависит ни от
генерации, ни от сэмплера.

Предсказания, которые эта кривая различает:
  плотность/типичность -> обвал согласия уже при beta ~ 0.02-0.04;
  граница декодера     -> согласие держится до beta ~ 0.5-0.7, затем обвал,
                          и одновременно падает длина текста.

ЗАПУСК
------
    python analyze_decoder_boundary.py
    python analyze_decoder_boundary.py --dataset rocstories
"""
import argparse
import glob
import json
import re
from typing import Dict, List

WORD = re.compile(r"[a-z0-9']+")


def tokens(text: str) -> List[str]:
    return WORD.findall(text.lower())


def f1(pred: List[str], ref: List[str]) -> float:
    """Униграммный F1 по мультимножествам."""
    if not pred or not ref:
        return 0.0
    counts: Dict[str, int] = {}
    for w in ref:
        counts[w] = counts.get(w, 0) + 1
    hit = 0
    for w in pred:
        if counts.get(w, 0) > 0:
            counts[w] -= 1
            hit += 1
    if hit == 0:
        return 0.0
    p, r = hit / len(pred), hit / len(ref)
    return 2 * p * r / (p + r)


def load(paths: List[str], dataset: str) -> List[dict]:
    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            blob = json.load(f)
        if blob.get("degrade") != "offmanifold":
            continue
        if dataset and blob.get("dataset") != dataset:
            continue
        texts = blob.get("texts")
        if not texts:
            # прогоны до правки, кладущей все тексты: по двадцати примерам
            # кривую не построить, и молча считать по ним нельзя
            rows.append({"level": blob.get("level"), "dataset": blob.get("dataset"),
                         "steer": None, "path": path})
            continue
        rows.append({
            "level": blob.get("level"),
            "dataset": blob.get("dataset"),
            "steer": [t["STEER"] for t in texts],
            "bert": blob.get("metrics", {}).get("bert-score"),
            "path": path,
        })
    return rows


def report(rows: List[dict], dataset: str) -> None:
    rows = [r for r in rows if r["dataset"] == dataset]
    if not rows:
        return
    rows.sort(key=lambda r: r["level"] if r["level"] is not None else -1)

    base = next((r for r in rows if r["level"] == 0), None)
    if base is None or base["steer"] is None:
        print(f"\n{dataset}: нет прогона beta = 0 с полными текстами, "
              f"сравнивать не с чем")
        return
    ref = base["steer"]

    print("")
    print("=" * 76)
    print(f"=== {dataset}: когда ЦЕЛЬ перестаёт декодироваться в тот же текст ===")
    print("")
    print("  beta   совпали  F1 с beta=0   слов в цели   доля целей <=2 слов   bert")
    for r in rows:
        if r["steer"] is None:
            print(f"  {str(r['level']):<6} — прогон без полных текстов ({r['path']})")
            continue
        cur = r["steer"]
        n = min(len(cur), len(ref))
        exact = sum(1 for i in range(n) if cur[i].strip() == ref[i].strip()) / n
        agree = sum(f1(tokens(cur[i]), tokens(ref[i])) for i in range(n)) / n
        lens = [len(tokens(t)) for t in cur[:n]]
        mean_len = sum(lens) / n
        short = sum(1 for x in lens if x <= 2) / n
        bert = "—" if r["bert"] is None else f"{r['bert']:.5f}"
        print(f"  {str(r['level']):<6} {exact:7.3f}  {agree:11.3f}  {mean_len:11.1f}  "
              f"{short:19.3f}  {bert:>7}")

    print("")
    print("  КАК ЧИТАТЬ. Плотность и типичность предсказывают обвал согласия уже")
    print("  при beta около 0.02-0.04. Граница декодера предсказывает плато до")
    print("  beta порядка 0.5-0.7 и обвал вместе с падением длины: усреднённый")
    print("  вектор пересекает границу решения и уходит в частотный токен.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--dataset", default="")
    args = ap.parse_args()
    paths = args.paths or sorted(
        glob.glob("probe_results/information_axis-*offmanifold*.json"))
    if not paths:
        raise SystemExit("не найдено ни одного json оси")
    rows = load(paths, args.dataset)
    if not rows:
        raise SystemExit("подходящих прогонов не нашлось")
    for ds in sorted({r["dataset"] for r in rows}):
        report(rows, ds)


if __name__ == "__main__":
    main()
