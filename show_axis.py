"""Таблица оси информативности из уже снятых json. Ничего не считает.

ЗАЧЕМ
-----
Прогоны оси пишут в json больше, чем печатают в лог, и после каждого свипа
числа приходится собирать глазами по десяти файлам. Этот скрипт печатает ту
самую таблицу, которая идёт в статью, и делает это по дампам -- то есть
задним числом, без GPU и без пересъёмки.

Отдельно важен столбец MAUVE ЦЕЛИ: это ПРЯМАЯ мера главного утверждения
работы. Всё остальное меряет следствие -- насколько просело качество
генерации, -- а MAUVE цели против реальных текстов отвечает на сам вопрос:
осталась ли цель, к которой ведут сэмплер, текстом вообще.

ЗАПУСК
------
    python show_axis.py                      # всё, что лежит в probe_results
    python show_axis.py --dataset wikipedia  # только один датасет
"""
import argparse
import glob
import json
import os
from typing import List


def load(paths: List[str]) -> List[dict]:
    rows = []
    for path in paths:
        try:
            with open(path, encoding="utf-8") as f:
                blob = json.load(f)
        except (OSError, ValueError) as exc:
            print(f"пропущен {path}: {exc}")
            continue
        if blob.get("probe") != "information_axis":
            continue
        rows.append({
            "path": os.path.basename(path),
            "dataset": blob.get("dataset", "?"),
            "degrade": blob.get("degrade", "?"),
            "level": blob.get("level"),
            "shape": blob.get("set_shape"),
            "sigma": blob.get("set_sigma_frac"),
            "bert": blob.get("metrics", {}).get("bert-score"),
            "recovery": blob.get("target_recovery", {}).get("bert-score"),
            "fidelity": blob.get("target_fidelity", {}).get("bert-score"),
            "target_mauve": blob.get("target_mauve"),
            "gen_mauve": blob.get("gen_mauve"),
            "entropy": blob.get("set_entropy"),
        })
    return rows


def fmt(value, width=7):
    if value is None:
        return "—".rjust(width)
    return f"{value:.5f}".rjust(width)


def main(paths: List[str], dataset: str) -> None:
    rows = load(paths)
    if dataset:
        rows = [r for r in rows if r["dataset"] == dataset]
    if not rows:
        raise SystemExit("не найдено ни одного json оси информативности")

    missing = [r for r in rows if r["target_mauve"] is None]
    groups = sorted({(r["dataset"], r["degrade"]) for r in rows})

    for ds, mode in groups:
        part = [r for r in rows if r["dataset"] == ds and r["degrade"] == mode]
        # уровень -- число, и сортировать его как строку нельзя: иначе 100
        # встаёт между 0 и 25 и таблица читается задом наперёд
        part.sort(key=lambda r: (r["level"] if r["level"] is not None else -1,
                                 str(r["shape"]), r["sigma"] or 0.0))
        print("")
        print(f"=== {ds} / {mode} ===")
        print("  уровень  форма        bert     верность восстан.  MAUVE цели  MAUVE ген.")
        for r in part:
            shape = r["shape"] or ""
            if r["sigma"] is not None:
                shape = f"{shape} s{r['sigma']:g}"
            print(f"  {str(r['level']):<8} {shape:<12} {fmt(r['bert'])} "
                  f"{fmt(r['fidelity'])} {fmt(r['recovery'])}  "
                  f"{fmt(r['target_mauve'], 10)}  {fmt(r['gen_mauve'], 10)}")

    if missing:
        print("")
        print(f"ВНИМАНИЕ: в {len(missing)} прогонах MAUVE не посчитан — они сняты "
              f"кодом до добавления метрики либо на ноде без gpt2-large:")
        for r in missing[:10]:
            print(f"  {r['path']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="*", help="json-ы; по умолчанию probe_results/*")
    ap.add_argument("--dataset", default="", help="оставить только один датасет")
    args = ap.parse_args()
    main(args.paths or sorted(glob.glob("probe_results/information_axis-*.json")),
         args.dataset)
