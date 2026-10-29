"""Склейка статистических прогонов по сырым значениям. Без GPU.

ЗАЧЕМ
-----
Строка таблицы на 20 сидов стоит около 7,5 GPU-часов, и если часы кончатся
на середине, не остаётся ничего. Два задания по 10 сидов дают РОВНО тот же
набор сидов (BASE_SEED=0 и BASE_SEED=10000 при SEED_STEP=1000 -- это
0, 1000, ... , 19000), а здесь они склеиваются обратно.

Склейка ТОЧНАЯ, а не приближённая: `_aggregate_and_save` кладёт в json поле
"values" с сырым значением по каждому сиду, поэтому среднее, std и интервал
считаются с нуля по объединённому списку, а не выводятся из двух средних.

ЧЕГО СКРИПТ НЕ ДЕЛАЕТ
---------------------
Не склеивает то, что склеивать нельзя. Отказ (а не предупреждение) при:

  * разных чекпоинтах, шагах, числе шагов решателя или числе текстов;
  * разных настройках guidance (сила, схема аугментации, time_scale);
  * ПЕРЕСЕЧЕНИИ множеств сидов -- иначе один прогон попал бы в среднее дважды
    и интервал сузился бы на ровном месте;
  * числе значений, не совпадающем с заявленным num_seeds.

ЗАПУСК
------
    python merge_stat_eval.py A.json B.json
    python merge_stat_eval.py A.json B.json --out merged.json
"""
import argparse
import json
import math
import os
from typing import Dict, List

# Та же таблица, что в DiffusionRunner._aggregate_and_save. Расхождение между
# ними поймал бы тест test_merge_uses_the_same_t_table.
#
# ОСОБЕННОСТЬ, КОТОРУЮ НЕЛЬЗЯ ПРАВИТЬ С ОДНОЙ СТОРОНЫ. Для n = 20 нужное
# df = 19 в таблице отсутствует, и поиск берёт ближайшее МЕНЬШЕЕ df = 15, то
# есть 2.131 вместо точного 2.093. Интервал выходит на ~2 % шире нужного --
# осторожнее, а не наоборот. Если когда-нибудь уточнять, то в обоих местах
# сразу, иначе склейка и обычный прогон начнут давать разные интервалы.
T_CRIT_95 = {
    1: float("inf"), 2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776,
    6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262,
    15: 2.131, 20: 2.086, 25: 2.060, 30: 2.042,
}

# Поля, которые обязаны совпасть: иначе это разные эксперименты.
MUST_MATCH = ("checkpoint_prefix", "checkpoint_name", "split", "N_steps",
              "num_gen_texts", "step")


def t_crit(n: int) -> float:
    df = n - 1
    if df <= 0:
        return float("inf")
    if df in T_CRIT_95:
        return T_CRIT_95[df]
    if df > 30:
        try:
            from scipy.stats import t as _t
            return float(_t.ppf(0.975, df))
        except Exception:
            return 1.96
    keys = sorted(k for k in T_CRIT_95 if k <= df)
    return T_CRIT_95[keys[-1]] if keys else float("inf")


def stats_from_values(values: List[float]) -> dict:
    n = len(values)
    mean = sum(values) / n
    if n > 1:
        var = sum((x - mean) ** 2 for x in values) / (n - 1)
        std = math.sqrt(var)
        stderr = std / math.sqrt(n)
        ci_half = t_crit(n) * stderr
    else:
        std = stderr = ci_half = 0.0
    return {"mean": mean, "std": std, "stderr": stderr,
            "ci95_half_width": ci_half, "n": n,
            "values": [float(x) for x in values]}


def seeds_of(blob: dict) -> List[int]:
    base = int(blob["base_seed"])
    step = int(blob.get("seed_step", 1000))
    return [base + i * step for i in range(int(blob["num_seeds"]))]


def guidance_key(blob: dict, path: str):
    """Настройки guidance. У старых дампов их в json нет -- тогда из имени."""
    g = blob.get("guidance")
    if g is not None:
        return (bool(g.get("use_guidance")), float(g.get("scale", 0.0)),
                str(g.get("augmentation_scheme")), float(g.get("time_scale", 1.0)))
    name = os.path.basename(path)
    if "-gs=" not in name:
        return (False, 0.0, "<из имени: нет>", 1.0)
    tail = name.split("-gs=", 1)[1].rsplit(".json", 1)[0]
    parts = tail.split("-")
    scale = float(parts[0])
    scheme = parts[1] if len(parts) > 1 else "<неизвестно>"
    ts = float(parts[2][2:]) if len(parts) > 2 and parts[2].startswith("ts") else 1.0
    return (True, scale, scheme, ts)


def check(blobs: List[dict], paths: List[str]) -> None:
    for field in MUST_MATCH:
        seen = {json.dumps(b.get(field), ensure_ascii=False) for b in blobs}
        if len(seen) > 1:
            lines = "".join(f"\n  {b.get(field)!r}: {p}"
                            for b, p in zip(blobs, paths))
            raise SystemExit(
                f"поле {field!r} различается -- это разные эксперименты, "
                f"склеивать нельзя:{lines}")

    keys = [guidance_key(b, p) for b, p in zip(blobs, paths)]
    if len(set(keys)) > 1:
        lines = "".join(f"\n  {k}: {p}" for k, p in zip(keys, paths))
        raise SystemExit(f"настройки guidance различаются:{lines}")

    for blob, path in zip(blobs, paths):
        for ds, metrics in blob["metrics"].items():
            for name, val in metrics.items():
                for leaf, where in walk(val, name):
                    if leaf["n"] != int(blob["num_seeds"]):
                        raise SystemExit(
                            f"{path}: у {ds}/{where} значений {leaf['n']}, "
                            f"а заявлено num_seeds={blob['num_seeds']} -- "
                            f"дамп неполный")

    used: Dict[int, str] = {}
    for blob, path in zip(blobs, paths):
        for seed in seeds_of(blob):
            if seed in used:
                raise SystemExit(
                    f"сид {seed} встречается дважды:\n  {used[seed]}\n  {path}\n"
                    f"Склейка удвоила бы его вес и сузила интервал на ровном "
                    f"месте. Нужны задания с РАЗНЫМ base_seed.")
            used[seed] = path


def walk(val, where: str):
    """Листья статистики: метрика может быть словарём подметрик."""
    if isinstance(val, dict) and "values" in val:
        yield val, where
        return
    if isinstance(val, dict):
        for key, sub in val.items():
            for leaf, deeper in walk(sub, f"{where}.{key}"):
                yield leaf, deeper


def merge(blobs: List[dict]) -> dict:
    out: Dict[str, dict] = {}
    for ds in blobs[0]["metrics"]:
        out[ds] = {}
        for name, val in blobs[0]["metrics"][ds].items():
            if isinstance(val, dict) and "values" in val:
                pooled: List[float] = []
                for b in blobs:
                    pooled += b["metrics"][ds][name]["values"]
                out[ds][name] = stats_from_values(pooled)
            else:
                out[ds][name] = {}
                for sub in val:
                    pooled = []
                    for b in blobs:
                        pooled += b["metrics"][ds][name][sub]["values"]
                    out[ds][name][sub] = stats_from_values(pooled)
    return out


def report(merged: dict, blobs: List[dict], paths: List[str]) -> None:
    seeds = sorted(s for b in blobs for s in seeds_of(b))
    print("")
    print("=" * 80)
    print(f"СКЛЕЙКА: {len(paths)} прогонов, {len(seeds)} сидов")
    print("=" * 80)
    print(f"  чекпоинт : {blobs[0].get('checkpoint_name')}")
    print(f"  шаг      : {blobs[0].get('step')}")
    print(f"  guidance : {guidance_key(blobs[0], paths[0])}")
    print(f"  сиды     : {seeds[0]}..{seeds[-1]} ({len(seeds)} шт.)")
    for path, blob in zip(paths, blobs):
        print(f"    + {blob['num_seeds']:>3} сидов от {blob['base_seed']:<6} {path}")
    for ds in merged:
        print(f"\n[{ds}]")
        print("-" * 80)
        for name, val in merged[ds].items():
            if "mean" in val:
                print(f"  {name:<22} {val['mean']:.5f} ± {val['ci95_half_width']:.5f}"
                      f"   (std {val['std']:.5f}, n = {val['n']})")
                continue
            print(f"  {name}")
            for sub, leaf in val.items():
                print(f"    {sub:<20} {leaf['mean']:.5f} ± "
                      f"{leaf['ci95_half_width']:.5f}"
                      f"   (std {leaf['std']:.5f}, n = {leaf['n']})")
    print("")
    print("  Интервал посчитан по ОБЪЕДИНЁННЫМ сырым значениям, а не из двух")
    print("  средних, поэтому он такой же, как у одного задания на столько же")
    print("  сидов. В статье так и писать: n = сумма, без оговорок.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if len(args.paths) < 2:
        raise SystemExit("склеивать нечего: нужно хотя бы два дампа")

    blobs = []
    for path in args.paths:
        with open(path, encoding="utf-8") as f:
            blobs.append(json.load(f))
    check(blobs, args.paths)
    merged = merge(blobs)
    report(merged, blobs, args.paths)

    if args.out:
        payload = dict(blobs[0])
        payload["metrics"] = merged
        payload["num_seeds"] = sum(int(b["num_seeds"]) for b in blobs)
        payload["merged_from"] = list(args.paths)
        payload["seeds"] = sorted(s for b in blobs for s in seeds_of(b))
        payload.pop("timing", None)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\n  записано: {args.out}")


if __name__ == "__main__":
    main()
