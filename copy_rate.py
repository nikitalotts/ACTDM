"""Сколько генерация копирует из промпта.

Зачем. diffuseq кладёт латенты промпта прямо в последовательность (latent
replacement), и дословное копирование кусков промпта у него заметно чаще, чем
у остальных подходов. Копирование надувает bleu и rouge: n-граммы промпта
часто встречаются и в референсном продолжении. Пока доля скопированного не
посчитана, разница между подходами по bleu/rouge НЕ интерпретируется -- нельзя
отличить «модель лучше продолжает» от «модель больше копирует».

Почему здесь две колонки, а не одна. Доля n-грамм генерации, встречающихся в
SOURCE, сама по себе ни о чём не говорит: референсное продолжение тоже
переиспользует слова промпта, это нормальный язык. Поэтому рядом считается то
же самое для TRG -- это пол, относительно которого и надо смотреть. Ровно та
же ошибка, что с div и ppl у unconditional: абсолютные числа без референса
не читаются.

Считает ОФЛАЙН по уже сохранённым текстам, GPU не нужен -- запускать на
login-ноде. Файлы берутся из generated_texts/<prefix>/<шаг>-seed=*.json,
которые пишет estimate() на каждом eval.

    python copy_rate.py --step 87500
    python copy_rate.py --step 87500 --out copy_rate-87500.json

У guidance в одном каталоге лежит несколько json на один шаг (разные gs,
схемы классификатора, time_scale) -- каждый идёт отдельной строкой таблицы.
"""
import argparse
import json
import os
import re
from statistics import mean

WORD = re.compile(r"\w+", re.UNICODE)


def tokenize(text):
    """Слова в нижнем регистре. Не wordpiece и не BPE: сравниваем подходы между
    собой, а не с токенизатором, и границы слов тут честнее."""
    return WORD.findall((text or "").lower())


def ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def copy_rate(gen_tokens, src_tokens, n):
    """Доля n-грамм генерации, которые встречаются в промпте.

    Считается с кратностью: каждая позиция генерации голосует отдельно, иначе
    повторяющийся скопированный кусок учитывался бы один раз.
    """
    g = ngrams(gen_tokens, n)
    if not g:
        return None
    s = set(ngrams(src_tokens, n))
    return sum(1 for x in g if x in s) / len(g)


def longest_copied_span(gen_tokens, src_tokens):
    """Длина самого длинного куска генерации, дословно взятого из промпта.

    Средняя доля n-грамм не отличает «везде по чуть-чуть» от «один огромный
    кусок целиком». Для latent replacement важно именно второе.
    """
    best = 0
    for k in range(1, len(gen_tokens) + 1):
        src_k = set(ngrams(src_tokens, k))
        if not src_k:
            break
        if any(tuple(gen_tokens[i:i + k]) in src_k
               for i in range(len(gen_tokens) - k + 1)):
            best = k
        else:
            # k-граммы нет -> (k+1)-граммы тем более нет, она её содержит
            break
    return best


def analyse(pairs, max_n=4):
    """pairs -- список словарей с ключами SRC/TRG/GEN."""
    rates = {("GEN", n): [] for n in range(1, max_n + 1)}
    rates.update({("TRG", n): [] for n in range(1, max_n + 1)})
    spans = {"GEN": [], "TRG": []}

    for item in pairs:
        src = tokenize(item.get("SRC"))
        if not src:
            continue
        for who in ("GEN", "TRG"):
            toks = tokenize(item.get(who))
            if not toks:
                continue
            for n in range(1, max_n + 1):
                r = copy_rate(toks, src, n)
                if r is not None:
                    rates[(who, n)].append(r)
            spans[who].append(longest_copied_span(toks, src))

    out = {"n_examples": len(spans["GEN"])}
    for who in ("GEN", "TRG"):
        out[who] = {
            **{f"copy_{n}gram": (mean(rates[(who, n)]) if rates[(who, n)] else None)
               for n in range(1, max_n + 1)},
            "longest_span": mean(spans[who]) if spans[who] else None,
        }
    # то, ради чего всё затевалось: насколько генерация копирует БОЛЬШЕ,
    # чем копирует сам референс
    out["excess"] = {
        f"copy_{n}gram": (out["GEN"][f"copy_{n}gram"] - out["TRG"][f"copy_{n}gram"])
        if out["GEN"][f"copy_{n}gram"] is not None
        and out["TRG"][f"copy_{n}gram"] is not None else None
        for n in range(1, max_n + 1)
    }
    if out["GEN"]["longest_span"] is not None and out["TRG"]["longest_span"] is not None:
        out["excess"]["longest_span"] = out["GEN"]["longest_span"] - out["TRG"]["longest_span"]
    return out


def find_files(texts_dir, step):
    """Все json нужного шага по всем подходам. У guidance их несколько на шаг."""
    found = []
    if not os.path.isdir(texts_dir):
        raise FileNotFoundError(f"Нет каталога {texts_dir}")
    for prefix in sorted(os.listdir(texts_dir)):
        folder = os.path.join(texts_dir, prefix)
        if not os.path.isdir(folder):
            continue
        for name in sorted(os.listdir(folder)):
            if name.startswith(f"{step}-") and name.endswith(".json"):
                # statistical_eval-*.json -- это агрегаты метрик, не тексты
                if name.startswith("statistical_eval"):
                    continue
                found.append((prefix, name, os.path.join(folder, name)))
    return found


def label(prefix, name):
    """Короткое имя строки: подход плюс то, чем файлы одного шага различаются.

    Подход берётся из каталога, а guidance -- из имени файла: три схемы
    классификатора делят checkpoints_prefix с unconditional, и различает их
    только суффикс gs/схема/time_scale.
    """
    if prefix.startswith("gpt2"):
        arch = "gpt"
    else:
        tail = prefix.split("wikipedia", 1)[-1]
        if "diffuseq" in tail:
            arch = "diffuseq"
        elif "uncond" in tail:
            arch = "uncond"
        else:
            arch = "genie"

    m = re.search(r"-gs=(\d+(?:\.\d+)?)(?:-([a-z]+)-ts(\d+(?:\.\d+)?))?", name)
    if m and float(m.group(1)) > 0:
        arch = f"guidance-{m.group(2) or '?'}"
        extra = [f"gs{m.group(1)}"]
        if m.group(3) and float(m.group(3)) != 1.0:
            extra.append(f"ts{m.group(3)}")
    else:
        extra = []

    # декодинг у gpt: сweep пишет его в имя файла, иначе строки неразличимы
    d = re.search(r"-(greedy|sampling(?:-[a-z0-9.]+)*)\.json$", name)
    if d:
        extra.append(d.group(1))
    return "-".join([arch] + extra)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", type=int, required=True,
                    help="шаг сравнения, например 87500")
    ap.add_argument("--texts-dir", default="generated_texts")
    ap.add_argument("--max-n", type=int, default=4)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    files = find_files(args.texts_dir, args.step)
    if not files:
        raise SystemExit(
            f"В {args.texts_dir} нет текстов шага {args.step}. "
            f"Проверьте, что eval на этом шаге отработал.")

    report = {}
    rows = []
    for prefix, name, path in files:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        pairs = []
        for ds_items in payload.values():
            pairs.extend(ds_items)
        if not pairs or "SRC" not in pairs[0]:
            print(f"[skip] {name}: нет SRC (безусловный режим)")
            continue
        res = analyse(pairs, max_n=args.max_n)
        key = label(prefix, name)
        report[key] = {"file": path, **res}
        rows.append((key, res))

    if not rows:
        raise SystemExit("Нечего считать: во всех файлах отсутствует SRC")

    width = max(len(k) for k, _ in rows) + 2
    print()
    print("Доля n-грамм, пришедших из промпта (GEN), и то же для референса (TRG).")
    print("'избыток' = GEN - TRG: насколько модель копирует БОЛЬШЕ, чем копирует")
    print("сам референс. Положительное и большое -- метрики надуты копированием.")
    print()
    head = (f"{'подход':<{width}}{'n':>3}{'GEN':>9}{'TRG':>9}{'избыток':>10}"
            f"{'макс.кусок GEN/TRG':>22}")
    print(head)
    print("-" * len(head))
    for key, res in sorted(rows, key=lambda kv: -(kv[1]["excess"].get("copy_4gram") or 0)):
        for n in range(1, args.max_n + 1):
            g = res["GEN"][f"copy_{n}gram"]
            t = res["TRG"][f"copy_{n}gram"]
            e = res["excess"][f"copy_{n}gram"]
            span = (f"{res['GEN']['longest_span']:.2f} / {res['TRG']['longest_span']:.2f}"
                    if n == 1 else "")
            print(f"{key if n == 1 else '':<{width}}{n:>3}"
                  f"{(g if g is not None else float('nan')):>9.4f}"
                  f"{(t if t is not None else float('nan')):>9.4f}"
                  f"{(e if e is not None else float('nan')):>10.4f}"
                  f"{span:>22}")
        print("-" * len(head))
    print(f"\nпримеров на подход: {rows[0][1]['n_examples']}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Сырые числа: {args.out}")


if __name__ == "__main__":
    main()
