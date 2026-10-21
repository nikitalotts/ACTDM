"""Сверка: имя файла чекпоинта против шага, записанного ВНУТРИ него.

Переименование файла не меняет его содержимого. Чекпоинт, переложенный под
другим именем, грузится молча, а `self.step` берётся из `load["step"]` --
то есть модель окажется не той, что обещает имя, и eval посчитает метрики
чужого шага. Именно так 20-сидовая стат-оценка gpt2-medium оказалась сделана
на шаге 50000 при файле `87500.pth`.

Читает только поле "step": веса не материализуются (torch.load с mmap), так
что проверка восьми четырёхгигабайтных файлов занимает секунды.

    python check_checkpoints.py
    python check_checkpoints.py --dir checkpoints/gpt2-medium-512-0.0002-wikipedia
"""
import argparse
import os
import re

import torch

NUMBERED = re.compile(r"^(\d+)\.pth$")


def read_step(path):
    """Шаг из чекпоинта, без загрузки весов."""
    try:
        load = torch.load(path, map_location="cpu", mmap=True)
    except (TypeError, RuntimeError):
        # mmap появился в torch 2.1 и работает только с новым форматом
        load = torch.load(path, map_location="cpu")
    return load.get("step")


def check_folder(folder):
    rows, bad = [], 0
    for name in sorted(os.listdir(folder)):
        m = NUMBERED.match(name)
        if not m and name != "last.pth":
            continue
        path = os.path.join(folder, name)
        try:
            step = read_step(path)
        except Exception as e:
            rows.append((name, None, f"не читается: {type(e).__name__}"))
            bad += 1
            continue

        if m:
            expected = int(m.group(1))
            ok = step == expected
            rows.append((name, step, "ok" if ok else f"ВНУТРИ ШАГ {step}, А НЕ {expected}"))
            bad += 0 if ok else 1
        else:
            # last.pth -- просто последний, его номер ничему не обязан
            rows.append((name, step, "последний, номера нет"))
    return rows, bad


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="checkpoints",
                    help="каталог чекпоинтов или его родитель")
    args = ap.parse_args()

    if not os.path.isdir(args.dir):
        raise SystemExit(f"Нет каталога {args.dir}")

    folders = [args.dir]
    if not any(NUMBERED.match(n) or n == "last.pth" for n in os.listdir(args.dir)):
        folders = [os.path.join(args.dir, d) for d in sorted(os.listdir(args.dir))
                   if os.path.isdir(os.path.join(args.dir, d))]

    total_bad = 0
    for folder in folders:
        rows, bad = check_folder(folder)
        if not rows:
            continue
        total_bad += bad
        print(f"\n{folder}")
        print("-" * len(folder))
        for name, step, note in rows:
            print(f"  {name:<16} step={str(step):<10} {note}")

    print()
    if total_bad:
        raise SystemExit(
            f"Расхождений: {total_bad}. Имя файла не соответствует шагу внутри -- "
            f"eval посчитает метрики не того шага, и сравнение подходов поедет.")
    print("Все номера совпадают с содержимым.")


if __name__ == "__main__":
    main()
