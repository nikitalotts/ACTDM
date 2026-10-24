"""Проверка, что подходы оценены на ОДНИХ И ТЕХ ЖЕ примерах.

ЗАЧЕМ
-----
Таблица статьи сравнивает четыре подхода по средним метрикам. Это корректно
только если все они считались на одном наборе примеров. Проверка саданных
дампов generated_texts показала, что это НЕ так:

    genie vs diffuseq      позиционно 5000/5000   пересечение 4999
    genie vs uncond        позиционно 5000/5000   пересечение 4999
    genie vs guidance      позиционно 1500/5000   пересечение 4500

То есть genie, diffuseq и безусловная модель сравнимы между собой полностью, а
прогоны guidance шли по другому порядку и на 10% другом множестве примеров.

Причина в том, как устроен сбор генераций:

  * data/dataset_wiki.py:spilt_data_across_gpu режет тестовый сплит на ранги
    срезом [rank*(N//W) : (rank+1)*(N//W)], последний ранг добирает остаток;
  * diffusion_holder.py:1069 останавливает ранг по условию
    len(TRG) >= num_gen_texts // world_size, то есть на ГРАНИЦЕ БАТЧА за этим
    порогом;
  * create_config.py:275 выставляет guidance validation.batch_size=128 вместо
    500 (на guidance нужна память под градиенты классификатора).

Разный размер батча и разное число карт двигают границу остановки у каждого
ранга, а потом конкатенация обрезается до num_gen_texts -- и состав набора
получается разным.

НАСКОЛЬКО ЭТО ВАЖНО
-------------------
На выводы не влияет: при n=5000 и разбросе метрики по примеру около 0.08 замена
500 примеров сдвигает среднее примерно на 0.0005, тогда как измеренные эффекты
-- сотые доли. Все замеры probe_* сравнимы между собой безусловно: они ходят по
своему загрузчику (probes/common.py:probe_loader), одной картой и одним батчем.

Но для таблицы статьи этого недостаточно: рецензент вправе потребовать, чтобы
подходы считались на одних примерах, и проверять это надо ДО прогона на 20
сидов, а не после. Поэтому ФИНАЛЬНАЯ таблица гоняется с одинаковыми
validation.batch_size и числом карт у всех подходов, а затем прогоняется этот
скрипт.

ЗАПУСК
------
    python check_eval_alignment.py generated_texts/<A>/87500-*.json \\
                                  generated_texts/<B>/87500-*.json ...

Код возврата 1, если наборы не совпали позиционно: это значит, что таблицу
строить нельзя.
"""
import json
import os
import sys


def load_refs(path):
    """Список референсов (TRG) в порядке оценки."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    # estimate пишет dict {датасет: [строки]}, а не список: на этом уже падал
    # probe_recon_guidance с 'unhashable type: slice'.
    if isinstance(data, dict):
        rows = []
        for key in sorted(data):
            rows += data[key]
    else:
        rows = data
    return [r["TRG"] if isinstance(r, dict) else r for r in rows]


def main(paths):
    if len(paths) < 2:
        raise SystemExit("нужно минимум два дампа для сравнения")

    refs = {}
    for path in paths:
        if not os.path.exists(path):
            raise SystemExit(f"нет файла: {path}")
        refs[path] = load_refs(path)

    width = max(len(os.path.basename(os.path.dirname(p))) for p in paths)
    print("=" * 72)
    print("СОСТАВ НАБОРА ОЦЕНКИ")
    print("=" * 72)
    for path, trg in refs.items():
        name = os.path.basename(os.path.dirname(path))
        print(f"  {name:<{width}}  примеров {len(trg):5d}  "
              f"уникальных {len(set(trg)):5d}")

    print()
    bad = []
    items = list(refs.items())
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            (pa, a), (pb, b) = items[i], items[j]
            na = os.path.basename(os.path.dirname(pa))
            nb = os.path.basename(os.path.dirname(pb))
            n = min(len(a), len(b))
            pos = sum(1 for k in range(n) if a[k] == b[k])
            overlap = len(set(a) & set(b))
            mark = "ok " if pos == n and len(a) == len(b) else "РАЗНО"
            print(f"  {mark} {na} vs {nb}: позиционно {pos}/{n}, "
                  f"пересечение множеств {overlap}")
            if mark != "ok ":
                bad.append((na, nb, pos, n, overlap))

    print("=" * 72)
    if bad:
        print("НАБОРЫ НЕ СОВПАЛИ. Таблицу по этим дампам строить нельзя:")
        print("средние посчитаны на разных примерах, и разница между подходами")
        print("частично объясняется разницей наборов, а не методами.")
        print()
        print("Что сделать: перегнать eval всех подходов с одинаковыми")
        print("validation.batch_size и числом карт, затем повторить проверку.")
        print("Для guidance размер батча задается GUIDANCE_EVAL_BS")
        print("(create_config.py:275, по умолчанию 128 против 500 у остальных).")
        return 1

    print("Наборы совпадают позиционно -- подходы сравнимы.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
