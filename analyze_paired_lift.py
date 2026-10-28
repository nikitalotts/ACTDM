"""Парная статистика и условный прирост по дампам probe-прогонов.

ЗАЧЕМ
-----
Свип 28.10 дал у alpha=0.25 bert 0.51223 против 0.50813 у контроля, то есть
+0.0041. Непарный доверительный интервал при n=1000 и разбросе по примеру ~0.08
равен +-0.005: ПО АГРЕГАТУ ЭФФЕКТ НЕ ОТЛИЧИМ ОТ НУЛЯ. Но прогоны идут на одних
промптах и одном сиде, то есть сравнение парное, и его точность на порядок
выше. Парную разность нельзя посчитать из агрегата -- нужны значения по
примерам, и ради них probe теперь кладёт в json все тексты.

Второй вопрос, на который агрегат не отвечает: улучшилась ОБУСЛОВЛЕННОСТЬ или
просто беглость. При alpha=1, где условного сигнала нет вообще, bert вышел
0.46596 против пола 0.4605 -- значит само притягивание к данным слегка
поднимает беглость, и часть прироста у alpha=0.25 может быть ею. По bert их не
различить.

Различает УСЛОВНЫЙ ПРИРОСТ:

    lift = F1(GEN, своя TRG) - F1(GEN, чужая TRG)

Второе слагаемое -- та же генерация против переставленной по кругу цели. Всё,
что улучшает текст вообще (беглость, длина, частотные слова, копирование
промпта), поднимает оба слагаемых одинаково и из разности уходит. Остаётся
только то, что связывает генерацию С ЕЁ СОБСТВЕННЫМ условием. У безусловной
модели lift равен нулю по построению -- это и есть встроенный контроль.

F1 считается по СОДЕРЖАТЕЛЬНЫМ униграммам: служебные слова составляют
большинство токенов и совпадают у любых двух английских текстов, то есть
поднимают обе части разности и только глушат сигнал.

ЗАПУСК
------
    python analyze_paired_lift.py probe_results/projected_guidance-*snap-pos*.json

Базой берётся прогон с наименьшим alpha (или первый по порядку, если alpha
нет). Для каждого прогона печатается lift и ПАРНАЯ разность с базой.
"""
import glob
import json
import math
import re
import sys
from typing import Dict, List, Sequence, Tuple

# Служебные слова. Список намеренно короткий и фиксированный: он не
# подбирается под результат, и любой разумный вариант даёт ту же картину,
# потому что из РАЗНОСТИ общий фон всё равно уходит.
STOP = frozenset("""
a an the and or but if then than that this these those of in on at to for from
by with without as is are was were be been being am do does did doing have has
had having it its it's he she they them his her their we you i not no nor so
such can could will would shall should may might must about into over under
again further once here there when where why how all any both each few more
most other some only own same too very s t just don now
""".split())

WORD = re.compile(r"[a-z0-9']+")


def content_words(text: str) -> List[str]:
    return [w for w in WORD.findall(text.lower()) if w not in STOP]


def f1(pred: Sequence[str], ref: Sequence[str]) -> float:
    """Униграммный F1 по мультимножествам. Пустые тексты дают 0."""
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


def per_example(texts: List[dict]) -> Tuple[List[float], List[float]]:
    """F1 со своей целью и с переставленной. -> (своя, чужая), по примерам.

    Перестановка -- сдвиг на один по кругу, а не случайная: она детерминирована,
    и два прогона сравниваются на ОДНИХ И ТЕХ ЖЕ парах. Случайная перестановка
    добавила бы к разности собственный шум, ради устранения которого всё и
    затевается.
    """
    gen = [content_words(t["GEN"]) for t in texts]
    trg = [content_words(t["TRG"]) for t in texts]
    n = len(gen)
    own = [f1(gen[i], trg[i]) for i in range(n)]
    other = [f1(gen[i], trg[(i + 1) % n]) for i in range(n)]
    return own, other


def mean_sem(xs: Sequence[float]) -> Tuple[float, float]:
    n = len(xs)
    if n < 2:
        return (float(xs[0]) if n else 0.0), 0.0
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var / n)


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        blob = json.load(f)
    texts = blob.get("texts")
    if not texts:
        raise SystemExit(
            f"{path}: в json нет поля 'texts'. Прогон снят до правки, которая "
            f"кладёт все тексты, -- по двадцати примерам парную статистику не "
            f"посчитать. Пересними прогон.")
    own, other = per_example(texts)
    return {
        "path": path,
        "alpha": blob.get("snap_alpha"),
        "bert": blob.get("metrics", {}).get("bert-score"),
        "own": own,
        "other": other,
        "lift": [o - x for o, x in zip(own, other)],
    }


def main(paths: List[str]) -> None:
    runs = [load(p) for p in paths]
    if not runs:
        raise SystemExit("не передано ни одного json")
    sizes = {len(r["own"]) for r in runs}
    if len(sizes) > 1:
        raise SystemExit(
            f"прогоны разной длины {sorted(sizes)}: парное сравнение требует "
            f"одних и тех же примеров в одном порядке")

    runs.sort(key=lambda r: (r["alpha"] is None, r["alpha"]))
    base = runs[0]
    bl, bs = mean_sem(base["lift"])
    print(f"база: alpha={base['alpha']}  n={len(base['own'])}")
    print(f"  условный прирост {bl:+.5f} +- {bs:.5f}")
    print("")
    print("  alpha     bert    прирост      парная разность с базой")
    for r in runs:
        lm, ls = mean_sem(r["lift"])
        diff = [a - b for a, b in zip(r["lift"], base["lift"])]
        dm, ds = mean_sem(diff)
        # Сигмы считаем по ПАРНОЙ ошибке: именно ради неё прогоны идут на
        # одних промптах и одном сиде.
        sig = abs(dm) / ds if ds > 0 else float("inf") if dm else 0.0
        mark = "" if r is base else f"{dm:+.5f} +- {ds:.5f}  ({sig:.1f} сигм)"
        bert = "нет" if r["bert"] is None else f"{r['bert']:.5f}"
        print(f"  {str(r['alpha']):<8} {bert:>8}  {lm:+.5f}   {mark}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        args = sorted(glob.glob("probe_results/projected_guidance-*snap*.json"))
    main(args)
