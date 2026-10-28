"""Ось информативности условия: от одного бита до полной цели.

ГЛАВНЫЙ ЭКСПЕРИМЕНТ РАБОТЫ
--------------------------
Замеры 24.10 дали две крайние точки одной шкалы на ОДНОЙ замороженной
безусловной диффузии, в одном латентном пространстве, при одной и той же
механике guidance:

  условие                        косинус градиента   закрыто разрыва
  бинарный классификатор         0.009 (случ. 0.0001)   52% / 18%
  точка-цель (оракул)            1 по построению        98% / 94%

Классификатор при этом имеет AUC 0.99 -- он почти безошибочен. Значит дело не
в его качестве, не в оптимизаторе и не в геометрии (проекция на касательное
проверена и мешает). Остается одно объяснение: бинарная метка переносит один
бит, а чтобы указать КОНКРЕТНОЕ продолжение из множества допустимых, нужны
сотни бит.

Две точки -- это не закон. Этот прогон строит СПЕКТР условий с контролируемым
количеством информации и проверяет, ложатся ли все они на одну кривую
«качество от косинуса». Ни одно из условий не требует обучения: все они
получены деградацией оракульной цели (см. probes/degrade.py), то есть ось
строится целиком на инференсе.

РЕЗУЛЬТАТ ПЕРВОГО ПРОГОНА И ПОЧЕМУ cos_to_true ЛЖЕТ
---------------------------------------------------
Прогон 24.10 (16 заданий) дал монотонную картину по качеству, но показал, что
ОСЬ X выбрана неверно. Косинус градиента с направлением mu*true - x_t ведет
себя так:

  оракул (полная цель)     cos 0.0909   bert 0.980
  quantize K=2 (пусто)     cos 0.1775   bert 0.452
  prefix k=1..16           cos отрицательный, bert растет 0.371 -> 0.606

То есть у ПОЛНОЙ цели косинус вдвое ниже, чем у почти пустой квантизации, а у
prefix он отрицателен при монотонно растущем качестве. Причина: при t -> 1
mu ~ 5e-9 (CosineSD d=9, float32), поэтому mu*true - x_t ~ -x_t. "Направление
к истине" вырождается в "убрать шум", и косинус с ним тем выше, чем ближе цель
к среднему данных. Центроиды лежат у среднего, поэтому они и выигрывают.
Величина измеряет не информативность условия, а близость цели к центру облака.

Поэтому ось переписана. Выравнивание считается в пространстве x_0:
cos(z - x_0, true - x_0). Для оракула это 1 по построению, масштаб mu в него не
входит. Старая величина сохранена в json как cos_to_true -- это
задокументированный отрицательный результат по диагностике, а не ось.

РАБОЧАЯ ОСЬ: ВОССТАНОВИМОСТЬ ЦЕЛИ
---------------------------------
Информативность условия измеряется тем же, чем качество: насколько сама
деградированная цель похожа на истину (STEER vs TRG, bert). Это уже считалось в
первом прогоне, и по этой оси картина чистая:

  seqquant (цель хуже пола)  0.307 -> 0.320   bert 0.390 -> 0.408
  quantize                   0.355 -> 0.471   bert 0.470 -> 0.580
  оракул                     0.998            bert 0.980

Наклон около единицы, причем генерация идет ВЫШЕ восстановимости цели
(0.419 -> 0.525): диффузия не просто переносит условие, она доводит его до
многообразия текста. Номинальные биты эту кривую НЕ описывают: prefix-32 и
quantize-32 оба дают ~320 бит, но bert 0.722 против 0.470. Значит правильный
инвариант -- достижимое восстановление, а не счет бит. Это методический
результат работы.

ГЛАВНЫЙ ВЫВОД И ЧТО ИЗ НЕГО СЛЕДУЕТ ПРОВЕРИТЬ
---------------------------------------------
Точка-цель информативностью ~2 бита (seqquant) дает bert 0.390-0.408, то есть
НИЖЕ пола 0.4605: вести к плохой точке хуже, чем не вести никуда. Реальный
бинарный классификатор несет один бит и дает 0.5081, то есть ВЫШЕ пола. Один
бит обыгрывает два бита на порядок величины эффекта -- значит дело не только в
объеме информации, но и в ФОРМЕ условия: классификатор задает множество
допустимых продолжений, а не точку.

Это проверяется контролируемо, без обучения (PROBE_SET). При одной и той же
информации (метка кластера, log2 K бит) сравниваются три формы:
  centroid -- среднее кластера: точка ВНЕ многообразия;
  medoid   -- самый центральный реальный элемент: точка НА многообразии;
  mixture  -- мягкий минимум по кандидатам кластера: МНОЖЕСТВО.
Разность centroid -> medoid измеряет цену схода с многообразия, medoid ->
mixture -- выигрыш от множественной формы при той же информации. Если mixture
обыгрывает medoid, различие "множество против точки" доказано внутри работы, а
не только сравнением с классификатором.

РЕЖИМЫ (PROBE_DEGRADE)
----------------------
  none     полная цель -- верхняя точка оси
  prefix   PROBE_LEVEL = сколько содержательных позиций открыто
  quantize PROBE_LEVEL = сколько центроидов (k-means по позициям банка)
  seqquant PROBE_LEVEL = сколько центроидов по ПОСЛЕДОВАТЕЛЬНОСТЯМ, то есть
           "веди к теме"; только с ним работает PROBE_SET
  dilute   PROBE_LEVEL = проценты: z = mean + (L/100) (target - mean). Это
           каноническая форма частичной информации: условное среднее истины по
           b битам -- сжатая копия истины. Непрерывная ось, проходит через тот
           диапазон, где живет классификатор
  noisy    PROBE_LEVEL = проценты std: z = target + (L/100) std eps. Вторая
           непрерывная ось, другая по характеру порчи. Если dilute и noisy
           лягут на одну кривую, важен объем информации, а не способ порчи
  rank     PROBE_LEVEL = сколько главных компонент банка оставить

ЗАПУСК
------
    ARCH_TYPE=unconditional CHECKPOINT_STEP=87500 \
    PROBE_DEGRADE=prefix PROBE_LEVEL=8 ./probe_information_axis.sh

Настройки через окружение:
    PROBE_DEGRADE      (prefix)   none | prefix | quantize | seqquant |
                                  dilute | noisy | rank
    PROBE_LEVEL        (8)        уровень деградации, смысл зависит от режима
    PROBE_SET          (centroid) centroid | medoid | mixture -- форма условия
                                  при равной информации, только с seqquant
    PROBE_SET_MEMBERS  (4)        кандидатов в множестве для mixture
    PROBE_SET_SIGMA    (0.3)      ширина мягкого минимума, в долях РАЗБРОСА
                                  расстояний по кандидатам: ->0 жесткий
                                  argmin (цель на данных), ->inf равномерный
                                  вес (цель -- среднее кандидатов)
    PROBE_SEQ_BANK     (1024)     последовательностей в банке (должно быть
                                  больше числа тем, иначе k-means падает)
    PROBE_TARGET_RATIO (0.1)      доля шага ||lam*grad||/||score||
    PROBE_BANK         (8192)     векторов в банке
    PROBE_NUM_TEXTS    (1000)     сколько текстов сгенерировать
    PROBE_BATCH_SIZE   (64)       размер батча
    PROBE_OUT          (...)      путь к json
"""
import math
import os
import time

import torch
import torch.distributed as dist

from create_config import create_config, metrics_for_mode
from estimation_utils.metrics import compute_metric
from estimation_utils.metrics import compute_mauve
from probes.common import (
    build_probe_runner, content_mask, dump_json, encode_batch,
    latents_to_tokens, log, masked_cosine, masked_norm, probe_loader,
    progress, reverse_process,
)
from probes.degrade import (cluster_bundle, content_position_stats,
                            degrade_target, estimated_bits, set_weights)
from probes.manifold import LatentBank
from utils.util import parse, set_seed


def make_axis_guidance(runner, target_ratio, diag):
    """Reconstruction guidance к ДЕГРАДИРОВАННОЙ цели.

    Якобиан приближается identity, сила задается долей шага -- ровно та
    конфигурация, на которой оракульная цель дала bleu 0.938. Менять ее нельзя:
    точки оси обязаны отличаться ТОЛЬКО информативностью цели.
    """
    state = {"target": None, "true": None, "mask": None, "cand": None,
             "sigma_frac": 1.0}

    def fn(x_t, t, out, **kwargs):
        if state["target"] is None and state["cand"] is None:
            return None
        params = runner.dynamic.marginal_params(t)
        mu = params["mu"]
        x0 = out["x_0"]
        content = state["content"]

        if state["cand"] is not None:
            # МНОЖЕСТВО вместо точки: мягкий минимум по кандидатам кластера.
            # Именно такую форму имеет градиент классификатора -- «будь во
            # множестве допустимых», а не «иди в эту точку». Информация та же
            # (метка кластера), форма другая.
            cand = state["cand"]                                  # [B, M, L, D]
            w2 = content.unsqueeze(1).unsqueeze(-1).to(cand.dtype)
            diff = cand - x0.unsqueeze(1)
            d2 = (diff * w2).pow(2).flatten(2).sum(dim=2)         # [B, M]
            # Температура берётся от РАЗБРОСА расстояний по кандидатам, а не
            # от их величины -- разбор в probes.degrade.set_weights, там же
            # обе неудачные попытки с числами.
            w = set_weights(d2, state["sigma_frac"])              # [B, M]
            corr = (w.view(*w.shape, 1, 1) * diff).sum(dim=1)     # [B, L, D]
            with torch.no_grad():
                diag["set_entropy"] += (
                    -(w.clamp_min(1e-12).log() * w).sum(dim=1)).cpu().tolist()
        else:
            corr = state["target"] - x0
            if state["mask"] is not None:
                corr = corr * state["mask"].unsqueeze(-1).to(corr.dtype)

        grad = corr / mu

        # Нормы МАСКИРОВАННЫЕ. На паддинг-позициях цель это zero_emb, а x_0 --
        # что предскажет сеть; их разность входила бы в n_g и занижала бы
        # коэффициент, то есть доля шага на содержательных позициях
        # оказывалась бы меньше запрошенной, причем по-разному у режимов с
        # разной длиной цели. Все соседние статистики здесь уже маскированы.
        n_s = masked_norm(out["score"], content).clamp_min(1e-12)
        n_g = masked_norm(grad, content).clamp_min(1e-12)
        g = (target_ratio * n_s / n_g).view(-1, 1, 1) * grad

        with torch.no_grad():
            # ИСПРАВЛЕННАЯ ось: выравнивание считаем в пространстве x_0, а не
            # x_t. Замер 24.10 показал, что cos_to_true измеряет не то: при
            # t -> 1 mu ~ 5e-9, поэтому mu*true - x_t ~ -x_t, то есть «куда
            # надо идти» вырождается в «убрать шум». Косинус с этим
            # направлением тем выше, чем ближе цель к среднему данных, и
            # оракул (0.0909) проиграл почти пустой quantize-2 (0.1775).
            # cos_x0 от этого свободен: он равен 1 для оракула по построению.
            diag["cos_x0"] += masked_cosine(
                corr, state["true"] - x0, content).cpu().tolist()
            true_dir = mu * state["true"] - x_t
            diag["cos_to_true"] += masked_cosine(
                g, true_dir, content).cpu().tolist()
            diag["rel_norm"] += (masked_norm(g, content) / n_s).cpu().tolist()
        return g

    fn.state = state
    return fn


def main():
    log("старт: собираю конфиг")
    args = parse()
    config = create_config(args)

    if config.architecture_type != "unconditional":
        raise SystemExit(
            "ось строится на БЕЗУСЛОВНОЙ диффузии -- той же, что под classifier "
            "guidance. Запускайте с ARCH_TYPE=unconditional.")

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    config.local_rank = rank
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", init_method="env://",
                            world_size=world_size, rank=rank)
    dist.barrier()
    config.training.batch_size_per_gpu = config.training.batch_size // dist.get_world_size()

    modes = ("none", "prefix", "quantize", "seqquant", "dilute", "noisy",
             "rank", "offmanifold")
    mode = os.environ.get("PROBE_DEGRADE", "prefix")
    if mode not in modes:
        raise SystemExit(f"PROBE_DEGRADE={mode!r}: ожидается "
                         + " | ".join(modes))
    level = int(os.environ.get("PROBE_LEVEL", "8"))
    target_ratio = float(os.environ.get("PROBE_TARGET_RATIO", "0.1"))
    bank_size = int(os.environ.get("PROBE_BANK", "8192"))
    num_texts = int(os.environ.get("PROBE_NUM_TEXTS", "1000"))
    batch_size = int(os.environ.get("PROBE_BATCH_SIZE", "64"))

    # ФОРМА условия при одинаковой информации (метка кластера, log2 level бит):
    #   centroid -- среднее кластера, точка вне многообразия (как было);
    #   medoid   -- самый центральный реальный элемент, точка НА многообразии;
    #   mixture  -- мягкий минимум по кандидатам, МНОЖЕСТВО на многообразии.
    # Выбор медоида и кандидатов идет по центроиду, а не по истине: иначе
    # условие несло бы больше метки кластера и сравнение сломалось бы.
    set_mode = os.environ.get("PROBE_SET", "centroid")
    if set_mode not in ("centroid", "medoid", "mixture"):
        raise SystemExit(f"PROBE_SET={set_mode!r}: ожидается "
                         f"centroid | medoid | mixture")
    if set_mode != "centroid" and mode != "seqquant":
        raise SystemExit("PROBE_SET работает только с PROBE_DEGRADE=seqquant: "
                         "форму условия сравниваем при равной информации, а ее "
                         "задает метка кластера последовательностей")
    set_members = int(os.environ.get("PROBE_SET_MEMBERS", "4"))
    sigma_frac = float(os.environ.get("PROBE_SET_SIGMA", "0.3"))
    seq_bank = int(os.environ.get("PROBE_SEQ_BANK", "1024"))

    tag = mode if set_mode == "centroid" else f"{mode}-{set_mode}"
    out_path = os.environ.get(
        "PROBE_OUT",
        f"probe_results/information_axis-{config.data.datasets.datasets_list[0]}"
        f"-{tag}-L{level}-r{target_ratio:g}.json")

    # Промпт самой диффузии не подается (она безусловная), но нужен collate и
    # условные метрики -- иначе точки оси не сравнить с таблицей подходов.
    config.is_pipeline_conditional = True
    config.data.datasets.metrics[config.data.datasets.datasets_list[0]] = \
        metrics_for_mode(True)

    set_seed(config.seed)
    runner = build_probe_runner(config)

    # Сколько тем у offmanifold: метка кластера фиксирована вдоль свипа, но
    # сам уровень квантизации задаётся отдельно, потому что level занят beta.
    topic_level = int(os.environ.get("PROBE_TOPICS_LEVEL", "16"))

    bank = None
    if mode in ("quantize", "seqquant", "rank", "offmanifold"):
        # Для seqquant банк должен быть заметно больше числа тем: при 8192
        # позициях целых последовательностей набиралось всего 192, и level=256
        # падал. Просим последовательности отдельным бюджетом.
        need_seq = seq_bank if mode in ("seqquant", "offmanifold") else 0
        if mode == "seqquant" and need_seq <= level:
            raise SystemExit(
                f"PROBE_SEQ_BANK={need_seq} не больше числа тем {level}: "
                f"квантизация по темам требует банка больше числа центроидов")
        bank = LatentBank.from_runner(
            runner, size=bank_size, skip_examples=num_texts,
            batch_size=batch_size, min_sequences=need_seq)

    if dist.get_rank() == 0:
        print("=" * 72)
        print("PROBE: ось информативности условия")
        print("=" * 72)
        print(f"  чекпоинт          : {runner.loaded_checkpoint_path}")
        print(f"  деградация        : {mode}, уровень {level}")
        print(f"  форма условия     : {set_mode}"
              + (f" (кандидатов {set_members}, sigma {sigma_frac})"
                 if set_mode == "mixture" else ""))
        print(f"  доля шага         : {target_ratio}")
        bits = estimated_bits(mode, level, config.data.max_sequence_len)
        print(f"  оценка бит о цели : {'нет' if bits is None else f'~{bits:.0f}'}")
        print(f"  текстов           : {num_texts} (батч {batch_size})")
        print("=" * 72)

    diag = {"cos_to_true": [], "cos_x0": [], "rel_norm": [],
            "set_entropy": [], "target_recovery": [], "manifold_gap": [],
            "sequence_gap": []}
    guidance_fn = make_axis_guidance(runner, target_ratio, diag)

    loader = probe_loader(runner, split="test", batch_size=batch_size)
    gen_texts, trg_texts, src_texts, steer_texts = [], [], [], []
    started = time.time()
    desc = ""
    anchor = None

    for b_idx, batch in enumerate(loader):
        if len(gen_texts) >= num_texts:
            break
        batch, src_x, trg_x = encode_batch(runner, batch)
        bsz = len(batch["text_trg"])
        cond_mask = batch.get("attention_mask_src")
        content = content_mask(runner, batch["input_ids_trg"])

        if mode in ("prefix", "dilute", "noisy") and anchor is None:
            # Якорь считаем ОДИН раз, по первому батчу, и переиспользуем на
            # всех остальных. Иначе «про это условие не говорит ничего»
            # означало бы разное на разных батчах, и ось x дрожала бы внутри
            # одной точки -- при том что вся ось собирается из отдельных
            # заданий, которые обязаны отличаться ТОЛЬКО уровнем.
            anchor = content_position_stats(trg_x, content)

        z, pos_mask, desc = degrade_target(
            mode, trg_x, level, bank=bank, content=content,
            seed=config.seed + b_idx,
            pos_mean=None if anchor is None else anchor[0],
            pos_std=None if anchor is None else anchor[1],
            topic_level=topic_level)
        cand = None
        if set_mode != "centroid":
            centroid, medoid, members = cluster_bundle(
                bank, trg_x, level, members=set_members)
            # STEER и для medoid, и для mixture -- медоид: лучшая ОДИНОЧНАЯ
            # реализация метки кластера. Значит у двух режимов совпадает и
            # информация, и достижимое восстановление цели, а отличается только
            # форма guidance. Это и делает сравнение контролируемым.
            z = medoid
            if set_mode == "mixture":
                cand = members
            desc = f"{desc}, форма {set_mode}"
        guidance_fn.state.update({"target": z, "true": trg_x, "cand": cand,
                                  "mask": pos_mask, "content": content,
                                  "sigma_frac": sigma_frac})

        # НАСКОЛЬКО ЦЕЛЬ УДАЛЕНА ОТ ДАННЫХ. Вторая ось работы рядом с
        # восстановимостью: 25.10 выяснилось, что при РОВНО одной информации
        # (метка кластера) центроид даёт 0.394, а реальный элемент кластера --
        # 0.496. Разницу между ними объясняет не информация, а вот эта
        # величина, и без неё главный вывод держится на двух точках.
        if bank is not None:
            with torch.no_grad():
                # Главная колонка -- по ЦЕЛЫМ последовательностям. Позиционный
                # зазор на свипе 26.10 поехал в обратную сторону (центроид
                # 0.795 против реального медоида 1.110): усреднение сотен
                # последовательностей сносит каждую позицию в плотное ядро
                # позиционного облака, туда, где "the" и запятая. Разбор в
                # LatentBank.sequence_gap. Позиционный оставлен второй
                # колонкой -- он про другое: правдоподобна ли позиция сама по
                # себе. Их расхождение и есть содержание наблюдения.
                diag["sequence_gap"] += bank.sequence_gap(
                    z, content).cpu().tolist()
                diag["manifold_gap"] += bank.manifold_gap(
                    z, content).cpu().tolist()

        # Во что превращается сама ЦЕЛЬ после деградации -- вторая проверка
        # информативности, независимая от генерации.
        steer, _ = latents_to_tokens(runner, z, cond_x=src_x, cond_mask=cond_mask)

        set_seed(config.seed + b_idx)
        lat = reverse_process(
            runner, batch_size=bsz, cond_x=src_x, cond_mask=cond_mask,
            attention_mask=None, guidance_fn=guidance_fn,
            show_steps=(b_idx == 0))
        texts, _ = latents_to_tokens(runner, lat, cond_x=src_x, cond_mask=cond_mask)

        gen_texts += texts
        steer_texts += steer
        trg_texts += list(runner.tokenizer.batch_decode(
            batch["input_ids_trg"], skip_special_tokens=True))
        src_texts += list(batch["text_src"])

        if dist.get_rank() == 0 and b_idx == 0:
            print(f"\n  деградация: {desc}")
            for i in range(min(3, bsz)):
                print(f"\n  --- пример {i + 1} ---")
                print(f"  TRG   : {trg_texts[i][:160]}")
                print(f"  STEER : {steer_texts[i][:160]}")
                print(f"  GEN   : {gen_texts[i][:160]}")

        c = diag["cos_x0"]
        progress(min(len(gen_texts), num_texts), num_texts, started, "текстов",
                 extra=f"cos(x0) к истине {sum(c) / len(c):.4f}" if c else "")

    gen_texts = gen_texts[:num_texts]
    trg_texts = trg_texts[:num_texts]
    src_texts = src_texts[:num_texts]
    steer_texts = steer_texts[:num_texts]

    metric_names = config.data.datasets.metrics[
        config.data.datasets.datasets_list[0]]["metrics"]

    def score(preds, refs, label):
        res = {}
        for i, name in enumerate(metric_names, start=1):
            log(f"метрики {label}: {name} ({i}/{len(metric_names)})...")
            res[name] = compute_metric(name, predictions=preds,
                                       references=refs, sources=src_texts)
        return res

    metrics = score(gen_texts, trg_texts, "GEN vs TRG")
    # Сколько информации о цели осталось в самой деградированной цели -- шкала,
    # не зависящая от того, насколько хорошо guidance ее реализовал.
    recovery = score(steer_texts, trg_texts, "STEER vs TRG")
    # ДОШЁЛ ЛИ сэмплер до цели, которую ему дали. Без этого числа свип
    # 26.10 не читается: на beta=0.9 и выше падают СРАЗУ две вещи -- цель
    # перестаёт декодироваться в текст ("s the the the") и сэмплер перестаёт
    # её отслеживать, и по одному GEN vs TRG их не различить. Верность
    # отвечает ровно на вторую половину вопроса и от информативности цели не
    # зависит: сравнение идёт с тем, что цель есть, а не с истиной.
    fidelity = score(gen_texts, steer_texts, "GEN vs STEER")

    # ЯВЛЯЕТСЯ ЛИ ЦЕЛЬ ЕЩЁ ТЕКСТОМ -- прямое измерение главного утверждения.
    #
    # Весь свип построен на том, что цель уходит с многообразия реальных
    # текстов. До сих пор это измерялось КОСВЕННО: падает bert у генерации.
    # Евклидово расстояние до данных для этого непригодно (концентрация
    # расстояний в 49152 измерениях), и как ось оно опровергнуто.
    #
    # MAUVE сравнивает РАСПРЕДЕЛЕНИЯ, а не пары, и отвечает ровно на нужный
    # вопрос: похож ли корпус декодированных целей на корпус реальных текстов.
    # На beta=1 цель декодируется в "The", и это обязано обрушить MAUVE, тогда
    # как попарные метрики такое различают плохо.
    #
    # Метрика необязательная: ей нужен gpt2-large, и на ноде без интернета его
    # может не оказаться. Падать из-за этого весь прогон не должен.
    def mauve_or_none(preds, refs, label):
        try:
            log(f"MAUVE {label} (gpt2-large)...")
            return float(compute_mauve(predictions=preds, references=refs))
        except Exception as exc:  # noqa: BLE001 -- диагностика не должна падать
            print(f"[PROBE] MAUVE {label} не посчитан: {exc}", flush=True)
            return None

    target_mauve = mauve_or_none(steer_texts, trg_texts, "цель против реальных")
    gen_mauve = mauve_or_none(gen_texts, trg_texts, "генерация против реальных")

    def mean(xs):
        return float(sum(xs) / len(xs)) if xs else None

    def fmt(v):
        # Режим без диагностики оставляет список пустым; падать на
        # форматировании None ПОСЛЕ того, как метрики посчитаны, -- это терять
        # весь прогон, со свипом так уже было.
        return "нет" if v is None else f"{v:.5f}"

    if dist.get_rank() == 0:
        print("\n" + "=" * 72)
        print(f"----- {mode} / уровень {level} / доля {target_ratio} -----")
        for k, v in metrics.items():
            print(f"{k}: {v:0.5f}")
        print(f"\nвосстановимость цели (bert)      : {recovery['bert-score']:.5f}   <- ОСЬ X")
        print(f"качество генерации   (bert)      : {metrics['bert-score']:.5f}   <- ОСЬ Y")
        print(f"удалённость цели от данных       : {fmt(mean(diag['sequence_gap']))}   "
              f"<- ВТОРАЯ ОСЬ (1.0 = как у реальной последовательности)")
        print(f"верность цели (GEN vs STEER)     : {fidelity['bert-score']:.5f}   "
              f"<- ДОШЁЛ ЛИ сэмплер до цели")
        print(f"зазор по отдельным позициям      : {fmt(mean(diag['manifold_gap']))}   "
              f"(другая величина, не главная ось)")
        print(f"cos(x0) условия с истиной        : {fmt(mean(diag['cos_x0']))}")
        print(f"cos(x_t) старая ось, непригодна  : {fmt(mean(diag['cos_to_true']))}")
        print(f"доля шага                        : {mean(diag['rel_norm']):.4f}")
        if diag["set_entropy"]:
            print(f"энтропия веса по кандидатам      : "
                  f"{fmt(mean(diag['set_entropy']))} из "
                  f"{math.log(set_members):.3f}")
        print("=" * 72)
        print("Точка оси: (восстановимость цели, качество генерации).")
        print("Сравнивать с полом без guidance и с реальным классификатором "
              "надо по json\nтех прогонов, а не по числам в этой подписи: "
              "они снимаются другой стадией,\nи зашитая здесь копия молча "
              "устареет при любом перезапуске.")

    dump_json({
        "probe": "information_axis",
        "dataset": config.data.datasets.datasets_list[0],
        "diffusion_checkpoint": runner.loaded_checkpoint_path,
        "step": runner.step,
        "degrade": mode,
        "level": level,
        "description": desc,
        "estimated_bits": estimated_bits(mode, level, config.data.max_sequence_len),
        "target_ratio": target_ratio,
        "solver_steps": runner.dynamic.N,
        "n_texts": len(gen_texts),
        "metrics": metrics,
        "target_recovery": recovery,
        "target_fidelity": fidelity,
        "cos_to_true": mean(diag["cos_to_true"]),
        "cos_x0": mean(diag["cos_x0"]),
        "sequence_gap": mean(diag["sequence_gap"]),
        "manifold_gap": mean(diag["manifold_gap"]),
        "topic_level": topic_level if mode == "offmanifold" else None,
        "set_shape": set_mode,
        "set_members": set_members if set_mode == "mixture" else None,
        "set_sigma_frac": sigma_frac if set_mode == "mixture" else None,
        "set_entropy": mean(diag["set_entropy"]),
        "rel_norm_mean": mean(diag["rel_norm"]),
        "target_mauve": target_mauve,
        "gen_mauve": gen_mauve,
        # ВСЕ тексты, а не двадцать: по двадцати примерам не посчитать ни
        # парную статистику, ни распределённые метрики задним числом, а
        # пересъёмка прогона стоит GPU-часов.
        "texts": [{"SRC": src_texts[i], "STEER": steer_texts[i],
                   "TRG": trg_texts[i], "GEN": gen_texts[i]}
                  for i in range(len(gen_texts))],
    }, out_path)


if __name__ == "__main__":
    main()
