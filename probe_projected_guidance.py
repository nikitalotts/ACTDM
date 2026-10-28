"""Guidance с проекцией градиента на касательное пространство.

ЗАЧЕМ ЭТОТ ПРОГОН
-----------------
probe_signal_survival дал факт, из которого все дальнейшее и следует:
классификатор различает правильное продолжение от чужого с AUC 0.99 на всем
диапазоне шума, но косинус его градиента с направлением к истинному
продолжению равен 0.009 при СКО случайного косинуса 0.0045 (знаковое среднее
0.0001 базой быть не может: оно сходится к нулю). Градиент вдвое лучше
случайного -- и при этом на 99% ортогонален нужному направлению.

Гипотеза: ортогональная часть -- это движение ПРОЧЬ С МНОГООБРАЗИЯ. Поэтому
lambda и нельзя поднять: ортогональная компонента разрушает сэмпл раньше, чем
касательная накопит полезный сдвиг. Это объясняет и свип lambda из ВКР, где при
lambda >= 5000 все метрики обваливались.

Отсюда метод: спроецировать градиент на локальное касательное подпространство
многообразия энкодингов (оценивается локальным PCA по соседям из банка чистых
латентов, см. probes/manifold.py) и только после этого применять.

ЧТО СРАВНИВАЕТСЯ (PROBE_GRAD_MODE)
----------------------------------
  raw     -- как сейчас: градиент классификатора целиком. Базовая линия метода.
  tangent -- ПРЕДЛАГАЕМЫЙ МЕТОД: только касательная компонента.
  normal  -- ТОЛЬКО ортогональная компонента. Контроль, без которого tangent не
             доказателен: если выигрыш дает не касательное направление, а просто
             уменьшение шага, то normal при той же силе шага покажет то же
             самое. Если normal заметно хуже -- дело именно в направлении.

  snap-pos  -- МЕТОД ИЗ ЗАКОНА 26.10: к градиенту подмешивается сила,
               тянущая текущую оценку x_0 к ближайшим РЕАЛЬНЫМ
               позиционным векторам (квантизация по банку).
  snap-seq  -- то же, но к ближайшей реальной ПОСЛЕДОВАТЕЛЬНОСТИ целиком.
               При alpha=1 это уже retrieval, а не генерация, и так его и надо
               называть; содержательна промежуточная сила.

У обоих snap-режимов alpha=0 обязан воспроизводить raw побитово -- это
встроенный контроль на то, что свип меряет притягивание, а не побочный эффект
переписанного кода. Банк для них берётся из train, а не из test.

КАК ЗАДАЕТСЯ СИЛА (PROBE_SCALE_MODE)
------------------------------------
  const    -- постоянная lambda, как в ВКР.
  adaptive -- lambda_t подбирается на каждом шаге так, чтобы
              ||lambda_t * grad|| / ||score|| равнялось PROBE_TARGET_RATIO.

Adaptive -- не украшение, а условие корректности сравнения. Во-первых, замер
показал, что при постоянной lambda=10 эта доля меняется вдоль траектории в 170
раз (0.0001 при t=0.01 против 0.017 при t=0.8), то есть постоянная lambda
катастрофически рассогласована по уровню шума. Во-вторых, проекция уменьшает
норму градиента в разы, и сравнивать raw с tangent при одной lambda значило бы
сравнивать разные размеры шага, а не разные направления. По умолчанию здесь
adaptive именно поэтому.

МНОЖИТЕЛЬ (1 - sigma(f)) (PROBE_SIGMOID)
----------------------------------------
Точная форма градиента логарифма: grad log sigma(f) = (1 - sigma(f)) * grad f.
Математически верно, но следствие тяжелое: множитель обнуляет guidance там, где
классификатор уверен, -- а замер показал, что уверен он как раз при малом шуме,
где AUC = 0.9997 и где текст и решается. PROBE_SIGMOID=0 убирает множитель
(ведем по grad f) и проверяет, оживает ли guidance при малом t. Это отдельный
результат независимо от знака.

ЗАПУСК
------
    ARCH_TYPE=guidance AUG_SCHEME=augmented CHECKPOINT_STEP=87500 \
    PROBE_GRAD_MODE=tangent PROBE_TARGET_RATIO=0.1 ./probe_projected_guidance.sh

Настройки через окружение:
    PROBE_GRAD_MODE    (tangent)  raw | tangent | normal
    PROBE_SCALE_MODE   (adaptive) adaptive | const
    PROBE_TARGET_RATIO (0.1)      целевая доля ||lam*grad||/||score|| у adaptive
    PROBE_LAMBDA       (10)       постоянная lambda у const; 0 -- чистая
                                  безусловная генерация (базовая линия)
    PROBE_SIGMOID      (1)        оставить множитель (1 - sigma(f))
    PROBE_TANGENT_DIM  (16)       размерность касательного
    PROBE_KNN          (32)       соседей для локального PCA
    PROBE_BANK         (8192)     векторов в банке
    PROBE_SNAP_ALPHA   (0)        сила притягивания к данным, 0..1; 0 это raw
    PROBE_SEQ_BANK     (8192)     последовательностей в банке для snap-seq
    PROBE_BANK_SPLIT   (train у snap) сплит для банка; test у диагностик
    PROBE_BANK_CACHE   (авто)     файл кэша банка; пустая строка отключает.
                                  На train препроцессинг стоит ~1.5 часа и
                                  не кэшируется сам (см. data/dataset_wiki.py)
    PROBE_NUM_TEXTS    (1000)     сколько текстов сгенерировать
    PROBE_BATCH_SIZE   (64)       размер батча
    PROBE_OUT          (...)      путь к json
"""
import os
import time

import torch
import torch.distributed as dist

from create_config import create_config
from estimation_utils.metrics import compute_metric
from probes.common import (
    build_probe_runner, dump_json, encode_batch, latents_to_tokens, log,
    probe_loader, progress, reverse_process,
)
from probes.manifold import LatentBank
from utils.util import parse, set_seed


def classifier_grad(runner, x_t, cond_x, t, cond_mask, sigmoid_factor=True):
    """Градиент классификатора по x_t.

    При sigmoid_factor=True совпадает с боевым compute_classifier_guidance
    (grad log sigma(f)); при False возвращает чистый grad f -- без множителя,
    который гасит guidance там, где классификатор уверен.
    """
    x = x_t.detach().clone().requires_grad_(True)
    logits = runner.cond_encoder(
        src_embeds=cond_x.detach(), noisy_trg_embeds=x, t=t, src_mask=cond_mask)
    grad = torch.autograd.grad(logits.sum(), x, create_graph=False)[0]
    if sigmoid_factor:
        grad = (1 - torch.sigmoid(logits)).view(-1, 1, 1) * grad
    return grad


def bins_by_t(ts, values, edges=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)):
    """Средние values по корзинам уровня шума. -> [(lo, hi, n, mean)].

    Нужна, потому что среднее по всей траектории отвечает не на тот вопрос.
    Замер 27.10 дал удалённость 1.2617 и косинус -0.0656 в среднем по всем
    шагам солвера -- но при t -> 1 оценка x_0 заведомо мусорна, уводить её
    некуда, и среднее определяется именно этими шагами. Решается текст при
    МАЛОМ t (там же классификатор уверен: AUC 0.9997), и содержателен сход с
    данных именно там. Если он сосредоточен при большом t, притягивание надо
    включать не на всей траектории, и это уже другой метод.

    Правая граница последней корзины включается: t ровно 1.0 иначе выпал бы из
    разбивки целиком, а это самый шумный конец, где и живёт артефакт.
    """
    if not ts or len(ts) != len(values):
        return []
    out = []
    last = edges[-1]
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [v for tv, v in zip(ts, values)
               if lo <= tv < hi or (hi == last and tv == hi)]
        out.append((lo, hi, len(sel),
                    float(sum(sel) / len(sel)) if sel else None))
    return out


def _blend_with_manifold(bank, mode, alpha, g, out, score_norm,
                         target_ratio, diag, t=None):
    """Подмешать к градиенту классификатора силу, тянущую цель НА данные.

    МЕТОД, КОТОРЫЙ СЛЕДУЕТ ИЗ ЗАКОНА. Свип 26.10 показал: при постоянной
    информации качество монотонно падает, когда цель едет от реальной
    последовательности к усреднению (bert 0.5059 -> 0.4019, верность
    0.980 -> 0.464). Лучшая форма условия во всём исследовании -- предел
    sigma -> 0 у множества, то есть «цель = ближайшая реальная
    последовательность». Здесь та же операция применяется не к оракульной
    цели, а к настоящему classifier guidance.

    СМЕШИВАЮТСЯ НАПРАВЛЕНИЯ, А НЕ ВЕЛИЧИНЫ, и результат возвращается к той же
    доле шага. Иначе alpha менял бы заодно размер шага, и сравнение показывало
    бы не «куда идти», а «как далеко», -- та же причина, по которой здесь
    вообще заведён adaptive.

    ЧЕРЕЗ x_0 ОБРАТНО НЕ ПЕРЕСЧИТЫВАЕМ. Подразумеваемая цель
    x_0 + std^2 * g / mu при t -> 1 бессмысленна: mu ~ 5e-9 (CosineSD d=9),
    и десятипроцентная правка score означает изменение x_0 в десятки
    миллионов раз. Обратный ход (mu*x_0_target - x_t)/std^2 к тому же
    вычитает почти равные числа и возвращает ошибку, усиленную на 1/mu ~ 2e8.
    Поэтому тянем ОТ x_0, который сеть выдаёт напрямую, а сдвиг в пространстве
    score отличается на положительный множитель mu/std^2, который всё равно
    снимается нормировкой.

    alpha=0 обязан воспроизводить raw побитово -- это встроенный контроль.
    ДИАГНОСТИКИ ПРИ ЭТОМ СЧИТАЮТСЯ ВСЁ РАВНО. Прогон 26.10 с alpha=0 вернул
    «нет (guidance выключен)» ровно в тех двух колонках, ради которых
    контрольная точка и запускалась первой: удалённость x_0 от данных и
    косинус градиента с направлением к ним не зависят от того, применяем мы
    притягивание или нет, и в контроле они нужны больше всего -- это замер
    НЕТРОНУТОГО guidance.
    """
    with torch.no_grad():
        x0 = out["x_0"]
        if mode == "snap-pos":
            target = bank.snap_positions(x0)
        else:
            # Маска из ЕДИНИЦ, а не содержательные позиции цели: длина цели
            # методу неизвестна, и подсовывать её сюда значило бы протащить
            # в метод информацию об ответе.
            ones = torch.ones(x0.shape[:2], dtype=torch.bool, device=x0.device)
            target = bank.snap_sequence(x0, ones)
        delta = target - x0

        def unit(v):
            return v / v.flatten(1).norm(dim=1).clamp_min(1e-12).view(-1, 1, 1)

        dir_cls, dir_man = unit(g), unit(delta)

        # Две диагностики, которые сами по себе проверяют диагноз работы:
        # насколько текущая оценка далека от ближайшего реального объекта и
        # толкает ли классификатор ПРОЧЬ от него. Если косинус около нуля,
        # классификатор с многообразия не уводит, и чинить надо не это.
        diag["snap_dist"] += (
            delta.flatten(1).norm(dim=1)
            / x0.flatten(1).norm(dim=1).clamp_min(1e-12)).cpu().tolist()
        diag["cos_cls_manifold"] += (
            (dir_cls * dir_man).flatten(1).sum(dim=1)).cpu().tolist()
        # Уровень шума пишем рядом с каждым замером. Среднее по всей
        # траектории отвечает не на тот вопрос: при t -> 1 оценка x_0 заведомо
        # мусорна, уводить её некуда, и среднее определяется именно этими
        # шагами. Содержателен сход с данных там, где РЕШАЕТСЯ ТЕКСТ, то есть
        # при малом t, -- а это видно только с разбивкой.
        if t is not None:
            diag["snap_t"] += t.detach().reshape(-1).cpu().tolist()

        if alpha <= 0.0:
            return g
        blended = (1.0 - alpha) * dir_cls + alpha * dir_man
        return (target_ratio * score_norm).view(-1, 1, 1) * unit(blended)


def make_guidance(runner, bank, mode, scale_mode, target_ratio, lam,
                  sigmoid_factor, diag, snap_alpha=0.0):
    def fn(x_t, t, out, cond=None, cond_mask=None, **kwargs):
        if cond is None:
            raise RuntimeError("guidance требует энкодинги промпта")
        if scale_mode == "const" and lam == 0.0:
            return None

        with torch.enable_grad():
            grad = classifier_grad(runner, x_t, cond, t, cond_mask, sigmoid_factor)

        raw_norm = grad.flatten(1).norm(dim=1)
        if mode in ("tangent", "normal"):
            # Касательное оценивается в x_0 -- текущей оценке ЧИСТОГО состояния.
            # В x_t его оценивать нельзя: x_t зашумлен и лежит вне многообразия
            # по построению, соседи там бессмысленны.
            proj = bank.project(grad, out["x_0"])
            grad = proj if mode == "tangent" else grad - proj

        score_norm = out["score"].flatten(1).norm(dim=1).clamp_min(1e-12)
        grad_norm = grad.flatten(1).norm(dim=1).clamp_min(1e-12)

        if scale_mode == "adaptive":
            coef = (target_ratio * score_norm / grad_norm).view(-1, 1, 1)
        else:
            coef = torch.full_like(score_norm, lam).view(-1, 1, 1)

        g = coef * grad

        if mode.startswith("snap"):
            g = _blend_with_manifold(bank, mode, snap_alpha, g, out,
                                     score_norm, target_ratio, diag, t)

        with torch.no_grad():
            diag["rel_norm"] += (g.flatten(1).norm(dim=1) / score_norm).cpu().tolist()
            diag["tangent_frac"] += (grad_norm / raw_norm.clamp_min(1e-12)
                                     ).pow(2).cpu().tolist()
        return g

    return fn


def main():
    log("старт: собираю конфиг")
    args = parse()
    config = create_config(args)

    if not config.classifier_guidance:
        raise SystemExit(
            "probe_projected_guidance управляет безусловной диффузией через "
            "классификатор -- запускайте с ARCH_TYPE=guidance и AUG_SCHEME=...")

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    config.local_rank = rank
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", init_method="env://",
                            world_size=world_size, rank=rank)
    dist.barrier()
    config.training.batch_size_per_gpu = config.training.batch_size // dist.get_world_size()

    mode = os.environ.get("PROBE_GRAD_MODE", "tangent")
    if mode not in ("raw", "tangent", "normal", "snap-pos", "snap-seq"):
        raise SystemExit(f"PROBE_GRAD_MODE={mode!r}: ожидается "
                         f"raw | tangent | normal | snap-pos | snap-seq")
    scale_mode = os.environ.get("PROBE_SCALE_MODE", "adaptive")
    if scale_mode not in ("adaptive", "const"):
        raise SystemExit(f"PROBE_SCALE_MODE={scale_mode!r}: ожидается adaptive | const")

    target_ratio = float(os.environ.get("PROBE_TARGET_RATIO", "0.1"))
    lam = float(os.environ.get("PROBE_LAMBDA", "10"))
    sigmoid_factor = os.environ.get("PROBE_SIGMOID", "1") == "1"
    tangent_dim = int(os.environ.get("PROBE_TANGENT_DIM", "16"))
    knn = int(os.environ.get("PROBE_KNN", "32"))
    bank_size = int(os.environ.get("PROBE_BANK", "8192"))
    num_texts = int(os.environ.get("PROBE_NUM_TEXTS", "1000"))
    batch_size = int(os.environ.get("PROBE_BATCH_SIZE", "64"))
    snap_alpha = float(os.environ.get("PROBE_SNAP_ALPHA", "0"))
    seq_bank = int(os.environ.get("PROBE_SEQ_BANK", "8192"))
    # Банк для МЕТОДА берётся из train. Касательное -- диагностика, там сойдёт
    # и test с пропуском оцениваемых примеров; но притягивать генерации к
    # латентам тестового сплита значит подглядывать в ответ, и рецензент снимет
    # работу на этом одном предложении.
    bank_split = os.environ.get(
        "PROBE_BANK_SPLIT", "train" if mode.startswith("snap") else "test")

    if mode.startswith("snap"):
        if scale_mode != "adaptive":
            raise SystemExit(
                "snap-режимы требуют PROBE_SCALE_MODE=adaptive: alpha смешивает "
                "НАПРАВЛЕНИЯ при одной доле шага, иначе он менял бы и шаг")
        if not 0.0 <= snap_alpha <= 1.0:
            raise SystemExit(f"PROBE_SNAP_ALPHA={snap_alpha}: ожидается [0, 1]")

    strength = (f"r{target_ratio:g}" if scale_mode == "adaptive" else f"l{lam:g}")
    tag = mode if not mode.startswith("snap") else f"{mode}-a{snap_alpha:g}"
    out_path = os.environ.get(
        "PROBE_OUT",
        f"probe_results/projected_guidance-{config.data.datasets.datasets_list[0]}"
        f"-{config.cond_encoder.augmentation_scheme}-{tag}-{scale_mode}-{strength}"
        f"-sig{int(sigmoid_factor)}.json")

    set_seed(config.seed)
    runner = build_probe_runner(config)
    # Боевая ветка guidance внутри calc_score гасится: градиент применяет этот
    # прогон сам, иначе он применился бы дважды и с разными правилами.
    runner.use_guidance = False

    bank = None
    if mode != "raw":
        need_seq = seq_bank if mode == "snap-seq" else 0
        # Кэш банка на диске. Препроцессинг сплита в data/dataset_wiki.py идёт
        # с load_from_cache_file=False, то есть повторяется в КАЖДОМ задании:
        # на train это 1 час 22 минуты (замер 26.10) против секунд на test.
        # Свип из десяти заданий отдал бы этому 13 GPU-часов. Первое задание
        # собирает банк и пишет, остальные читают за секунды.
        default_cache = (
            f"probe_results/latent_bank-{config.data.datasets.datasets_list[0]}"
            f"-{bank_split}-{bank_size}-s{need_seq}-{runner.step}.pt")
        cache_path = os.environ.get("PROBE_BANK_CACHE", default_cache) or None
        # skip_examples нужен только на test: там банк иначе собрался бы по тем
        # же примерам, которые генерируются. На train пересечения нет.
        bank = LatentBank.from_runner(
            runner, size=bank_size,
            skip_examples=0 if bank_split != "test" else num_texts,
            split=bank_split, batch_size=batch_size, knn=knn, dim=tangent_dim,
            min_sequences=need_seq, cache_path=cache_path)

    if dist.get_rank() == 0:
        print("=" * 72)
        print("PROBE: guidance с проекцией на касательное")
        print("=" * 72)
        print(f"  чекпоинт          : {runner.loaded_checkpoint_path}")
        print(f"  классификатор     : {config.cond_encoder.augmentation_scheme}")
        print(f"  режим градиента   : {mode}")
        print(f"  сила              : {scale_mode} "
              + (f"(доля {target_ratio})" if scale_mode == "adaptive"
                 else f"(lambda {lam})"))
        print(f"  множитель (1-sig) : {'есть' if sigmoid_factor else 'убран'}")
        if bank is not None:
            print(f"  касательное       : dim={tangent_dim}, knn={knn}, "
                  f"банк={tuple(bank.bank.shape)}")
            print(f"  доля у случайного вектора: {bank.random_tangent_fraction:.4f} "
                  f"(база сравнения)")
        print(f"  текстов           : {num_texts} (батч {batch_size})")
        print("=" * 72)

    diag = {"rel_norm": [], "tangent_frac": [],
            "snap_dist": [], "cos_cls_manifold": [], "snap_t": []}
    guidance_fn = make_guidance(runner, bank, mode, scale_mode, target_ratio,
                                lam, sigmoid_factor, diag, snap_alpha)

    loader = probe_loader(runner, split="test", batch_size=batch_size)
    gen_texts, trg_texts, src_texts = [], [], []
    started = time.time()

    for b_idx, batch in enumerate(loader):
        if len(gen_texts) >= num_texts:
            break
        batch, src_x, _ = encode_batch(runner, batch)
        bsz = len(batch["text_trg"])
        cond_mask = batch.get("attention_mask_src")

        set_seed(config.seed + b_idx)
        lat = reverse_process(
            runner, batch_size=bsz, cond_x=src_x, cond_mask=cond_mask,
            attention_mask=None, guidance_fn=guidance_fn,
            show_steps=(b_idx == 0))
        texts, _ = latents_to_tokens(runner, lat, cond_x=src_x, cond_mask=cond_mask)

        gen_texts += texts
        trg_texts += list(runner.tokenizer.batch_decode(
            batch["input_ids_trg"], skip_special_tokens=True))
        src_texts += list(batch["text_src"])

        if dist.get_rank() == 0 and b_idx == 0:
            for i in range(min(3, bsz)):
                print(f"\n  --- пример {i + 1} ---")
                print(f"  SRC : {src_texts[i][:160]}")
                print(f"  TRG : {trg_texts[i][:160]}")
                print(f"  GEN : {gen_texts[i][:160]}")

        tf = diag["tangent_frac"]
        rn = diag["rel_norm"]
        progress(min(len(gen_texts), num_texts), num_texts, started, "текстов",
                 extra=(f"касательная доля {sum(tf) / len(tf):.4f}, "
                        f"шаг {sum(rn) / len(rn):.4f}" if tf else ""))

    gen_texts = gen_texts[:num_texts]
    trg_texts = trg_texts[:num_texts]
    src_texts = src_texts[:num_texts]

    metric_names = config.data.datasets.metrics[
        config.data.datasets.datasets_list[0]]["metrics"]
    metrics = {}
    for i, name in enumerate(metric_names, start=1):
        log(f"метрики: {name} ({i}/{len(metric_names)})...")
        metrics[name] = compute_metric(
            name, predictions=gen_texts, references=trg_texts, sources=src_texts)

    def mean(xs):
        return float(sum(xs) / len(xs)) if xs else None

    def fmt(v, prec=4):
        # У базовой линии (lambda=0) перехват возвращает None сразу, диагностика
        # остается пустой, и прямое форматирование None роняло прогон УЖЕ ПОСЛЕ
        # того, как метрики посчитаны: json не сохранялся, карта потрачена зря.
        return "нет (guidance выключен)" if v is None else f"{v:.{prec}f}"

    if dist.get_rank() == 0:
        print("\n" + "=" * 72)
        print(f"----- {mode} / {scale_mode} / {strength} / "
              f"sigmoid={int(sigmoid_factor)} -----")
        for k, v in metrics.items():
            print(f"{k}: {v:0.5f}")
        print(f"\nсредняя доля шага ||lam*grad||/||score||: "
              f"{fmt(mean(diag['rel_norm']))}")
        if bank is not None and mode in ("tangent", "normal"):
            # В режиме normal здесь лежит доля ОСТАВШЕЙСЯ (нормальной)
            # компоненты, а не касательной: считается ||grad после проекции|| /
            # ||grad до||, а проекция там берет дополнение. В raw проекции нет
            # вовсе и значение тождественно 1.0 -- печатать его как «долю»
            # значит показывать величину, которая ничего не измеряет.
            what = "касательная" if mode == "tangent" else "нормальная"
            print(f"средняя {what} доля энергии градиента: "
                  f"{fmt(mean(diag['tangent_frac']))} "
                  f"(у случайного вектора {bank.random_tangent_fraction:.4f})")
        if mode.startswith("snap"):
            print(f"alpha притягивания к данным      : {snap_alpha:g}"
                  + ("   <- КОНТРОЛЬ: обязан совпасть с raw"
                     if snap_alpha == 0 else ""))
            print(f"удалённость x_0 от ближайшего реального: "
                  f"{fmt(mean(diag['snap_dist']))}")
            print(f"cos(градиент классификатора, к данным): "
                  f"{fmt(mean(diag['cos_cls_manifold']))}   "
                  f"<- около нуля = классификатор не уводит с данных")
            rows = list(zip(
                bins_by_t(diag["snap_t"], diag["snap_dist"]),
                bins_by_t(diag["snap_t"], diag["cos_cls_manifold"])))
            if rows:
                print("")
                print("разбивка по уровню шума (текст решается при МАЛОМ t):")
                print("      t          замеров   удалённость   cos к данным")
                for (lo, hi, n, dv), (_, _, _, cv) in rows:
                    print(f"  {lo:.1f}-{hi:.1f}   {n:9d}   "
                          f"{fmt(dv):>11}   {fmt(cv):>12}")
        print("=" * 72)

    dump_json({
        "probe": "projected_guidance",
        "dataset": config.data.datasets.datasets_list[0],
        "augmentation_scheme": config.cond_encoder.augmentation_scheme,
        "diffusion_checkpoint": runner.loaded_checkpoint_path,
        "step": runner.step,
        "grad_mode": mode,
        "snap_alpha": snap_alpha if mode.startswith("snap") else None,
        "bank_split": bank_split if bank is not None else None,
        "snap_dist_mean": mean(diag["snap_dist"]),
        "cos_cls_manifold_mean": mean(diag["cos_cls_manifold"]),
        "snap_dist_by_t": bins_by_t(diag["snap_t"], diag["snap_dist"]),
        "cos_cls_manifold_by_t": bins_by_t(diag["snap_t"],
                                           diag["cos_cls_manifold"]),
        "scale_mode": scale_mode,
        "target_ratio": target_ratio,
        "lambda": lam,
        "sigmoid_factor": sigmoid_factor,
        "tangent_dim": tangent_dim if bank is not None else None,
        "knn": knn if bank is not None else None,
        "bank_size": int(bank.bank.shape[0]) if bank is not None else None,
        "random_tangent_fraction": bank.random_tangent_fraction if bank else None,
        "solver_steps": runner.dynamic.N,
        "n_texts": len(gen_texts),
        "metrics": metrics,
        "rel_norm_mean": mean(diag["rel_norm"]),
        "tangent_frac_mean": mean(diag["tangent_frac"]),
        # ВСЕ тексты, а не двадцать. Эффект метода мал: свип 28.10 дал
        # +0.0041 bert у alpha=0.25 против alpha=0, а непарный доверительный
        # интервал при n=1000 и разбросе по примеру ~0.08 равен +-0.005. То
        # есть по агрегату эффект НЕ отличим от нуля, хотя прогоны идут на
        # одних промптах и одном сиде, и ПАРНОЕ сравнение на порядок точнее.
        # Парную статистику нельзя посчитать без значений по примерам, а
        # условный прирост (F1 со своей целью минус F1 с переставленной) --
        # тем более: он и есть метрика, не зависящая ни от беглости, ни от
        # копирования промпта, и именно её придётся предъявлять.
        "texts": [{"SRC": src_texts[i], "TRG": trg_texts[i], "GEN": gen_texts[i]}
                  for i in range(len(gen_texts))],
    }, out_path)


if __name__ == "__main__":
    main()
