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

ЧТО ПРОВЕРЯЕТСЯ
---------------
1. Монотонность: растет косинус -- растет качество.
2. Совпадение осей. prefix дает ТОЧНУЮ информацию о немногих позициях,
   quantize -- ГРУБУЮ обо всех. Если обе ложатся на одну кривую, качество
   определяется количеством информации, а не тем, как она распределена.
3. Попадание реального классификатора на кривую. Если точка (0.009, 52%)
   лежит на кривой, построенной оракульными деградациями, то классификатор
   ОПТИМАЛЕН для своего объема информации, и провал guidance на продолжении по
   промпту -- информационный предел, а не дефект метода.

Пункт 3 -- главный. Он превращает частный отрицательный результат в правило:
улучшать guidance надо, увеличивая информативность условия, а не подбирая
lambda, не меняя архитектуру классификатора и не чиня геометрию.

РЕЖИМЫ (PROBE_DEGRADE)
----------------------
  none     полная цель -- верхняя точка оси
  prefix   PROBE_LEVEL = сколько содержательных позиций открыто
  quantize PROBE_LEVEL = сколько центроидов (k-means по позициям банка)
  seqquant PROBE_LEVEL = сколько центроидов по ПОСЛЕДОВАТЕЛЬНОСТЯМ. Это и есть
           «веди к теме»: оракульный классификатор темы дал бы градиент ровно
           в центроид своего кластера, поэтому отдельно обучать его не нужно
  rank     PROBE_LEVEL = сколько главных компонент банка оставить

ЗАПУСК
------
    ARCH_TYPE=unconditional CHECKPOINT_STEP=87500 \
    PROBE_DEGRADE=prefix PROBE_LEVEL=8 ./probe_information_axis.sh

Настройки через окружение:
    PROBE_DEGRADE      (prefix)  none | prefix | quantize | seqquant | rank
    PROBE_LEVEL        (8)       уровень деградации, смысл зависит от режима
    PROBE_TARGET_RATIO (0.1)     доля шага ||lam*grad||/||score||
    PROBE_BANK         (8192)    векторов в банке
    PROBE_NUM_TEXTS    (1000)    сколько текстов сгенерировать
    PROBE_BATCH_SIZE   (64)      размер батча
    PROBE_OUT          (...)     путь к json
"""
import os
import time

import torch
import torch.distributed as dist

from create_config import create_config, metrics_for_mode
from estimation_utils.metrics import compute_metric
from probes.common import (
    build_probe_runner, content_mask, dump_json, encode_batch,
    latents_to_tokens, log, masked_cosine, probe_loader, progress,
    reverse_process,
)
from probes.degrade import degrade_target, estimated_bits
from probes.manifold import LatentBank
from utils.util import parse, set_seed


def make_axis_guidance(runner, target_ratio, diag):
    """Reconstruction guidance к ДЕГРАДИРОВАННОЙ цели.

    Якобиан приближается identity, сила задается долей шага -- ровно та
    конфигурация, на которой оракульная цель дала bleu 0.938. Менять ее нельзя:
    точки оси обязаны отличаться ТОЛЬКО информативностью цели.
    """
    state = {"target": None, "true": None, "mask": None}

    def fn(x_t, t, out, **kwargs):
        if state["target"] is None:
            return None
        params = runner.dynamic.marginal_params(t)
        mu = params["mu"]

        grad = (state["target"] - out["x_0"]) / mu
        if state["mask"] is not None:
            grad = grad * state["mask"].unsqueeze(-1).to(grad.dtype)

        n_s = out["score"].flatten(1).norm(dim=1).clamp_min(1e-12)
        n_g = grad.flatten(1).norm(dim=1).clamp_min(1e-12)
        g = (target_ratio * n_s / n_g).view(-1, 1, 1) * grad

        with torch.no_grad():
            # ГЛАВНАЯ ИЗМЕРЯЕМАЯ ВЕЛИЧИНА: насколько направление, которое задает
            # деградированное условие, совпадает с направлением к ИСТИННОМУ
            # продолжению. Это ось x всей работы.
            true_dir = mu * state["true"] - x_t
            diag["cos_to_true"] += masked_cosine(
                g, true_dir, state["content"]).cpu().tolist()
            diag["rel_norm"] += (g.flatten(1).norm(dim=1) / n_s).cpu().tolist()
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

    mode = os.environ.get("PROBE_DEGRADE", "prefix")
    if mode not in ("none", "prefix", "quantize", "seqquant", "rank"):
        raise SystemExit(f"PROBE_DEGRADE={mode!r}: ожидается "
                         f"none | prefix | quantize | seqquant | rank")
    level = int(os.environ.get("PROBE_LEVEL", "8"))
    target_ratio = float(os.environ.get("PROBE_TARGET_RATIO", "0.1"))
    bank_size = int(os.environ.get("PROBE_BANK", "8192"))
    num_texts = int(os.environ.get("PROBE_NUM_TEXTS", "1000"))
    batch_size = int(os.environ.get("PROBE_BATCH_SIZE", "64"))
    out_path = os.environ.get(
        "PROBE_OUT",
        f"probe_results/information_axis-{config.data.datasets.datasets_list[0]}"
        f"-{mode}-L{level}-r{target_ratio:g}.json")

    # Промпт самой диффузии не подается (она безусловная), но нужен collate и
    # условные метрики -- иначе точки оси не сравнить с таблицей подходов.
    config.is_pipeline_conditional = True
    config.data.datasets.metrics[config.data.datasets.datasets_list[0]] = \
        metrics_for_mode(True)

    set_seed(config.seed)
    runner = build_probe_runner(config)

    bank = None
    if mode in ("quantize", "seqquant", "rank"):
        bank = LatentBank.from_runner(
            runner, size=bank_size, skip_examples=num_texts,
            batch_size=batch_size)

    if dist.get_rank() == 0:
        print("=" * 72)
        print("PROBE: ось информативности условия")
        print("=" * 72)
        print(f"  чекпоинт          : {runner.loaded_checkpoint_path}")
        print(f"  деградация        : {mode}, уровень {level}")
        print(f"  доля шага         : {target_ratio}")
        bits = estimated_bits(mode, level, config.data.max_sequence_len)
        print(f"  оценка бит о цели : {'нет' if bits is None else f'~{bits:.0f}'}")
        print(f"  текстов           : {num_texts} (батч {batch_size})")
        print("=" * 72)

    diag = {"cos_to_true": [], "rel_norm": [], "target_recovery": []}
    guidance_fn = make_axis_guidance(runner, target_ratio, diag)

    loader = probe_loader(runner, split="test", batch_size=batch_size)
    gen_texts, trg_texts, src_texts, steer_texts = [], [], [], []
    started = time.time()
    desc = ""

    for b_idx, batch in enumerate(loader):
        if len(gen_texts) >= num_texts:
            break
        batch, src_x, trg_x = encode_batch(runner, batch)
        bsz = len(batch["text_trg"])
        cond_mask = batch.get("attention_mask_src")
        content = content_mask(runner, batch["input_ids_trg"])

        z, pos_mask, desc = degrade_target(mode, trg_x, level, bank=bank,
                                           content=content)
        guidance_fn.state.update({"target": z, "true": trg_x,
                                  "mask": pos_mask, "content": content})

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

        c = diag["cos_to_true"]
        progress(min(len(gen_texts), num_texts), num_texts, started, "текстов",
                 extra=f"cos к истине {sum(c) / len(c):.4f}" if c else "")

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

    def mean(xs):
        return float(sum(xs) / len(xs)) if xs else None

    if dist.get_rank() == 0:
        print("\n" + "=" * 72)
        print(f"----- {mode} / уровень {level} / доля {target_ratio} -----")
        for k, v in metrics.items():
            print(f"{k}: {v:0.5f}")
        print(f"\nкосинус с направлением к истине : {mean(diag['cos_to_true']):.5f}")
        print(f"доля шага                        : {mean(diag['rel_norm']):.4f}")
        print(f"сама цель против истины (bert)   : {recovery['bert-score']:.5f}")
        print("=" * 72)
        print("Точка оси: (косинус, качество). Сравнивать с реальным "
              "классификатором:\nего косинус 0.009 при bert 0.5081 и полом 0.4605.")

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
        "cos_to_true": mean(diag["cos_to_true"]),
        "rel_norm_mean": mean(diag["rel_norm"]),
        "examples": [{"SRC": src_texts[i], "STEER": steer_texts[i],
                      "TRG": trg_texts[i], "GEN": gen_texts[i]}
                     for i in range(min(20, len(gen_texts)))],
    }, out_path)


if __name__ == "__main__":
    main()
