"""Guidance к ОДНОЙ цели: решающий эксперимент о причине провала.

ЗАЧЕМ ЭТОТ ПРОГОН
-----------------
На провал classifier guidance есть два объяснения, и они разные.

  (A) МНОГОМОДАЛЬНОСТЬ ЦЕЛИ. Условие «любое корректное продолжение этого
      промпта» задает не точку, а множество далеко разнесенных мод. Градиент
      классификатора при большом шуме -- это взвешенное среднее направлений ко
      всем модам, то есть указывает в барицентр, где данных нет.

  (B) ГЕОМЕТРИЯ ПРОСТРАНСТВА. Энкодинги лежат кластерами с пустотами между
      ними. Любой шаг, достаточный чтобы изменить выход, выбрасывает латент
      туда, где ни декодер, ни классификатор не определены.

Эти версии дают РАЗНЫЕ предсказания, и различить их можно одним прогоном. Здесь
классификатор заменяется на reconstruction guidance к одной фиксированной цели
z: градиент логарифма гауссова правдоподобия N(z; x_0(x_t), gamma^2 I). Механизм
тот же самый -- градиент лог-вероятности по x_t, добавленный к score безусловной
модели, -- но многомодальность цели равна нулю по построению: цель одна точка.

  не работает  -> дело НЕ в многомодальности. Даже с идеальной одномодальной
                  целью через это пространство нельзя пройти. Побеждает (B).
  работает     -> механика исправна, проблема в постановке цели. Побеждает (A).
  частично     -> обе причины, и прогон дает их количественное разложение.

РЕЖИМЫ ЦЕЛИ (PROBE_TARGET)
--------------------------
  oracle   -- z = энкодинг истинного продолжения. Верхняя граница: столько
              качества механизм способен выдать при идеально заданной цели.
  shuffled -- z = энкодинг продолжения СОСЕДНЕГО примера. Отрицательный
              контроль, и он важнее oracle: если guidance действительно
              управляет, метрики против истинного референса обязаны УПАСТЬ, а
              против цели, к которой вели, -- вырасти. Это отличает «guidance
              ничего не делает» от «guidance двигает, но не туда».
  json     -- z = энкодинг генерации другой модели (например genie) из
              сохраненного generated_texts/<prefix>/*.json. Практичный вариант
              двухэтапной схемы: сначала выбрать моду, потом вести к ней.

РЕЖИМЫ ЯКОБИАНА (PROBE_JACOBIAN)
--------------------------------
  identity -- d x_0 / d x_t приближается (1/mu_t) I. Градиент выходит
              пропорционален (z - x_0(x_t)): от текущей оценки сети к цели.
              Дешево, один лишний тензор на шаг.
  full     -- честный backward через denoising network. Дороже примерно вдвое
              по времени шага и требует меньшего батча.

ЧТО СМОТРЕТЬ В РЕЗУЛЬТАТЕ
-------------------------
Две группы метрик: vs TRG (истинный референс) и vs STEER (текст, к которому
вели). В режиме oracle они совпадают. В режиме shuffled они расходятся, и
именно их расхождение -- доказательство, что механизм работает.

ЗАПУСК
------
    ARCH_TYPE=unconditional DATASET=wikipedia CHECKPOINT_STEP=87500 \
    PROBE_TARGET=oracle PROBE_RECON_SCALE=1.0 ./probe_recon_guidance.sh

Настройки через окружение:
    PROBE_TARGET      (oracle)  oracle | shuffled | json
    PROBE_TARGET_JSON ()        путь к json при PROBE_TARGET=json
    PROBE_JACOBIAN    (identity) identity | full
    PROBE_RECON_SCALE (1.0)     сила guidance. 0 -- БАЗОВАЯ ЛИНИЯ: чистая
                                безусловная генерация, прогнанная через те же
                                метрики, те же сиды и то же число текстов.
                                Без нее числа oracle не читаются: непонятно,
                                пол это или уже результат
    PROBE_GAMMA2      (1.0)     дисперсия гауссова правдоподобия цели
    PROBE_NUM_TEXTS   (1000)    сколько текстов сгенерировать
    PROBE_BATCH_SIZE  (64)      размер батча
    PROBE_OUT         (...)     путь к json с результатом
"""
import json as _json
import os
import time

import torch
import torch.distributed as dist

from create_config import create_config, metrics_for_mode
from estimation_utils.metrics import compute_metric
from probes.common import (
    build_probe_runner, dump_json, encode_batch, latents_to_tokens, log,
    masked_norm, probe_loader, progress, reverse_process, content_mask,
)
from utils.util import parse, set_seed


def make_recon_guidance(runner, scale, jacobian, gamma2, diag):
    """Градиент log N(z; x_0(x_t), gamma^2 I) по x_t. Цель задается на батч."""
    state = {"target": None}

    def fn(x_t, t, out, cond=None, cond_mask=None, attention_mask=None,
           x_0_self_cond=None, **kwargs):
        target = state["target"]
        if target is None:
            return None
        if scale == 0.0:
            # БАЗОВАЯ ЛИНИЯ lambda=0. Возврат None, а не нулевого тензора:
            # обертка тогда отдает score нетронутым, тогда как score + 0*grad
            # заставило бы пересчитать x_0 по алгебраической формуле и дало бы
            # расхождение на уровне арифметики с плавающей точкой. База обязана
            # быть ТОЧНО безусловной генерацией, иначе ее не с чем сравнивать.
            return None
        params = runner.dynamic.marginal_params(t)
        mu = params["mu"].view(-1, 1, 1)

        if jacobian == "identity":
            grad = (target - out["x_0"]) / (gamma2 * mu)
        else:
            # Честный backward через denoising network. Берется
            # score_estimator, а не DDP-обертка: синхронизация градиентов здесь
            # не нужна и на одном процессе она бы просто мешала.
            with torch.enable_grad():
                xg = x_t.detach().clone().requires_grad_(True)
                x0 = runner.score_estimator(
                    x_t=xg, time_t=t,
                    cond=cond if runner.config.is_conditional else None,
                    attention_mask=attention_mask,
                    cond_mask=cond_mask if runner.config.is_conditional else None,
                    x_0_self_cond=x_0_self_cond,
                )
                logp = -((target - x0) ** 2).sum() / (2.0 * gamma2)
                grad = torch.autograd.grad(logp, xg)[0]

        g = scale * grad
        # Диагностика силы: какую долю шага дает guidance. Нужна, чтобы отличить
        # «scale подобран слишком мал» от «направление бесполезно».
        with torch.no_grad():
            n_g = g.flatten(1).norm(dim=1)
            n_s = out["score"].flatten(1).norm(dim=1).clamp_min(1e-12)
            diag["rel_norm"] += (n_g / n_s).cpu().tolist()
        return g

    fn.state = state
    return fn


def _load_target_texts(path):
    with open(path, encoding="utf-8") as f:
        rows = _json.load(f)
    return rows


def main():
    log("старт: собираю конфиг")
    args = parse()
    config = create_config(args)

    if config.architecture_type != "unconditional":
        raise SystemExit(
            "probe_recon_guidance управляет БЕЗУСЛОВНОЙ диффузией -- той же, "
            "что лежит под classifier guidance. Запускайте с "
            "ARCH_TYPE=unconditional, иначе сравнение с guidance некорректно.")

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    config.local_rank = rank
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", init_method="env://",
                            world_size=world_size, rank=rank)
    dist.barrier()
    config.training.batch_size_per_gpu = config.training.batch_size // dist.get_world_size()

    target_mode = os.environ.get("PROBE_TARGET", "oracle")
    if target_mode not in ("oracle", "shuffled", "json"):
        raise SystemExit(f"PROBE_TARGET={target_mode!r}: ожидается oracle | shuffled | json")
    target_json = os.environ.get("PROBE_TARGET_JSON", "")
    if target_mode == "json" and not target_json:
        raise SystemExit("PROBE_TARGET=json требует PROBE_TARGET_JSON=<путь>")

    jacobian = os.environ.get("PROBE_JACOBIAN", "identity")
    if jacobian not in ("identity", "full"):
        raise SystemExit(f"PROBE_JACOBIAN={jacobian!r}: ожидается identity | full")

    scale = float(os.environ.get("PROBE_RECON_SCALE", "1.0"))
    gamma2 = float(os.environ.get("PROBE_GAMMA2", "1.0"))
    num_texts = int(os.environ.get("PROBE_NUM_TEXTS", "1000"))
    batch_size = int(os.environ.get("PROBE_BATCH_SIZE", "64"))
    out_path = os.environ.get(
        "PROBE_OUT",
        f"probe_results/recon_guidance-{config.data.datasets.datasets_list[0]}"
        f"-{target_mode}-{jacobian}-s{scale:g}.json")

    # Безусловная диффузия в пайплайне промпт не получает, но здесь он нужен:
    # collate_fn обязан собрать input_ids_src, иначе нечего кодировать для SRC.
    # Сама диффузия при этом остается безусловной -- is_conditional не трогаем,
    # и calc_score по-прежнему подает в сеть cond=None. Управляет только
    # reconstruction guidance, ровно как у classifier guidance.
    config.is_pipeline_conditional = True
    # create_config выбрал метрики ПО РЕЖИМУ: у unconditional это mauve/div/ppl.
    # Нам нужны те же bleu/rouge/bert-score, что в таблицах подходов, иначе
    # результат этого прогона не с чем сравнивать.
    config.data.datasets.metrics[config.data.datasets.datasets_list[0]] = \
        metrics_for_mode(True)

    set_seed(config.seed)
    runner = build_probe_runner(config)

    if dist.get_rank() == 0:
        print("=" * 72)
        print("PROBE: reconstruction guidance к одной цели")
        print("=" * 72)
        print(f"  чекпоинт          : {runner.loaded_checkpoint_path}")
        print(f"  шаг               : {runner.step}")
        print(f"  цель              : {target_mode}"
              + (f" ({target_json})" if target_mode == "json" else ""))
        print(f"  якобиан           : {jacobian}")
        print(f"  scale / gamma^2   : {scale} / {gamma2}")
        print(f"  текстов           : {num_texts} (батч {batch_size})")
        print("=" * 72)

    diag = {"rel_norm": []}
    guidance_fn = make_recon_guidance(runner, scale, jacobian, gamma2, diag)

    target_rows = _load_target_texts(target_json) if target_mode == "json" else None
    loader = probe_loader(runner, split="test", batch_size=batch_size)

    gen_texts, trg_texts, src_texts, steer_texts = [], [], [], []
    started = time.time()
    seen = 0

    for b_idx, batch in enumerate(loader):
        if len(gen_texts) >= num_texts:
            break
        batch, src_x, trg_x = encode_batch(runner, batch)
        bsz = len(batch["text_trg"])
        cond_mask = batch.get("attention_mask_src")

        if target_mode == "oracle":
            target = trg_x
            steer = list(runner.tokenizer.batch_decode(
                batch["input_ids_trg"], skip_special_tokens=True))
        elif target_mode == "shuffled":
            target = torch.roll(trg_x, shifts=1, dims=0)
            decoded = list(runner.tokenizer.batch_decode(
                batch["input_ids_trg"], skip_special_tokens=True))
            steer = [decoded[-1]] + decoded[:-1]
        else:
            chunk = target_rows[seen:seen + bsz]
            if len(chunk) < bsz:
                break
            # Порядок обязан совпадать: загрузчик идет с shuffle=False, но если
            # json снят с другого сплита или другой нарезки -- молча сопоставлять
            # нельзя, это дало бы бессмысленную цель и правдоподобные метрики.
            for i, row in enumerate(chunk):
                if "SRC" in row and row["SRC"] != batch["text_src"][i]:
                    raise SystemExit(
                        f"PROBE_TARGET_JSON не выровнен с тестовым сплитом: "
                        f"пример {seen + i} имеет другой SRC. Сверьте датасет и "
                        f"split_scheme, которыми снят json.")
            steer = [row["GEN"] for row in chunk]
            tok = runner.tokenizer(
                steer, add_special_tokens=True, padding="max_length",
                truncation=True, max_length=config.data.max_sequence_len,
                return_tensors="pt", return_attention_mask=True,
                return_token_type_ids=False)
            with torch.no_grad():
                target = runner.encoder(
                    input_ids=tok["input_ids"].to(runner.device),
                    attention_mask=tok["attention_mask"].to(runner.device))

        guidance_fn.state["target"] = target

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
        steer_texts += steer
        seen += bsz

        if dist.get_rank() == 0 and b_idx == 0:
            mask = content_mask(runner, batch["input_ids_trg"])
            print(f"\n  ||x_0|| цели (медиана): "
                  f"{masked_norm(target, mask).median().item():.3f}")
            for i in range(min(3, bsz)):
                print(f"\n  --- пример {i + 1} ---")
                print(f"  SRC   : {src_texts[i][:160]}")
                print(f"  STEER : {steer_texts[i][:160]}")
                print(f"  TRG   : {trg_texts[i][:160]}")
                print(f"  GEN   : {gen_texts[i][:160]}")

        rel = diag["rel_norm"]
        progress(min(len(gen_texts), num_texts), num_texts, started, "текстов",
                 extra=(f"||scale*grad||/||score|| = "
                        f"{sum(rel) / len(rel):.3f}" if rel else ""))

    gen_texts = gen_texts[:num_texts]
    trg_texts = trg_texts[:num_texts]
    src_texts = src_texts[:num_texts]
    steer_texts = steer_texts[:num_texts]

    metric_names = config.data.datasets.metrics[
        config.data.datasets.datasets_list[0]]["metrics"]

    def metrics_against(refs, label):
        res = {}
        for i, name in enumerate(metric_names, start=1):
            # bert-score тянет DeBERTa-xlarge и на тысяче текстов считается
            # минуты. Без отметки на каждую метрику лог молчит весь этот блок,
            # уже ПОСЛЕ генерации -- самое обидное место для подозрения в зависании.
            log(f"метрики {label}: {name} ({i}/{len(metric_names)})...")
            res[name] = compute_metric(
                name, predictions=gen_texts, references=refs, sources=src_texts)
        return res

    vs_trg = metrics_against(trg_texts, "vs TRG")
    vs_steer = (metrics_against(steer_texts, "vs STEER")
                if target_mode != "oracle" else vs_trg)

    if dist.get_rank() == 0:
        def show(title, d):
            print(f"\n----- {title} -----")
            for k, v in d.items():
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        print(f"{kk}: {vv:0.5f}")
                else:
                    print(f"{k}: {v:0.5f}")

        print("\n" + "=" * 72)
        show("метрики против ИСТИННОГО референса (TRG)", vs_trg)
        if target_mode != "oracle":
            show("метрики против ЦЕЛИ, к которой вели (STEER)", vs_steer)
            print("\nРасхождение этих двух групп -- доказательство, что механизм "
                  "управляет.\nЕсли они совпадают, guidance не делает ничего.")
        rel = diag["rel_norm"]
        if rel:
            print(f"\n||scale*grad|| / ||score||: среднее "
                  f"{sum(rel) / len(rel):.4f}, максимум {max(rel):.4f}")
        print("=" * 72)

        dump_json({
            "probe": "recon_guidance",
            "dataset": config.data.datasets.datasets_list[0],
            "diffusion_checkpoint": runner.loaded_checkpoint_path,
            "step": runner.step,
            "target_mode": target_mode,
            "target_json": target_json or None,
            "jacobian": jacobian,
            "recon_scale": scale,
            "gamma2": gamma2,
            "solver_steps": runner.dynamic.N,
            "n_texts": len(gen_texts),
            "metrics_vs_trg": vs_trg,
            "metrics_vs_steer": vs_steer,
            "rel_norm_mean": (sum(diag["rel_norm"]) / len(diag["rel_norm"]))
                             if diag["rel_norm"] else None,
            "examples": [
                {"SRC": src_texts[i], "STEER": steer_texts[i],
                 "TRG": trg_texts[i], "GEN": gen_texts[i]}
                for i in range(min(20, len(gen_texts)))
            ],
        }, out_path)


if __name__ == "__main__":
    main()
