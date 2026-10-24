"""Кривая предрешенности: с какого момента обратный процесс уже нельзя свернуть.

ЗАЧЕМ ЭТОТ ПРОГОН
-----------------
probe_signal_survival.py дает верхний край окна управления: уровень шума, выше
которого классификатор не несет информации. Нужен второй край -- уровень, ниже
которого менять уже нечего, потому что текст фактически определен.

Измеряется ветвлением траектории. Обратный процесс гоняется один раз до конца и
запоминается в нескольких точках. Затем из каждой точки он доигрывается заново с
ДРУГИМ сидом, то есть со всем остальным шумом солвера другим. Если итоговый текст
после ветвления совпадает с исходным -- в этой точке результат уже предрешен, и
никакая коррекция score после нее на выход не повлияет. Если расходится -- точка
еще в зоне, где выбор не сделан.

  divergence(t) = 1 - unigram_F1(базовый текст, текст после ветвления)

На t = T (старт из шума) divergence максимальна: это два независимых сэмпла по
одному промпту, то есть естественный пол расхождения для данного подхода. На
t -> 0 divergence стремится к нулю. Точка, где кривая выходит на ноль, и есть
tau_commit.

ГЛАВНЫЙ РИСУНОК СТАТЬИ
----------------------
accuracy(t) из первого прогона и divergence(t) из этого -- на одной оси шума.
Утверждение работы: их носители не пересекаются. Там, где классификатор еще
что-то различает, выход уже предрешен; там, где выход еще пластичен,
классификатор не отличает правильное продолжение от чужого. Окно управления
пусто -- значит, подбор lambda не мог помочь ни при каком значении, и дело не в
настройке.

Прогон имеет смысл для ЛЮБОГО подхода: ARCH_TYPE=genie дает кривую для рабочей
условной модели (контроль: у нее окно должно быть непустым, иначе измерение
ничего не меряет), ARCH_TYPE=unconditional и guidance -- для исследуемой.

ЗАПУСК
------
    ARCH_TYPE=unconditional DATASET=wikipedia CHECKPOINT_STEP=87500 \
    ./probe_commitment.sh

Настройки через окружение:
    PROBE_BATCHES      (4)   сколько батчей теста усреднять
    PROBE_BATCH_SIZE   (64)  размер батча
    PROBE_BRANCH_EVERY (5)   шаг сетки точек ветвления по индексу шага солвера
    PROBE_OUT          (...) путь к json
"""
import os
import time

import torch
import torch.distributed as dist

from create_config import create_config
from probes.common import (
    build_probe_runner, dump_json, encode_batch, latents_to_tokens, log,
    make_timesteps, probe_loader, progress, reverse_process, unigram_f1,
)
from utils.util import parse, set_seed


def _tau_commit(rows, threshold=0.05):
    """Наименьший уровень шума, на котором ветвление еще меняет выход.

    Порог 0.05 по divergence: ниже него тексты совпадают с точностью до
    единичных токенов, то есть результат уже предрешен.
    """
    movable = [r["t_state"] for r in rows if r["divergence"] >= threshold]
    return min(movable) if movable else None


def main():
    log("старт: собираю конфиг")
    args = parse()
    config = create_config(args)

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    config.local_rank = rank
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", init_method="env://",
                            world_size=world_size, rank=rank)
    dist.barrier()
    config.training.batch_size_per_gpu = config.training.batch_size // dist.get_world_size()

    n_batches = int(os.environ.get("PROBE_BATCHES", "4"))
    batch_size = int(os.environ.get("PROBE_BATCH_SIZE", "64"))
    branch_every = int(os.environ.get("PROBE_BRANCH_EVERY", "5"))
    out_path = os.environ.get(
        "PROBE_OUT",
        f"probe_results/commitment-{config.data.datasets.datasets_list[0]}"
        f"-{config.architecture_type}.json")

    set_seed(config.seed)
    runner = build_probe_runner(config)

    n_steps = runner.dynamic.N
    branch_idx = list(range(0, n_steps, branch_every))
    timesteps = make_timesteps(runner)

    if dist.get_rank() == 0:
        print("=" * 72)
        print("PROBE: кривая предрешенности")
        print("=" * 72)
        print(f"  подход            : {config.architecture_type}")
        print(f"  чекпоинт          : {runner.loaded_checkpoint_path}")
        print(f"  шаг               : {runner.step}")
        print(f"  шагов солвера N   : {n_steps}")
        print(f"  точек ветвления   : {len(branch_idx)} (каждый {branch_every}-й)")
        print(f"  батчей x размер   : {n_batches} x {batch_size}")
        print("=" * 72)

    stats = {i: {"f1": [], "exact": [], "latent_rel": []} for i in branch_idx}

    def mean(xs):
        return float(sum(xs) / len(xs)) if xs else float("nan")

    def build_rows():
        out = []
        for idx in branch_idx:
            s = stats[idx]
            f1 = mean(s["f1"])
            out.append({
                "step_idx": idx,
                "t_state": float(timesteps[idx + 1]),
                "f1": f1,
                "divergence": 1.0 - f1,
                "exact_match": mean(s["exact"]),
                "latent_rel_dist": mean(s["latent_rel"]),
                "n": len(s["f1"]),
            })
        return out

    def save(rows, partial):
        # Промежуточное сохранение после каждого батча: задание, снятое по
        # лимиту, иначе не оставляло бы ничего, хотя кривая по двум батчам из
        # четырех уже читается.
        dump_json({
            "probe": "commitment",
            "partial": partial,
            "dataset": config.data.datasets.datasets_list[0],
            "architecture_type": config.architecture_type,
            "diffusion_checkpoint": runner.loaded_checkpoint_path,
            "step": runner.step,
            "solver_steps": n_steps,
            "branch_every": branch_every,
            "n_batches": n_batches,
            "batch_size": batch_size,
            "tau_commit_div005": _tau_commit(rows),
            "rows": rows,
        }, out_path)

    loader = probe_loader(runner, split="test", batch_size=batch_size)
    started = time.time()
    # Базовая траектория плюс хвост из каждой точки ветвления. Считаем в шагах
    # солвера -- это и есть единица работы, по ней честно строится остаток.
    steps_per_batch = n_steps + sum(n_steps - i - 1 for i in branch_idx)
    total_steps = n_batches * steps_per_batch
    done_steps = 0
    log(f"считаю {total_steps} шагов солвера "
        f"({n_batches} батчей x {steps_per_batch})")

    for b_idx, batch in enumerate(loader):
        if b_idx >= n_batches:
            break
        batch, src_x, _ = encode_batch(runner, batch)
        bsz = len(batch["text_trg"])
        cond_mask = batch.get("attention_mask_src")

        # --- базовая траектория со снимками ---------------------------------
        snapshots = []
        set_seed(config.seed + b_idx)
        base_lat = reverse_process(
            runner, batch_size=bsz, cond_x=src_x, cond_mask=cond_mask,
            attention_mask=None, record_at=branch_idx, record_out=snapshots,
            show_steps=(b_idx == 0))
        done_steps += n_steps
        progress(done_steps, total_steps, started, "шагов солвера",
                 extra=f"батч {b_idx + 1}: базовая траектория готова")
        base_text, base_tok = latents_to_tokens(
            runner, base_lat, cond_x=src_x, cond_mask=cond_mask)

        # --- ветвления ------------------------------------------------------
        for snap in snapshots:
            idx = snap["idx"]
            # Другой сид -- значит весь оставшийся шум солвера другой. Сид
            # привязан к батчу и к точке ветвления, чтобы прогон был
            # воспроизводим, а ветки не совпадали между собой.
            set_seed(config.seed + 100000 + b_idx * 1000 + idx)
            alt_lat = reverse_process(
                runner, batch_size=bsz, cond_x=src_x, cond_mask=cond_mask,
                attention_mask=None,
                resume={"z": snap["z"],
                        "x_0_self_cond": snap["x_0_self_cond"],
                        "start_idx": snap["start_idx"]})
            alt_text, alt_tok = latents_to_tokens(
                runner, alt_lat, cond_x=src_x, cond_mask=cond_mask)

            for a, b in zip(base_tok, alt_tok):
                stats[idx]["f1"].append(unigram_f1(a, b))
                stats[idx]["exact"].append(1.0 if a == b else 0.0)

            rel = ((alt_lat - base_lat).flatten(1).norm(dim=1)
                   / base_lat.flatten(1).norm(dim=1).clamp_min(1e-12))
            stats[idx]["latent_rel"] += rel.detach().cpu().tolist()

            done_steps += n_steps - idx - 1
            f1s = stats[idx]["f1"]
            progress(done_steps, total_steps, started, "шагов солвера",
                     extra=f"ветка с шага {idx}: divergence "
                           f"{1.0 - sum(f1s) / len(f1s):.3f}")

            if dist.get_rank() == 0 and b_idx == 0 and idx in (branch_idx[0], branch_idx[len(branch_idx) // 2], branch_idx[-1]):
                print(f"\n--- ветвление на шаге {idx} (t_state="
                      f"{float(timesteps[snap['start_idx']]):.3f}) ---")
                print(f"  база : {base_text[0][:160]}")
                print(f"  ветка: {alt_text[0][:160]}")

        log(f"батч {b_idx + 1}/{n_batches} готов")
        save(build_rows(), partial=(b_idx + 1 < n_batches))

    rows = build_rows()

    if dist.get_rank() == 0:
        print("\n" + "=" * 84)
        print(f"{'step':>5} {'t_state':>8} {'divergence':>11} {'f1':>8} "
              f"{'exact':>8} {'latent_rel':>11}")
        print("-" * 84)
        for r in rows:
            print(f"{r['step_idx']:>5} {r['t_state']:>8.3f} "
                  f"{r['divergence']:>11.4f} {r['f1']:>8.4f} "
                  f"{r['exact_match']:>8.4f} {r['latent_rel_dist']:>11.4f}")
        print("=" * 84)

        print(f"\ntau_commit (divergence >= 0.05): {_tau_commit(rows)}")
        print("Ниже этого уровня шума выход уже предрешен: любая коррекция "
              "score после этой точки\nна итоговый текст не влияет.\n")

    save(rows, partial=False)


if __name__ == "__main__":
    main()
