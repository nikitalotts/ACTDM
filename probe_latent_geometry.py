"""Геометрия латентного пространства: дырявость в числах, текст против картинок.

ЗАЧЕМ ЭТОТ ПРОГОН
-----------------
Утверждение «латентное пространство картинок гладкое, а текстовое -- кластеры с
пустотами» до сих пор живет в работе как интуиция в тексте. Рецензент попросит
число. Прогон считает безразмерные статистики, которые можно положить на одну
ось для текста и для картинок, несмотря на разную размерность.

ОСНОВНАЯ СТАТИСТИКА: hole ratio
-------------------------------
    H(q) = median_i d(m_ij, bank \\ {i, j}) / median_i d_NN(i)

где j -- точка на расстоянии q-квантиля от i, m_ij -- середина отрезка [i, j],
а d_NN(i) -- расстояние до ближайшего соседа. Числитель -- насколько далеко от
данных лежит середина между двумя точками данных; знаменатель -- характерный
масштаб расстояний, он и делает величину безразмерной.

Сами i и j из поиска ближайшего ИСКЛЮЧЕНЫ, и это принципиально. Иначе для
близкой пары ответом всегда была бы одна из концов, то есть d_pair/2, и
статистика мерила бы не покрытие промежутка, а расстояние внутри пары. Вопрос
же стоит так: населено ли ДРУГИМИ данными пространство между двумя объектами.

  гладкое связное многообразие -- промежуток населен, H порядка единицы;
  разделенные кластеры         -- промежуток пуст, H сильно больше единицы.

ОСТАЛЬНЫЕ СТАТИСТИКИ
--------------------
  tangent_frac -- доля энергии направления «к соседу» в касательном
                  подпространстве, оцененном локальным PCA по kNN. Показывает,
                  сколько у пространства локальных степеней свободы: на гладком
                  многообразии движение к соседу почти целиком касательное.
  local_dim    -- число главных компонент kNN, покрывающих 90% дисперсии.
  flip_rate(e) -- доля позиций, у которых декодер меняет токен при сдвиге
                  латента на e (в единицах медианной нормы латента). Отсюда
                  r_tok -- радиус, на котором меняется 10% токенов: это бюджет,
                  который есть у шага guidance.
  interp       -- декодирование середины отрезка между двумя латентами: доля
                  позиций, совпавших с одним из концов, и уверенность декодера.
                  Это проба на выпуклость валидной области (как в Cosmos).

КАРТИНКИ
--------
Передаются готовым массивом через PROBE_IMAGE_LATENTS -- .npy формы [N, D] или
[N, L, D]. Зависимость от diffusers в репозиторий не тащится: массив снимается
отдельно, рецепт -- в RUNBOOK.md.

ЗАПУСК
------
    ARCH_TYPE=unconditional DATASET=wikipedia CHECKPOINT_STEP=87500 \
    ./probe_latent_geometry.sh

Настройки через окружение:
    PROBE_SPACES         (encodings,embeddings,whitened)  через запятую
    PROBE_IMAGE_LATENTS  ()      путь к .npy с латентами картинок
    PROBE_BANK           (20000) сколько векторов в банке
    PROBE_ANCHORS        (512)   сколько опорных точек для H(q)
    PROBE_WITH_DECODER   (1)     считать ли flip_rate и interp (нужен декодер)
    PROBE_OUT            (...)   путь к json
"""
import os
import time

import torch
import torch.distributed as dist

from create_config import create_config
from probes.common import (
    build_probe_runner, content_mask, dump_json, encode_batch, log,
    probe_loader, progress,
)
from utils.util import parse, set_seed

QUANTILES = [0.01, 0.05, 0.25, 0.50]


def _min_dist_excluding(points, bank, exclude_idx, chunk=4096):
    """Для каждой точки -- расстояние до ближайшей точки банка, кроме указанных.

    points      [P, D]
    bank        [N, D]
    exclude_idx [P, k] индексы банка, которые нельзя брать в ответ
    """
    best = torch.full((points.shape[0],), float("inf"), device=points.device)
    for start in range(0, bank.shape[0], chunk):
        sl = bank[start:start + chunk]
        d = torch.cdist(points, sl)
        # вычеркиваем запрещенные индексы, попавшие в этот кусок
        local = exclude_idx - start
        ok = (local >= 0) & (local < sl.shape[0])
        if ok.any():
            rows = torch.nonzero(ok, as_tuple=False)
            d[rows[:, 0], local[ok]] = float("inf")
        best = torch.minimum(best, d.min(dim=1).values)
    return best


def hole_ratio(bank, n_anchors, knn=32, seed=0):
    """H(q), а также локальная размерность и касательная доля."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = bank.shape[0]
    anchors = torch.randperm(n, generator=g)[:min(n_anchors, n)].to(bank.device)
    a = bank[anchors]

    d_all = torch.cdist(a, bank)
    # расстояние до самой себя обнуляем, чтобы не попасть в ближайшие соседи
    d_all[torch.arange(a.shape[0], device=bank.device), anchors] = float("inf")
    order = d_all.argsort(dim=1)
    d_sorted = torch.gather(d_all, 1, order)

    nn_dist = d_sorted[:, 0]
    med_nn = nn_dist.median()

    out = {"nn_dist_median": float(med_nn.item()), "H": {}, "mid_gap": {}}

    for q in QUANTILES:
        # Сама опорная точка после обнуления расстояния уезжает в конец сортировки,
        # поэтому последний ранг брать нельзя -- партнером оказалась бы она же.
        rank = max(1, int(round(q * (n - 1))))
        rank = min(rank, n - 2)
        partners = order[:, rank]
        mid = 0.5 * (a + bank[partners])
        excl = torch.stack([anchors, partners], dim=1)
        gap = _min_dist_excluding(mid, bank, excl)
        out["mid_gap"][str(q)] = float(gap.median().item())
        out["H"][str(q)] = float((gap.median() / med_nn.clamp_min(1e-12)).item())

    # --- локальная геометрия по kNN -------------------------------------
    k = min(knn, n - 1)
    nbr = bank[order[:, :k]]                      # [A, k, D]
    centered = nbr - nbr.mean(dim=1, keepdim=True)
    # Полный SVD по 768 измерениям на каждую опорную точку дорог; берем
    # подвыборку опорных точек -- дисперсия этих двух величин мала.
    sub = min(64, centered.shape[0])
    sv = torch.linalg.svdvals(centered[:sub].float())     # [sub, min(k, D)]
    energy = sv ** 2
    frac = energy / energy.sum(dim=1, keepdim=True).clamp_min(1e-12)
    cum = frac.cumsum(dim=1)
    local_dim = (cum < 0.9).sum(dim=1).float() + 1

    # Доля направления «к ближайшему соседу» в касательном подпространстве,
    # натянутом на local_dim главных компонент kNN.
    tangent = []
    for i in range(sub):
        d_i = int(local_dim[i].item())
        _, _, vh = torch.linalg.svd(centered[i].float(), full_matrices=False)
        basis = vh[:d_i]                                  # [d_i, D]
        v = (bank[order[i, 0]] - a[i]).float()
        v = v / v.norm().clamp_min(1e-12)
        tangent.append(float((basis @ v).pow(2).sum().item()))

    out["local_dim_mean"] = float(local_dim.mean().item())
    out["tangent_frac_mean"] = float(sum(tangent) / len(tangent))
    out["n_anchors"] = int(a.shape[0])
    out["bank_size"] = int(n)
    return out


@torch.no_grad()
def decoder_probes(runner, seq_latents, cond_x=None, cond_mask=None, seed=0):
    """flip_rate(e), r_tok и проба интерполяции. Требует обученный декодер."""
    torch.manual_seed(seed)
    x0 = seq_latents
    base_logits = runner.pred_logits(x0, cond_x=cond_x, cond_mask=cond_mask)
    base_tok = base_logits.argmax(dim=-1)
    base_conf = base_logits.softmax(dim=-1).max(dim=-1).values.mean().item()

    # Масштаб сдвига -- в единицах медианной нормы латента позиции, иначе
    # число e несопоставимо между пространствами.
    scale = x0.flatten(0, 1).norm(dim=-1).median().clamp_min(1e-12)

    flips = []
    for eps in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0):
        u = torch.randn_like(x0)
        u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        tok = runner.pred_logits(
            x0 + eps * scale * u, cond_x=cond_x, cond_mask=cond_mask).argmax(dim=-1)
        flips.append({"eps": eps, "flip_rate": float((tok != base_tok).float().mean().item())})

    r_tok = None
    for a, b in zip(flips, flips[1:]):
        if a["flip_rate"] < 0.1 <= b["flip_rate"]:
            # линейная интерполяция между двумя замерами
            span = b["flip_rate"] - a["flip_rate"]
            w = (0.1 - a["flip_rate"]) / span if span > 0 else 0.0
            r_tok = a["eps"] + w * (b["eps"] - a["eps"])
            break
    if r_tok is None and flips[0]["flip_rate"] >= 0.1:
        r_tok = flips[0]["eps"]

    # --- интерполяция ----------------------------------------------------
    half = x0.shape[0] // 2
    if half >= 1:
        xa, xb = x0[:half], x0[half:2 * half]
        # Середине отрезка нужен какой-то один промпт, и выбор здесь
        # произволен -- берем промпт первого конца. Значение это имеет только
        # для условного декодера (genie); проба задумана для безусловного, где
        # pred_logits условие все равно отбрасывает.
        ca = cond_x[:half] if cond_x is not None else None
        mid_logits = runner.pred_logits(0.5 * (xa + xb), cond_x=ca, cond_mask=None)
        mid_tok = mid_logits.argmax(dim=-1)
        mid_conf = mid_logits.softmax(dim=-1).max(dim=-1).values.mean().item()
        tok_a = base_tok[:half]
        tok_b = base_tok[half:2 * half]
        agree = ((mid_tok == tok_a) | (mid_tok == tok_b)).float().mean().item()
    else:
        mid_conf, agree = float("nan"), float("nan")

    return {
        "flip_rate": flips,
        "r_tok_at_10pct": r_tok,
        "decoder_conf_endpoints": base_conf,
        "decoder_conf_midpoint": mid_conf,
        "midpoint_token_agreement": agree,
    }


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

    spaces = [s.strip() for s in os.environ.get(
        "PROBE_SPACES", "encodings,embeddings,whitened").split(",") if s.strip()]
    image_path = os.environ.get("PROBE_IMAGE_LATENTS", "")
    bank_size = int(os.environ.get("PROBE_BANK", "20000"))
    n_anchors = int(os.environ.get("PROBE_ANCHORS", "512"))
    with_decoder = os.environ.get("PROBE_WITH_DECODER", "1") == "1"
    out_path = os.environ.get(
        "PROBE_OUT",
        f"probe_results/latent_geometry-{config.data.datasets.datasets_list[0]}.json")

    set_seed(config.seed)
    runner = build_probe_runner(config)

    if dist.get_rank() == 0:
        print("=" * 72)
        print("PROBE: геометрия латентного пространства")
        print("=" * 72)
        print(f"  пространства      : {', '.join(spaces)}"
              + (f" + images({image_path})" if image_path else ""))
        print(f"  банк / опорных    : {bank_size} / {n_anchors}")
        print(f"  пробы декодера    : {'да' if with_decoder else 'нет'}")
        print("=" * 72)

    # --- сбор банка ------------------------------------------------------
    enc_bank, emb_bank, seq_bank, cond_bank = [], [], [], []
    loader = probe_loader(runner, split="test", batch_size=64)

    used_ids, _ = runner.encoder.get_used_ids(encoder_link=config.model.encoder_link)
    emb_w = runner.encoder.encoder.embeddings.word_embeddings.weight.detach()
    emb_mean = emb_w[used_ids].mean(dim=0, keepdim=True)
    emb_std = emb_w[used_ids].std(dim=0, keepdim=True)

    started = time.time()
    for batch in loader:
        collected = sum(b.shape[0] for b in enc_bank)
        if collected >= bank_size:
            break
        if enc_bank and len(enc_bank) % 10 == 0:
            progress(min(collected, bank_size), bank_size, started,
                     "векторов в банке")
        batch, src_x, trg_x = encode_batch(runner, batch)
        mask = content_mask(runner, batch["input_ids_trg"])

        enc_bank.append(trg_x[mask].detach())
        ids = batch["input_ids_trg"][mask]
        emb_bank.append(((emb_w[ids] - emb_mean) / emb_std.clamp_min(1e-12)).detach())

        # Банк ПОСЛЕДОВАТЕЛЬНЫХ латентов нужен только пробам декодера, и он
        # дорог: логиты декодера это [N x 64 x 28996] в fp32, то есть ~1 ГБ на
        # 128 примеров, и в decoder_probes одновременно живут два таких тензора.
        if sum(s.shape[0] for s in seq_bank) < 128:
            seq_bank.append(trg_x.detach())
            cond_bank.append(src_x.detach() if src_x is not None else None)

    enc = torch.cat(enc_bank)[:bank_size].float()
    emb = torch.cat(emb_bank)[:bank_size].float()
    seq = torch.cat(seq_bank)[:128].float()
    cond = (torch.cat([c for c in cond_bank if c is not None])[:128].float()
            if any(c is not None for c in cond_bank) else None)

    if dist.get_rank() == 0:
        print(f"  банк собран за {time.time() - started:.0f} c: "
              f"encodings {tuple(enc.shape)}, embeddings {tuple(emb.shape)}, "
              f"sequences {tuple(seq.shape)}")

    # ZCA-отбеливание энкодингов. Статистики считаются на ПЕРВОЙ половине
    # банка, применяются ко всему: иначе отбеливание подгонялось бы под те же
    # точки, на которых потом меряется геометрия.
    def whiten(x):
        h = x[: x.shape[0] // 2]
        mu = h.mean(dim=0, keepdim=True)
        cov = torch.cov((h - mu).T) + 1e-5 * torch.eye(x.shape[1], device=x.device)
        evals, evecs = torch.linalg.eigh(cov)
        w = evecs @ torch.diag(evals.clamp_min(1e-8).rsqrt()) @ evecs.T
        return (x - mu) @ w

    banks = {}
    if "encodings" in spaces:
        banks["encodings"] = enc
    if "embeddings" in spaces:
        banks["embeddings"] = emb
    if "whitened" in spaces:
        banks["whitened"] = whiten(enc)
    if image_path:
        import numpy as np
        arr = np.load(image_path)
        t = torch.from_numpy(arr).float()
        if t.dim() == 3:
            t = t.flatten(0, 1)
        banks["images"] = t[:bank_size].to(runner.device)

    results = {}
    geom_started = time.time()
    for i, (name, bank) in enumerate(banks.items(), start=1):
        log(f"считаю геометрию: {name} {tuple(bank.shape)}")
        results[name] = hole_ratio(bank, n_anchors=n_anchors, seed=config.seed)
        results[name]["dim"] = int(bank.shape[1])
        progress(i, len(banks), geom_started, "пространств",
                 extra=f"{name}: H(0.5) = {results[name]['H']['0.5']:.2f}")

    if with_decoder:
        log("пробы декодера (flip_rate, интерполяция)")
        results.setdefault("encodings", {})["decoder"] = decoder_probes(
            runner, seq, cond_x=cond if runner.config.decoder.is_conditional else None,
            seed=config.seed)

    if dist.get_rank() == 0:
        print("\n" + "=" * 96)
        head = f"{'пространство':>14} {'dim':>6} {'nn_dist':>9} " + \
               " ".join(f"{'H(' + str(q) + ')':>9}" for q in QUANTILES) + \
               f" {'tangent':>8} {'loc_dim':>8}"
        print(head)
        print("-" * 96)
        for name, r in results.items():
            if "H" not in r:
                continue
            hs = " ".join(f"{r['H'][str(q)]:>9.2f}" for q in QUANTILES)
            print(f"{name:>14} {r['dim']:>6} {r['nn_dist_median']:>9.3f} {hs} "
                  f"{r['tangent_frac_mean']:>8.3f} {r['local_dim_mean']:>8.1f}")
        print("=" * 96)
        print("H порядка единицы -- промежуток между данными населен (гладко).")
        print("H много больше единицы -- между кластерами пусто (дыры).")
        if with_decoder:
            d = results["encodings"]["decoder"]
            print(f"\nr_tok (10% смененных токенов): {d['r_tok_at_10pct']}")
            print(f"уверенность декодера: концы {d['decoder_conf_endpoints']:.3f}, "
                  f"середина {d['decoder_conf_midpoint']:.3f}")
            print(f"середина совпала с одним из концов: "
                  f"{d['midpoint_token_agreement']:.3f} позиций")

        dump_json({
            "probe": "latent_geometry",
            "dataset": config.data.datasets.datasets_list[0],
            "diffusion_checkpoint": runner.loaded_checkpoint_path,
            "step": runner.step,
            "quantiles": QUANTILES,
            "bank_size": bank_size,
            "n_anchors": n_anchors,
            "image_latents": image_path or None,
            "spaces": results,
        }, out_path)


if __name__ == "__main__":
    main()
