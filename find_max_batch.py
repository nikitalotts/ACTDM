"""Подбор максимального батча, который влезает в память GPU.

Зачем: батчи в конфигах перенесены из ВКР и могли остаться заниженными.
GPT2-medium -- это всего ~1.4 ГБ весов, и на карте с 32 ГБ микробатч 8
почти наверняка сильно меньше возможного. Больший микробатч при том же
эффективном батче означает меньше шагов накопления и меньше времени обучения.

Что делает: для каждого подхода собирает РОВНО те объекты, что живут в памяти
при обучении (энкодер, декодер, denoising network / GPT, состояния Adam, EMA),
и прогоняет настоящий шаг обучения -- forward, backward, шаг оптимизатора --
увеличивая батч, пока не поймает OOM.

Почему нужен настоящий шаг, а не просто загрузка модели:
  * градиенты -- это вторая копия параметров;
  * AdamW хранит еще две (m и v), причем аллоцирует их на ПЕРВОМ шаге,
    поэтому шаг оптимизатора обязателен;
  * активации forward держатся до backward и обычно и есть основной расход;
  * autocast bf16 добавляет копии активаций.

Датасет не читается: батчи синтетические, но той же формы и длины, что настоящие
(max_context_len + max_sequence_len). На память влияет только форма.

Запуск:  python find_max_batch.py [--arch gpt diffuseq ...] [--max-batch 4096]
"""

import argparse
import gc
import json
import os
import subprocess
import sys
import threading
import time

import torch


def free_all():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def gb(x):
    return x / 1024 ** 3


class Probe:
    """Общая обвязка: собрать модель один раз, потом пробовать разные батчи."""

    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.build()

    def build(self):
        raise NotImplementedError

    def step(self, batch_size):
        raise NotImplementedError

    def teardown(self):
        for name in list(vars(self)):
            if name not in ("config", "device"):
                delattr(self, name)
        free_all()


class GPTProbe(Probe):
    """GPT2-medium с нуля: модель + AdamW + autocast bf16, как в gpt2_holder."""

    def build(self):
        from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

        tok = GPT2Tokenizer.from_pretrained("gpt2-medium")
        cfg = GPT2Config.from_pretrained("gpt2-medium")
        self.model = GPT2LMHeadModel(cfg).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.optim.lr,
            weight_decay=self.config.optim.weight_decay,
            betas=(self.config.optim.beta_1, self.config.optim.beta_2),
            eps=self.config.optim.eps,
        )
        self.total_len = self.config.data.max_context_len + self.config.data.max_sequence_len
        self.vocab = cfg.vocab_size
        self.params = sum(p.numel() for p in self.model.parameters())

    def step(self, batch_size):
        ids = torch.randint(0, self.vocab, (batch_size, self.total_len), device=self.device)
        attn = torch.ones_like(ids)
        labels = ids.clone()
        labels[:, : self.config.data.max_context_len] = -100  # промпт вне лосса
        pos = (attn.cumsum(dim=-1) - 1).clamp(min=0)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = self.model(input_ids=ids, attention_mask=attn,
                             position_ids=pos, labels=labels)
            loss = out.loss / self.config.training.accum_batch_steps
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)


class DiffusionProbe(Probe):
    """Диффузия: BERT-энкодер (заморожен) + декодер + denoising network + Adam + EMA.

    Декодер при обучении не учится, но лежит на карте -- он загружается в
    __init__ раннера и участвует в расходе памяти.
    """

    def build(self):
        from copy import deepcopy
        from model import Encoder
        from model.decoder import Decoder
        from model.enc_normalizer import EncNormalizer
        from model.score_estimator import ScoreEstimatorEMB
        from utils.ema_model import ExponentialMovingAverage
        from diffusion_utils.dynamic import DynamicSDE
        from transformers import AutoConfig

        cfg = self.config
        enc_normalizer = None
        if cfg.normalize_encodings:
            if os.path.exists(cfg.data.enc_gen_mean):
                enc_normalizer = EncNormalizer(cfg.data.enc_gen_mean, cfg.data.enc_gen_std)
            else:
                print("  [!] статистик нет, энкодер без нормализации "
                      "(на память это почти не влияет)")

        gen_cfg = AutoConfig.from_pretrained(cfg.model.encoder_link)
        self.encoder = Encoder(
            cfg.model.encoder_link, enc_normalizer=enc_normalizer,
            is_change_sp_tokens=True, emb=cfg.emb,
        ).eval().to(self.device)

        self.decoder = Decoder(decoder_config=cfg.decoder,
                               diffusion_config=cfg.se_config).eval().to(self.device)

        se_config = deepcopy(cfg.se_config)
        se_config.use_self_cond = cfg.use_self_cond
        self.score_estimator = ScoreEstimatorEMB(config=se_config).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.score_estimator.parameters(),
            lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay,
            betas=(cfg.optim.beta_1, cfg.optim.beta_2), eps=cfg.optim.eps,
        )
        self.ema = ExponentialMovingAverage(self.score_estimator.parameters(),
                                            cfg.model.ema_rate)
        self.dynamic = DynamicSDE(config=cfg)
        self.grad_scaler = torch.cuda.amp.GradScaler()
        self.vocab = se_config.vocab_size
        self.params = sum(p.numel() for p in self.score_estimator.parameters())

    def step(self, batch_size):
        cfg = self.config
        dev = self.device
        trg_ids = torch.randint(0, self.vocab, (batch_size, cfg.data.max_sequence_len), device=dev)
        trg_mask = torch.ones_like(trg_ids)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16), torch.no_grad():
            clean_x = self.encoder(input_ids=trg_ids, attention_mask=trg_mask)
            if cfg.is_conditional:
                src_ids = torch.randint(0, self.vocab,
                                        (batch_size, cfg.data.max_context_len), device=dev)
                src_mask = torch.ones_like(src_ids)
                cond_x = self.encoder(input_ids=src_ids, attention_mask=src_mask)
            else:
                cond_x, src_mask = None, None

        t = torch.rand(batch_size, device=dev) * (1 - 1e-5) + 1e-5
        x_t = self.dynamic.marginal(clean_x, t)["x_t"]

        # diffuseq склеивает промпт с зашумленным продолжением в одну
        # последовательность -- вдвое длиннее, и это заметно по памяти
        if cfg.use_latent_replacement and cond_x is not None:
            z_t = torch.cat([cond_x, x_t], dim=1)
            mask = torch.cat([src_mask, trg_mask], dim=1)
            se_cond, se_cond_mask, src_len = None, None, cond_x.shape[1]
        else:
            z_t, mask = x_t, None
            se_cond, se_cond_mask, src_len = cond_x, src_mask, 0

        x_0_self_cond = torch.zeros_like(z_t)
        # self-conditioning: половина шагов делает лишний forward без градиента,
        # его пик тоже должен попасть в замер
        if cfg.use_self_cond:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16), torch.no_grad():
                x_0_self_cond = self.score_estimator(
                    x_t=z_t, time_t=t, cond=se_cond,
                    attention_mask=mask, cond_mask=se_cond_mask,
                    x_0_self_cond=x_0_self_cond,
                ).detach()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            z_0 = self.score_estimator(
                x_t=z_t, time_t=t, cond=se_cond,
                attention_mask=mask, cond_mask=se_cond_mask,
                x_0_self_cond=x_0_self_cond,
            )
            loss = torch.mean((z_0[:, src_len:, :] - clean_x) ** 2)

        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.ema.update(self.score_estimator.parameters())


def probe_for(arch, config, device):
    return GPTProbe(config, device) if arch == "gpt" else DiffusionProbe(config, device)


class UtilSampler:
    """Фоновый опрос загрузки GPU во время замера.

    Пиковая память говорит, влезает ли батч, но не говорит, занята ли карта
    делом. Мониторинг кластера показывал 20% загрузки GPU при 22% занятой
    памяти -- то есть карты простаивали. Здесь тот же показатель меряется
    сразу для каждого батча, чтобы видеть, с какого размера GPU наконец
    загружается полностью.
    """

    def __init__(self, device_index=0, period=0.05):
        self.device_index = device_index
        self.period = period
        self.samples = []
        self._stop = threading.Event()
        self._thread = None
        self._read = self._pick_reader()

    def _pick_reader(self):
        try:
            import pynvml  # noqa: F401
            torch.cuda.utilization(self.device_index)

            def read():
                return torch.cuda.utilization(self.device_index)
            return read
        except Exception:
            pass

        # запасной путь: на кластере nvidia-smi есть всегда, даже когда
        # питоновской обертки к NVML в окружении нет
        def read_smi():
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu",
                 "--format=csv,noheader,nounits", "-i", str(self.device_index)],
                text=True, stderr=subprocess.DEVNULL, timeout=5)
            return int(out.strip().splitlines()[0])
        try:
            read_smi()
            return read_smi
        except Exception:
            return None

    def _loop(self):
        while not self._stop.is_set():
            try:
                self.samples.append(self._read())
            except Exception:
                break
            self._stop.wait(self.period)

    def __enter__(self):
        if self._read is not None:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    @property
    def stats(self):
        if not self.samples:
            return None, None
        return sum(self.samples) / len(self.samples), max(self.samples)


def measure(probe, bs, total_mem, n_steps=6, warmup=2):
    """Полный замер одного размера батча. None -- значит не влез."""
    free_all()
    try:
        for _ in range(warmup):      # первый шаг аллоцирует состояния Adam
            probe.step(bs)
        torch.cuda.synchronize()
        static = torch.cuda.memory_allocated()   # веса + градиенты + Adam
        torch.cuda.reset_peak_memory_stats()

        with UtilSampler() as sampler:
            t0 = time.perf_counter()
            for _ in range(n_steps):
                probe.step(bs)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

        peak = torch.cuda.max_memory_allocated()
        reserved = torch.cuda.max_memory_reserved()
        if peak > total_mem:
            raise SystemExit(
                f"\nОСТАНОВЛЕНО: пик {gb(peak):.2f} ГБ превысил объем карты "
                f"{gb(total_mem):.2f} ГБ, но OOM не случился.\n"
                f"Значит драйвер сливает память в RAM, и подобранный батч будет "
                f"неверным.\nЗапускайте на кластере: sbatch find_max_batch.sh"
            )
        util_mean, util_max = sampler.stats
        step_time = elapsed / n_steps
        return {
            "batch": bs,
            "step_time": step_time,
            "examples_per_sec": bs / step_time,
            "peak_gb": gb(peak),
            "reserved_gb": gb(reserved),
            "static_gb": gb(static),
            "activations_gb": gb(max(peak - static, 0)),
            "mem_percent": 100 * reserved / total_mem,
            "util_mean": util_mean,
            "util_max": util_max,
        }
    except torch.cuda.OutOfMemoryError:
        free_all()
        return None
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise
        free_all()
        return None


def sweep(arch, config, device, max_batch, verbose=True):
    """Прогон по степеням двойки до OOM, затем уточнение границы делением пополам."""
    probe = probe_for(arch, config, device)
    total_mem = torch.cuda.get_device_properties(device).total_memory
    free_all()
    print(f"  обучаемых параметров: {probe.params / 1e6:.0f}M")

    rows, bs = [], 1
    while bs <= max_batch:
        r = measure(probe, bs, total_mem)
        if r is None:
            if verbose:
                print(f"    батч {bs:5d}: OOM")
            break
        rows.append(r)
        if verbose:
            u = f"{r['util_mean']:.0f}%" if r["util_mean"] is not None else "n/a"
            print(f"    батч {bs:5d}: {r['step_time'] * 1000:7.1f} мс/шаг, "
                  f"{r['examples_per_sec']:8.1f} прим/с, "
                  f"память {r['reserved_gb']:5.2f} ГБ ({r['mem_percent']:.0f}%), "
                  f"загрузка GPU {u}")
        bs *= 2

    # точная граница между последним влезшим и первым упавшим
    if rows and bs <= max_batch:
        lo, hi = rows[-1]["batch"], bs
        while hi - lo > 1:
            mid = (lo + hi) // 2
            r = measure(probe, mid, total_mem)
            if r is None:
                hi = mid
            else:
                lo = mid
                rows.append(r)
        rows.sort(key=lambda x: x["batch"])

    probe.teardown()
    return rows


def report(arch, rows, config, world_size):
    """Таблица по батчам плюс оценка времени боевого прогона."""
    if not rows:
        print("  нечего показать\n")
        return None

    from create_config import data_budget
    need = data_budget(config)["examples_seen"]

    print(f"\n  {arch}: замеры по батчам (на одну карту)")
    print("  " + "-" * 92)
    print(f"  {'батч':>6}{'мс/шаг':>10}{'прим/с':>10}{'память ГБ':>12}{'% памяти':>10}"
          f"{'активации':>11}{'GPU %':>8}{'прогон, ч':>12}")
    print("  " + "-" * 92)
    for r in rows:
        # боевой прогон идет на world_size картах, каждая тянет свою долю
        hours = need / (r["examples_per_sec"] * world_size) / 3600
        u = f"{r['util_mean']:.0f}" if r["util_mean"] is not None else "n/a"
        print(f"  {r['batch']:>6}{r['step_time'] * 1000:>10.1f}{r['examples_per_sec']:>10.1f}"
              f"{r['reserved_gb']:>12.2f}{r['mem_percent']:>10.0f}"
              f"{r['activations_gb']:>11.2f}{u:>8}{hours:>12.1f}")
    print("  " + "-" * 92)

    best = max(rows, key=lambda r: r["examples_per_sec"])
    fits = rows[-1]
    print(f"  максимум по памяти: батч {fits['batch']} "
          f"({fits['reserved_gb']:.2f} ГБ, {fits['mem_percent']:.0f}% карты)")
    print(f"  максимум по скорости: батч {best['batch']} "
          f"({best['examples_per_sec']:.0f} прим/с)")
    # Берем с запасом: в замере все тексты полной длины и одинаковые, а в
    # реальном батче длины плавают, плюс память фрагментируется на длинном
    # прогоне. Оставляем незанятыми не меньше 15% карты.
    roomy = [r for r in rows if r["mem_percent"] < 85]
    safe = max(roomy, key=lambda r: r["batch"])["batch"] if roomy else rows[0]["batch"]
    print(f"  рекомендую: {safe} на карту "
          f"(= {safe * world_size} суммарно на {world_size} картах)\n")
    return {"rows": rows, "max_fit": fits["batch"], "fastest": best["batch"],
            "recommended": safe}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", nargs="+",
                    default=["gpt", "genie", "diffuseq", "unconditional"])
    ap.add_argument("--dataset", default="wikipedia")
    ap.add_argument("--max-batch", type=int, default=4096)
    ap.add_argument("--world-size", type=int, default=4,
                    help="сколько карт в боевом прогоне (для оценки времени)")
    ap.add_argument("--json", default="max_batch_report.json",
                    help="куда сложить сырые замеры для дальнейшего анализа")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("Нужна GPU: скрипт меряет именно память и загрузку карты")

    from create_config import create_config
    from utils.util import parse as parse_train_args

    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}, памяти {gb(props.total_memory):.1f} ГБ, "
          f"SM {props.multi_processor_count}\n")

    summary = {}
    for arch in args.arch:
        print(f"=== {arch} ===")
        sys.argv = ["find_max_batch", "--dataset_name", args.dataset,
                    "--architecture_type", arch]
        config = create_config(parse_train_args())
        config.local_rank = 0
        current_per_gpu = config.training.batch_size // args.world_size

        try:
            rows = sweep(arch, config, device, args.max_batch, verbose=not args.quiet)
        except SystemExit:
            raise
        except Exception as e:
            print(f"  ОШИБКА: {type(e).__name__}: {e}\n")
            continue

        res = report(arch, rows, config, args.world_size)
        if res:
            res["current_per_gpu"] = current_per_gpu
            summary[arch] = res

    print("=" * 84)
    print(f"{'подход':<16}{'сейчас/GPU':>12}{'влезает':>10}{'быстрее всего':>16}"
          f"{'рекомендую':>13}{'ускорение':>12}")
    print("-" * 84)
    for arch, r in summary.items():
        cur = r["current_per_gpu"]
        rows = {x["batch"]: x for x in r["rows"]}
        speedup = ""
        if cur in rows and r["recommended"] in rows:
            speedup = f"{rows[r['recommended']]['examples_per_sec'] / rows[cur]['examples_per_sec']:.1f}x"
        print(f"{arch:<16}{cur:>12}{r['max_fit']:>10}{r['fastest']:>16}"
              f"{r['recommended']:>13}{speedup:>12}")
    print("=" * 84)
    print("'ускорение' -- во сколько раз вырастет пропускная способность")
    print("относительно текущего батча из конфига.")
    print("\nБатч в конфиге задается СУММАРНО по картам:")
    print(f"  training.batch_size = <рекомендую> x {args.world_size}")

    if summary:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"gpu": props.name,
                       "total_memory_gb": gb(props.total_memory),
                       "world_size": args.world_size,
                       "results": summary}, f, ensure_ascii=False, indent=2)
        print(f"\nСырые замеры: {args.json}")


if __name__ == "__main__":
    main()
