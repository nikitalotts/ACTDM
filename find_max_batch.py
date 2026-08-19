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
import os
import sys

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


def find_max(arch, config, device, max_batch, verbose=True):
    """Удвоение до OOM, затем бинарный поиск границы."""
    probe = probe_for(arch, config, device)
    free_all()
    base_mem = torch.cuda.memory_allocated()
    print(f"  параметров обучаемой модели: {probe.params / 1e6:.0f}M, "
          f"занято после загрузки: {gb(base_mem):.2f} ГБ")

    total_mem = torch.cuda.get_device_properties(0).total_memory

    def fits(bs):
        free_all()
        try:
            probe.step(bs)          # первый шаг аллоцирует состояния Adam
            probe.step(bs)          # второй идет уже с ними -- это и есть пик
            peak = torch.cuda.max_memory_allocated()
            if peak > total_mem:
                # Драйвер молча вытеснил часть данных в системную память вместо
                # того, чтобы бросить OOM (так делает WDDM на Windows). Тогда
                # предела памяти не существует и подбор бессмысленен.
                raise SystemExit(
                    f"\nОСТАНОВЛЕНО: пик {gb(peak):.2f} ГБ превысил объем карты "
                    f"{gb(total_mem):.2f} ГБ, но OOM не случился.\n"
                    f"Значит драйвер сливает память в RAM, и подобранный батч "
                    f"будет неверным.\nЗапускайте скрипт на кластере: "
                    f"sbatch find_max_batch.sh"
                )
            if verbose:
                print(f"    батч {bs:5d}: OK, пик {gb(peak):.2f} ГБ")
            return True, peak
        except torch.cuda.OutOfMemoryError:
            if verbose:
                print(f"    батч {bs:5d}: OOM")
            free_all()
            return False, None
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            if verbose:
                print(f"    батч {bs:5d}: OOM")
            free_all()
            return False, None

    ok, peak_ok = 0, 0
    bs = 1
    while bs <= max_batch:
        good, peak = fits(bs)
        if not good:
            break
        ok, peak_ok = bs, peak
        bs *= 2
    else:
        probe.teardown()
        return ok, peak_ok, True     # уперлись в потолок перебора, не в память

    lo, hi = ok, bs                  # lo влезает, hi нет
    while hi - lo > 1:
        mid = (lo + hi) // 2
        good, peak = fits(mid)
        if good:
            lo, peak_ok = mid, peak
        else:
            hi = mid
    probe.teardown()
    return lo, peak_ok, False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", nargs="+",
                    default=["gpt", "genie", "diffuseq", "unconditional"],
                    help="какие подходы проверять")
    ap.add_argument("--dataset", default="wikipedia")
    ap.add_argument("--max-batch", type=int, default=4096)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("Нужна GPU: скрипт меряет именно память карты")

    from create_config import create_config
    from utils.util import parse as parse_train_args

    device = torch.device("cuda:0")
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory
    print(f"GPU: {name}, всего памяти {gb(total):.1f} ГБ\n")

    results = {}
    for arch in args.arch:
        print(f"=== {arch} ===")
        sys.argv = ["find_max_batch", "--dataset_name", args.dataset,
                    "--architecture_type", arch]
        config = create_config(parse_train_args())
        config.local_rank = 0

        current = config.training.batch_size
        try:
            best, peak, capped = find_max(arch, config, device, args.max_batch,
                                          verbose=not args.quiet)
        except Exception as e:
            print(f"  ОШИБКА: {type(e).__name__}: {e}\n")
            continue

        results[arch] = (best, peak, current, capped)
        print(f"  --> максимум {best} на карту"
              f"{' (уперлись в --max-batch)' if capped else ''}, "
              f"пик {gb(peak):.2f} ГБ\n")

    print("=" * 72)
    print(f"{'подход':<16}{'сейчас/GPU':>12}{'максимум':>12}{'запас':>10}{'рекомендую':>14}")
    print("-" * 72)
    for arch, (best, peak, current, capped) in results.items():
        per_gpu = current // 4          # боевые прогоны идут на 4 картах
        # берем ~75% от предела: активации плавают от длины текстов, плюс
        # фрагментация памяти на длинном прогоне
        rec = int(best * 0.75)
        rec = max(1, 1 << (rec.bit_length() - 1))   # ближайшая степень двойки вниз
        print(f"{arch:<16}{per_gpu:>12}{best:>12}{best / max(per_gpu, 1):>9.1f}x{rec:>14}")
    print("=" * 72)
    print("'сейчас/GPU' -- батч из конфига, деленный на 4 карты.")
    print("'рекомендую' -- степень двойки около 75% от предела, с запасом на")
    print("фрагментацию и на разброс длин в реальных батчах.")
    print("\nЧтобы поднять батч, правьте training.batch_size в create_config.py")
    print("(это батч СУММАРНО по картам: batch_size_per_gpu = batch_size // 4).")


if __name__ == "__main__":
    main()
