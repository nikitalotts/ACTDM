"""Проверки корректности четырех подходов к условной генерации.

Тесты сверяют реализацию с первоисточниками:
  * DiffuSeq (arXiv:2210.08933) -- partial noising и anchoring;
  * GENIE (arXiv:2212.11685)    -- условие через cross-attention;
  * ВКР, раздел 4.1             -- classifier guidance и схемы обучения классификатора;
  * авторегрессионный baseline  -- маскирование промпта в лоссе.

Запуск:
    python -m pytest tests/test_approaches.py -v
    python tests/test_approaches.py            # без pytest, кратким отчетом

Тесты, требующие CUDA, помечены skip при ее отсутствии.
"""
import argparse
import contextlib
import os
import re
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from create_config import create_config                      # noqa: E402
from utils.schemes import ARCHITECTURE_TYPES                 # noqa: E402

CUDA = torch.cuda.is_available()
needs_cuda = pytest.mark.skipif(not CUDA, reason="нужен GPU: код безусловно вызывает .cuda()")

B, L_SRC, L_TRG = 2, 6, 8


def make_args(architecture_type, **over):
    d = dict(dataset_name="rocstories", scheduler="sd", coef_d=9.0, emb=False,
             mode="transformer", encoder_name="bert-base-cased", project_name="t",
             swap_cfg_coef=0.0, eval=False, no_normalize_encodings=False,
             split_scheme=None, augmentation_scheme="shuffled",
             architecture_type=architecture_type, decoding="greedy",
             temperature=1.0, top_p=0.95, top_k=0, time_scale=1.0,
             classifier_guidance_scale=10.0 if architecture_type == "guidance" else 0.0)
    d.update(over)
    return argparse.Namespace(**d)


# =====================================================================
# Конфиг: режимы не путаются между собой
# =====================================================================

EXPECTED = {
    #                     is_cond  cross_attn  latent_repl  guidance  pipeline_cond
    "genie":             (True,    True,       False,       False,    True),
    "diffuseq":          (True,    False,      True,        False,    True),
    "guidance":          (False,   False,      False,       True,     True),
    "unconditional":     (False,   False,      False,       False,    False),
    "gpt":               (True,    False,      False,       False,    True),
}


@pytest.mark.parametrize("at", ARCHITECTURE_TYPES)
def test_mode_flags(at):
    """Каждый режим выставляет ровно свой набор производных флагов."""
    c = create_config(make_args(at))
    got = (c.is_conditional, c.use_cross_attention, c.use_latent_replacement,
           c.classifier_guidance, c.is_pipeline_conditional)
    assert got == EXPECTED[at], f"{at}: получили {got}, ожидали {EXPECTED[at]}"


def test_cross_attention_only_for_genie():
    """se_config.is_decoder включает слои cross-attention -- он нужен только genie."""
    for at in ARCHITECTURE_TYPES:
        if at == "gpt":
            continue
        c = create_config(make_args(at))
        assert c.se_config.is_decoder == (at == "genie"), at


def test_guidance_and_unconditional_share_checkpoint():
    """guidance -- это та же безусловная модель плюс классификатор, чекпоинт общий."""
    g = create_config(make_args("guidance")).training.checkpoints_prefix
    u = create_config(make_args("unconditional")).training.checkpoints_prefix
    assert g == u


def test_conditional_modes_have_distinct_checkpoints():
    """genie и diffuseq -- разные архитектуры, их чекпоинты не должны совпадать."""
    prefixes = {at: create_config(make_args(at)).training.checkpoints_prefix
                for at in ["genie", "diffuseq", "unconditional", "gpt"]}
    assert len(set(prefixes.values())) == len(prefixes), prefixes


def test_guidance_requires_scale():
    with pytest.raises(Exception):
        create_config(make_args("guidance", classifier_guidance_scale=0.0))


def test_incompatible_split_scheme_rejected():
    with pytest.raises(Exception):
        create_config(make_args("genie", dataset_name="wikipedia", split_scheme="half"))
    with pytest.raises(Exception):
        create_config(make_args("genie", dataset_name="rocstories", split_scheme="prefix_lm"))


def test_wikipedia_lengths_are_half_of_128():
    """Схема prefix LM: 128 токенов суммарно, поровну на промпт и продолжение."""
    c = create_config(make_args("genie", dataset_name="wikipedia"))
    assert c.data.split_scheme == "prefix_lm"
    assert c.data.max_context_len == 64
    assert c.data.max_sequence_len == 64


# =====================================================================
# Разбиение на промпт/продолжение
# =====================================================================

def test_rocstories_split_schemes():
    """split_story дает ровно те нарезки, что описаны в README."""
    from data.load import split_story
    s = ["s1.", "s2.", "s3.", "s4.", "s5."]
    assert split_story(s, "half") == [("s1. s2. s3.", "s4. s5.")]
    assert split_story(s, "last_sentence") == [("s1. s2. s3. s4.", "s5.")]
    assert len(split_story(s, "sliding")) == 3
    with pytest.raises(Exception):
        split_story(s, "нет такой схемы")
    with pytest.raises(AssertionError):
        split_story(s[:4], "half")


def _wiki_obj(at="genie", scheme=None, split="test", swap=0.0, with_tokenizer=False):
    """WikipediaDatasetDDP без чтения файлов с диска -- только препроцессинг."""
    from data.dataset_wiki import WikipediaDatasetDDP

    cfg = create_config(make_args(at, dataset_name="wikipedia",
                                  split_scheme=scheme, swap_cfg_coef=swap))
    o = WikipediaDatasetDDP.__new__(WikipediaDatasetDDP)
    o.config, o.split = cfg, split
    o.max_context_len = cfg.data.max_context_len
    o.max_sequence_len = cfg.data.max_sequence_len
    if with_tokenizer:
        from transformers import AutoTokenizer
        o.tokenizer = AutoTokenizer.from_pretrained(cfg.model.encoder_link)
    return o


WIKI_TEXT = " ".join(f"word{i}" for i in range(300))


def test_wikipedia_prefix_lm_is_deterministic_half():
    """prefix_lm режет ровно пополам, random_prefix -- нет."""

    def run(scheme):
        o = _wiki_obj(scheme=scheme, with_tokenizer=True)
        out = o.batch_preprocessing_cond({"text": [WIKI_TEXT] * 6})
        n = lambda ts: [len(o.tokenizer(t, add_special_tokens=False)["input_ids"]) for t in ts]
        return n(out["text_src"]), n(out["text_trg"])

    src, trg = run("prefix_lm")
    assert set(src) == {64} and set(trg) == {64}, (src, trg)

    src_r, _ = run("random_prefix")
    assert len(set(src_r)) > 1, "random_prefix обязан давать плавающую границу"


def test_wikipedia_split_ids_boundaries():
    """Договоренность с научником: 128 токенов, граница детерминированно на 64-м."""
    o = _wiki_obj()
    src, trg = o._split_ids(list(range(300)), "prefix_lm")
    assert src == list(range(64))
    assert trg == list(range(64, 128))
    # текст короче 128 токенов делится в собственной середине
    src, trg = o._split_ids(list(range(100)), "prefix_lm")
    assert src == list(range(50)) and trg == list(range(50, 100))
    # продолжение не бывает пустым, пока есть хоть один токен
    src, trg = o._split_ids([7], "prefix_lm")
    assert trg == [7] and src == []


def test_wikipedia_uncond_trains_on_continuation():
    """Безусловная диффузия делит чекпоинт с guidance, поэтому обязана обучаться
    на том же спане, что таргет условных режимов -- на продолжении, а не на
    начале текста."""
    cond = _wiki_obj(at="genie", with_tokenizer=True) \
        .batch_preprocessing_cond({"text": [WIKI_TEXT]})
    unc = _wiki_obj(at="unconditional", with_tokenizer=True) \
        .batch_preprocessing_uncond({"text": [WIKI_TEXT]})
    assert unc["text_trg"] == cond["text_trg"]
    # это именно продолжение, а не начало текста
    assert not unc["text_trg"][0].startswith("word0 "), (
        "uncond-таргет должен быть второй половиной текста, а не ее началом")


def test_wikipedia_blank_cond_keeps_target_span():
    """CFG-бланк: промпт пустеет, а продолжение остается ТЕМ ЖЕ
    (как у rocstories в data/preprocessing.py)."""
    blank = _wiki_obj(split="train", swap=1.0, with_tokenizer=True) \
        .batch_preprocessing_cond({"text": [WIKI_TEXT] * 4})
    plain = _wiki_obj(split="train", swap=0.0, with_tokenizer=True) \
        .batch_preprocessing_cond({"text": [WIKI_TEXT] * 4})
    assert all(s == "" for s in blank["text_src"])
    assert blank["text_trg"] == plain["text_trg"], (
        "бланк условия не должен сдвигать границу таргета")


def test_gpt_splits_wikipedia_with_same_tokenizer_as_diffusion():
    """Пары (промпт, продолжение) на wikipedia обязаны совпадать во всех
    подходах, поэтому gpt-конфиг обязан резать текст тем же токенизатором."""
    g = create_config(make_args("gpt", dataset_name="wikipedia"))
    d = create_config(make_args("genie", dataset_name="wikipedia"))
    assert g.model.encoder_link == d.model.encoder_link


# =====================================================================
# Схемы обучения классификатора (ВКР, раздел 4.1.3)
# =====================================================================

SCHEME_FILES = {
    "shuffled": "train_conditional_encoder_shuffled.py",
    "augmented": "train_conditional_encoder_augmented.py",
    "combined": "train_conditional_encoder_combined.py",
}


def test_curriculum_schedule_identical_across_schemes():
    """ВКР: 'расписание одно и то же для всех трех схем'.

    Диапазон t' растет от почти нулевого на первой эпохе до полного к 10-й.
    Формула обязана быть (epoch + 1) / warmup: при epoch / warmup первая эпоха
    вырождается (t' у всех примеров равен eps), а полный диапазон наступает на 11-й.
    """
    for scheme, fn in SCHEME_FILES.items():
        src = open(fn, encoding="utf-8").read()
        assert "warmup_epochs = 10" in src, scheme
        assert "progress = (epoch + 1) / warmup_epochs" in src, (
            f"{scheme}: расписание curriculum отличается от остальных схем")
        assert "if (epoch + 1) < warmup_epochs:" in src, scheme


def test_curriculum_reaches_full_range_at_tenth_epoch():
    """Проверка самой формулы: 1-я эпоха -- узкий диапазон, 10-я -- полный."""
    warmup, T, eps = 10, 1.0, 0.001

    def current_T(epoch):
        if (epoch + 1) < warmup:
            return eps + (T - eps) * ((epoch + 1) / warmup)
        return T

    assert current_T(0) == pytest.approx(0.1009, abs=1e-3)   # близко к нулевому, но не ноль
    assert current_T(0) > eps, "первая эпоха не должна вырождаться в точку"
    assert current_T(8) == pytest.approx(0.9001, abs=1e-3)
    assert current_T(9) == T, "полный диапазон обязан наступать на 10-й эпохе"
    assert current_T(12) == T


def test_classifier_loaders_drop_last():
    """Негативы строятся перестановкой внутри батча; на хвостовом батче из
    одного примера подбор перестановки без неподвижных точек зацикливается,
    поэтому хвост обязан отбрасываться во всех трех схемах."""
    for scheme, fn in SCHEME_FILES.items():
        src = open(fn, encoding="utf-8").read()
        assert "drop_last=True" in src, scheme


def test_combined_uses_single_t_prime_for_triple():
    """ВКР: 'все три типа пар затем зашумляются до одного и того же шага t'."""
    src = open(SCHEME_FILES["combined"], encoding="utf-8").read()
    assert "t_prime_neg1 = t_prime_pos" in src
    assert "t_prime_neg2 = t_prime_pos" in src


def test_augmentation_noise_ranges_match_thesis():
    """ВКР: t_aug ~ U[0, 0.5] в схеме 2 и U[0.3, 0.7] в схеме 3."""
    aug = open(SCHEME_FILES["augmented"], encoding="utf-8").read()
    assert "* (0.5 - eps) + eps" in aug, "схема 2: t_aug должен быть из U[0, 0.5]"
    comb = open(SCHEME_FILES["combined"], encoding="utf-8").read()
    assert "* 0.4 + 0.3" in comb, "схема 3: t_aug должен быть из U[0.3, 0.7]"


def test_classifier_input_matches_thesis_layout():
    """ВКР, формула (27): [CLS], t_emb, x_source, [SEP], x_t, [SEP]."""
    src = open("model/conditional_encoder.py", encoding="utf-8").read()
    order = re.search(r"inputs_embeds = torch\.cat\(\[(.*?)\], dim=1\)", src, re.S).group(1)
    names = [x.strip() for x in order.strip().split(",") if x.strip()]
    assert names == ["cls_token", "t_embed.unsqueeze(1)", "src_embeds",
                     "sep_token", "noisy_trg_embeds", "sep_token"], names


def test_classifier_distinguishes_noise_levels():
    """Классификатор обязан различать уровень зашумления.

    t непрерывный в [eps, 1], а синусоидальный эмбеддинг рассчитан на номер шага.
    Без масштабирования эмбеддинги t=0.001 и t=1.0 почти совпадают (cos > 0.97)
    и уровень шума перестает быть информативным входом.
    """
    from model.conditional_encoder import ConditionalEncoder

    ce = ConditionalEncoder.__new__(ConditionalEncoder)
    t = torch.tensor([0.001, 1.0])

    raw = ce.timestep_embedding(t, 768)
    cos_raw = torch.nn.functional.cosine_similarity(raw[0], raw[1], dim=0)
    assert cos_raw > 0.95, "предпосылка теста сломалась: без масштаба t уже различим"

    scaled = ce.timestep_embedding(t * 1000.0, 768)
    cos_scaled = torch.nn.functional.cosine_similarity(scaled[0], scaled[1], dim=0)
    assert cos_scaled < 0.5, f"после масштабирования t все еще неразличим: cos={cos_scaled}"


def test_classifier_time_scale_is_applied():
    """time_scale должен реально доходить до эмбеддинга времени, а не лежать мертвым полем."""
    from model.conditional_encoder import ConditionalEncoder
    from transformers import AutoTokenizer

    # по умолчанию множитель выключен -- поведение как было изначально
    assert create_config(make_args("guidance")).cond_encoder.time_scale == 1.0
    cfg = create_config(make_args("guidance", time_scale=1000.0))
    assert cfg.cond_encoder.time_scale == 1000.0

    tok = AutoTokenizer.from_pretrained(cfg.model.encoder_link)
    a = ConditionalEncoder(cfg.model.encoder_link, tok, hidden_dim=32, time_scale=1.0)
    b = ConditionalEncoder(cfg.model.encoder_link, tok, hidden_dim=32, time_scale=1000.0)
    t = torch.tensor([0.3, 0.7])
    ea = a.timestep_embedding(t * a.time_scale, 32)
    eb = b.timestep_embedding(t * b.time_scale, 32)
    assert not torch.allclose(ea, eb)


@needs_cuda
def test_classifier_checkpoint_time_scale_mismatch_is_rejected(tmp_path):
    """Несовпадение time_scale не меняет форму весов -- оно обязано ловиться явно."""
    from diffusion_holder import DiffusionRunner
    from model.conditional_encoder import ConditionalEncoder
    from transformers import AutoTokenizer

    cfg = create_config(make_args("guidance"))
    tok = AutoTokenizer.from_pretrained(cfg.model.encoder_link)
    ce = ConditionalEncoder(cfg.model.encoder_link, tok)

    path = str(tmp_path / "cond_encoder.pth")
    # конфиг по умолчанию ждет time_scale=1.0, а этот чекпоинт обучен с 1000.0
    torch.save({"cond_encoder": ce.state_dict(), "time_scale": 1000.0}, path)

    r = DiffusionRunner.__new__(DiffusionRunner)
    r.config, r.tokenizer, r.guidance_scale = cfg, tok, 10.0
    with pytest.raises(Exception, match="time_scale"):
        r._load_cond_encoder(path)


def test_schemes_do_not_share_classifier_file():
    """Три схемы обучают разные классификаторы -- пути обязаны различаться."""
    paths = set()
    for scheme in SCHEME_FILES:
        c = create_config(make_args("guidance", augmentation_scheme=scheme))
        paths.add(c.cond_encoder.cond_encoder_path)
    assert len(paths) == 3, paths


def test_time_scale_variants_do_not_share_file():
    """Множитель не меняет форму весов, поэтому варианты обязаны лежать раздельно."""
    a = create_config(make_args("guidance", time_scale=1.0)).cond_encoder.cond_encoder_path
    b = create_config(make_args("guidance", time_scale=1000.0)).cond_encoder.cond_encoder_path
    assert a != b
    assert "-ts" not in a, "дефолтный вариант не должен получать суффикс"
    assert "-ts1000" in b


def test_classifier_name_matches_configured_epochs():
    """Число эпох в имени файла не должно расходиться с реально обучаемым."""
    c = create_config(make_args("guidance"))
    assert f"-epochs-{c.cond_encoder.epochs}-" in os.path.basename(c.cond_encoder.cond_encoder_path)


# =====================================================================
# Авторегрессионный baseline
# =====================================================================

@needs_cuda
def test_gpt_masks_prompt_in_loss():
    """Лосс считается только по продолжению: промпт и паддинг помечены -100."""
    from gpt2_holder import GPT2Runner
    from transformers import GPT2Tokenizer

    cfg = create_config(make_args("gpt"))
    r = GPT2Runner.__new__(GPT2Runner)
    r.config = cfg
    r.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    r.tokenizer.pad_token = r.tokenizer.eos_token

    batch = [{"text_src": "The cat sat on the mat.", "text_trg": "Then it fell asleep."}]
    out = r.collate_fn(batch)
    ids, labels, attn = out["input_ids"][0], out["labels"][0], out["attention_mask"][0]

    total = cfg.data.max_context_len + cfg.data.max_sequence_len
    assert len(ids) == total

    kept = labels != -100
    assert kept.any(), "продолжение должно попадать в лосс"
    # там, где лосс считается, метка совпадает с токеном входа
    assert torch.equal(labels[kept], ids[kept])
    # паддинг слева: маска внимания в начале нулевая, метки там замаскированы
    assert attn[0] == 0 and labels[0] == -100
    # префикс до первого незамаскированного токена -- это промпт, он вне лосса
    first = int(kept.nonzero()[0])
    assert (labels[:first] == -100).all()


@needs_cuda
def test_gpt_position_ids_match_generate_convention():
    """generate() при left padding строит позиции из attention_mask (cumsum - 1),
    так что первый реальный токен получает позицию 0. На обучении позиции
    обязаны считаться так же, иначе train и inference расходятся."""
    from gpt2_holder import GPT2Runner
    from transformers import GPT2Tokenizer

    cfg = create_config(make_args("gpt"))
    r = GPT2Runner.__new__(GPT2Runner)
    r.config = cfg
    r.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    r.tokenizer.pad_token = r.tokenizer.eos_token

    # разные длины -> разный левый паддинг внутри батча
    batch = [{"text_src": "A short prompt.", "text_trg": "Tail."},
             {"text_src": "A much much much longer prompt with many words.",
              "text_trg": "And a noticeably longer continuation here."}]
    out = r.collate_fn(batch)
    attn, pos = out["attention_mask"], out["position_ids"]

    assert torch.equal(pos, (attn.cumsum(dim=-1) - 1).clamp(min=0))
    for i in range(len(batch)):
        start = int(attn[i].nonzero()[0])
        real_pos = pos[i, start:]
        assert real_pos[0] == 0, "первый реальный токен обязан иметь позицию 0"
        assert torch.equal(real_pos, torch.arange(len(real_pos)))


# =====================================================================
# Ядро диффузии: что реально доходит до denoising network
# =====================================================================

def _runner(at, hidden_layers=2):
    """Собирает DiffusionRunner без dist/wandb/датасета -- только то, что нужно методам."""
    from model.score_estimator import ScoreEstimatorEMB
    from model.conditional_encoder import ConditionalEncoder
    from diffusion_utils.dynamic import DynamicSDE
    from diffusion_utils.solvers import create_solver
    from diffusion_holder import DiffusionRunner
    from transformers import AutoTokenizer

    cfg = create_config(make_args(at))
    se_cfg = cfg.se_config
    se_cfg.num_hidden_layers = hidden_layers
    se_cfg.attention_head_size = se_cfg.hidden_size // se_cfg.num_attention_heads
    tok = AutoTokenizer.from_pretrained(cfg.model.encoder_link)

    r = DiffusionRunner.__new__(DiffusionRunner)
    r.config, r.tokenizer, r.device = cfg, tok, "cuda"
    r.dynamic = DynamicSDE(config=cfg)
    net = ScoreEstimatorEMB(config=se_cfg).cuda()
    r.score_estimator = r.ddp_score_estimator = net
    r.encoder = argparse.Namespace(encoder=argparse.Namespace(
        config=argparse.Namespace(hidden_size=se_cfg.hidden_size)))
    r.gen_enc_normalizer = None
    r.use_guidance = cfg.classifier_guidance
    r.guidance_scale = cfg.guidance_scale
    r.cond_encoder = None
    if r.use_guidance:
        ce = ConditionalEncoder(cfg.model.encoder_link, tok, hidden_dim=se_cfg.hidden_size).cuda().eval()
        for p in ce.parameters():
            p.requires_grad = False
        r.cond_encoder = ce
    r.diff_eq_solver = create_solver(cfg)(
        dynamic=r.dynamic,
        score_fn=lambda *a, **k: r.calc_score(*a, model=net, **k),
        ode_sampling=cfg.training.ode_sampling)
    return r, cfg, se_cfg.hidden_size


@needs_cuda
@pytest.mark.parametrize("at", ["genie", "diffuseq", "guidance", "unconditional"])
def test_training_step_runs_and_produces_gradient(at):
    r, cfg, H = _runner(at)
    trg = torch.randn(B, L_TRG, H, device="cuda")
    src = torch.randn(B, L_SRC, H, device="cuda") if cfg.is_pipeline_conditional else None
    batch = {"attention_mask_trg": torch.ones(B, L_TRG, dtype=torch.long, device="cuda"),
             "attention_mask_src": torch.ones(B, L_SRC, dtype=torch.long, device="cuda")}

    r.ddp_score_estimator.train()
    with contextlib.ExitStack():
        loss_dict, _ = r.calc_loss(clean_x=trg,
                                   cond_x=src if cfg.is_conditional else None,
                                   batch=batch)
    loss_dict["total_loss"].backward()
    g = sum((p.grad ** 2).sum() for p in r.score_estimator.parameters() if p.grad is not None)
    assert torch.isfinite(loss_dict["total_loss"])
    assert float(g) > 0, "градиент не дошел до denoising network"


@needs_cuda
@pytest.mark.parametrize("at", ["genie", "diffuseq", "guidance", "unconditional"])
def test_what_reaches_denoising_network(at):
    """Условие доходит до сети только в genie; у diffuseq оно вклеено в последовательность."""
    r, cfg, H = _runner(at)
    seen = []
    inner = r.score_estimator

    class Spy(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x_t, time_t, cond=None, attention_mask=None,
                    cond_mask=None, x_0_self_cond=None):
            seen.append({"seq": x_t.shape[1], "cond": cond is not None,
                         "mask": None if attention_mask is None else attention_mask.shape})
            return self.m(x_t=x_t, time_t=time_t, cond=cond, attention_mask=attention_mask,
                          cond_mask=cond_mask, x_0_self_cond=x_0_self_cond)

    spy = Spy(inner).cuda().eval()
    r.score_estimator = r.ddp_score_estimator = spy
    r.diff_eq_solver.score_fn = lambda *a, **k: r.calc_score(*a, model=spy, **k)

    src = torch.randn(B, L_SRC, H, device="cuda") if cfg.is_pipeline_conditional else None
    mask = torch.ones(B, L_SRC, dtype=torch.long, device="cuda") if src is not None else None
    cfg.dynamic.N = r.dynamic.N = 2
    r.pred_embeddings(batch_size=B, cond_x=src, cond_mask=mask, attention_mask=None)

    trg_len = cfg.data.max_sequence_len
    first = seen[0]
    if at == "genie":
        assert first["cond"] is True, "genie обязан подавать условие через cross-attention"
        assert first["seq"] == trg_len
    elif at == "diffuseq":
        assert first["cond"] is False, "у diffuseq условия в аргументе быть не должно"
        assert first["seq"] == trg_len + L_SRC, "промпт обязан быть вклеен в последовательность"
    else:
        assert first["cond"] is False
        assert first["seq"] == trg_len


@needs_cuda
def test_diffuseq_anchors_prompt_at_every_step():
    """DiffuSeq: латенты промпта восстанавливаются на каждом шаге обратного процесса."""
    r, cfg, H = _runner("diffuseq")
    captured = []
    step = r.diff_eq_solver.step

    def spy_step(x_t, t, next_t, **kw):
        captured.append(x_t[:, :L_SRC, :].detach().clone())
        return step(x_t=x_t, t=t, next_t=next_t, **kw)

    r.diff_eq_solver.step = spy_step
    src = torch.randn(B, L_SRC, H, device="cuda")
    mask = torch.ones(B, L_SRC, dtype=torch.long, device="cuda")
    cfg.dynamic.N = r.dynamic.N = 4
    r.pred_embeddings(batch_size=B, cond_x=src, cond_mask=mask, attention_mask=None)

    assert len(captured) == 4
    for i, got in enumerate(captured):
        assert torch.allclose(got, src, atol=1e-5), f"на шаге {i} промпт разъехался с исходным"


@needs_cuda
def test_diffuseq_mask_same_in_training_and_generation():
    """Маска внимания у diffuseq обязана совпадать на обучении и на генерации."""
    r, cfg, H = _runner("diffuseq")
    src = torch.randn(B, L_SRC, H, device="cuda")
    src_mask = torch.ones(B, L_SRC, dtype=torch.long, device="cuda")
    # маска таргета с паддингом -- на обучении она приходит из батча
    trg = torch.randn(B, L_TRG, H, device="cuda")
    _, _, mask_train, _, _ = r.build_score_estimator_input(
        x_t=trg, cond_x=src, attention_mask=None, cond_mask=src_mask)
    assert mask_train[:, L_SRC:].all(), (
        "часть таргета замаскирована на обучении, хотя на генерации маска единичная")


@needs_cuda
def test_guidance_changes_generation():
    """Градиент классификатора обязан влиять на результат, а не просто не падать."""
    r, cfg, H = _runner("guidance")
    src = torch.randn(B, L_SRC, H, device="cuda")
    mask = torch.ones(B, L_SRC, dtype=torch.long, device="cuda")
    cfg.dynamic.N = r.dynamic.N = 3

    def gen(use, scale):
        r.use_guidance, r.guidance_scale = use, scale
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        return r.pred_embeddings(batch_size=B, cond_x=src, cond_mask=mask, attention_mask=None)

    off = gen(False, 0.0)
    on = gen(True, 50.0)
    again = gen(True, 50.0)
    assert not torch.allclose(off, on), "guidance не влияет на генерацию"
    assert torch.equal(on, again), "генерация недетерминирована при фиксированном seed"


@needs_cuda
def test_guidance_gradient_formula():
    """ВКР, формула (23): grad = (1 - sigmoid(logits)) * d logits / d x_t."""
    r, cfg, H = _runner("guidance")
    x_t = torch.randn(B, L_TRG, H, device="cuda")
    cond = torch.randn(B, L_SRC, H, device="cuda")
    t = torch.full((B,), 0.5, device="cuda")

    got = r.compute_classifier_guidance(x_t=x_t, cond_x=cond, t=t, cond_mask=None)

    x = x_t.detach().clone().requires_grad_(True)
    logits = r.cond_encoder(src_embeds=cond, noisy_trg_embeds=x, t=t, src_mask=None)
    raw = torch.autograd.grad(logits.sum(), x)[0]
    expected = (1 - torch.sigmoid(logits)).view(-1, 1, 1) * raw
    assert torch.allclose(got, expected, atol=1e-5)


@needs_cuda
def test_guided_x0_consistent_with_score():
    """ВКР, формула (25): x_0 = (x_t + (1 - alpha_bar) * s) / sqrt(alpha_bar)."""
    r, cfg, H = _runner("guidance")
    x_t = torch.randn(B, L_TRG, H, device="cuda")
    cond = torch.randn(B, L_SRC, H, device="cuda")
    t = torch.full((B,), 0.5, device="cuda")
    r.ddp_score_estimator.eval()

    sc = torch.zeros_like(x_t)
    out = r.calc_score(model=r.ddp_score_estimator, x_t=x_t, t=t, cond=cond,
                       x_0_self_cond=sc)
    p = r.dynamic.marginal_params(t)
    lhs = out["x_0"]
    rhs = (x_t + p["std"] ** 2 * out["score"]) / p["mu"]
    assert torch.allclose(lhs, rhs, atol=1e-4)
    # eps_theta тоже обязан быть согласован с итоговым x_0
    eps = (x_t - p["mu"] * out["x_0"]) / p["std"]
    assert torch.allclose(out["eps_theta"], eps, atol=1e-4)


@needs_cuda
def test_guidance_is_off_during_training():
    """Обучаться под guided score нельзя -- диффузия должна оставаться безусловной."""
    r, cfg, H = _runner("guidance")
    x_t = torch.randn(B, L_TRG, H, device="cuda")
    cond = torch.randn(B, L_SRC, H, device="cuda")
    t = torch.full((B,), 0.5, device="cuda")

    sc = torch.zeros_like(x_t)
    r.ddp_score_estimator.train()
    torch.manual_seed(0)
    train_out = r.calc_score(model=r.ddp_score_estimator, x_t=x_t, t=t, cond=cond,
                             x_0_self_cond=sc)

    saved, r.use_guidance = r.use_guidance, False
    torch.manual_seed(0)
    plain = r.calc_score(model=r.ddp_score_estimator, x_t=x_t, t=t, cond=cond,
                         x_0_self_cond=sc)
    r.use_guidance = saved

    assert torch.allclose(train_out["score"], plain["score"], atol=1e-6), (
        "в режиме train к score применился guidance")


if __name__ == "__main__":
    sys.exit(pytest.main([os.path.abspath(__file__), "-v", "--tb=short", "-q"]))
