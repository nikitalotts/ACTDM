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
    # ids 0..299 в bert-base-cased не являются '##'-кусками, снап границы не срабатывает
    o = _wiki_obj(with_tokenizer=True)
    src, trg = o._split_ids(list(range(300)), "prefix_lm")
    assert src == list(range(64))
    assert trg == list(range(64, 128))
    # текст короче 128 токенов делится в собственной середине
    src, trg = o._split_ids(list(range(100)), "prefix_lm")
    assert src == list(range(50)) and trg == list(range(50, 100))
    # продолжение не бывает пустым, пока есть хоть один токен
    src, trg = o._split_ids([7], "prefix_lm")
    assert trg == [7] and src == []


def test_wikipedia_split_never_starts_target_mid_word():
    """Если граница попадает в середину wordpiece-слова, продолжение начинается
    с '##'-куска: decode оставляет литеральные '##' в тексте, а повторная
    токенизация в collate дает мусорные токены '#','#' в начале каждого такого
    таргета. Граница обязана сдвигаться влево к началу слова."""
    o = _wiki_obj(with_tokenizer=True)
    # 'The' + повторы двухкускового 'unbelievable' (['un', '##believable'])
    # ставят на позицию 64 именно '##'-кусок
    text = "The " + " ".join(["unbelievable"] * 80)
    ids = o.tokenizer(text, add_special_tokens=False)["input_ids"]
    assert o.tokenizer.convert_ids_to_tokens(ids[64]).startswith("##"), (
        "предпосылка теста сломалась: на 64-й позиции должен быть '##'-кусок")

    src_ids, trg_ids = o._split_ids(ids, "prefix_lm")
    first_tok = o.tokenizer.convert_ids_to_tokens(trg_ids[0])
    assert not first_tok.startswith("##"), f"таргет начался с куска слова: {first_tok}"
    # сдвиг не дальше начала одного слова
    assert len(src_ids) >= 64 - 8, len(src_ids)
    # src оканчивается целым словом: decode/encode дает те же токены
    src_text = o.tokenizer.decode(src_ids)
    assert o.tokenizer(src_text, add_special_tokens=False)["input_ids"] == src_ids
    # в декодированном таргете нет литеральных '##'
    assert not o.tokenizer.decode(trg_ids).startswith("#")


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


def test_wikipedia_split_is_contiguous_and_capped_at_128():
    """Промпт и продолжение -- две подряд идущие части одного текста: продолжение
    начинается ровно там, где кончился промпт (ничего не потеряно и не
    продублировано на стыке), а суммарно берется не больше 128 токенов -- хвост
    длинного абзаца отбрасывается."""
    o = _wiki_obj(with_tokenizer=True)
    # реальный текст с wordpiece-разбиениями, чтобы сработал снап границы
    text = ("The unbelievable phenomenon of superconductivity was discovered in "
            "mercury by Heike Kamerlingh Onnes. ") * 12
    ids = o.tokenizer(text, add_special_tokens=False)["input_ids"]
    assert len(ids) > 128, "предпосылка теста: текст должен быть длиннее 128 токенов"

    src, trg = o._split_ids(ids, "prefix_lm")

    assert src + trg == ids[:len(src) + len(trg)], "на стыке промпт/продолжение потеря или дубль"
    assert len(src) + len(trg) <= o.max_context_len + o.max_sequence_len
    assert len(trg) == o.max_sequence_len
    # снап сдвигает границу максимум на одно слово влево
    assert o.max_context_len - 8 <= len(src) <= o.max_context_len


def test_wikipedia_prefix_lm_does_not_depend_on_rng():
    """Схема детерминированная: научник выбрал ее вместо случайной длины промпта
    именно потому, что плавающая граница размазывает позиционную статистику.
    Один и тот же текст обязан давать одну и ту же пару при любом состоянии RNG
    и на любой эпохе."""
    import random as pyrandom

    outs = []
    for seed in (0, 12345, 999):
        pyrandom.seed(seed)
        o = _wiki_obj(with_tokenizer=True)
        outs.append(o.batch_preprocessing_cond({"text": [WIKI_TEXT] * 3}))

    assert all(o == outs[0] for o in outs[1:]), "prefix_lm зависит от состояния RNG"

    # контраст: random_prefix обязан от RNG зависеть
    rnd = []
    for seed in (0, 12345):
        pyrandom.seed(seed)
        o = _wiki_obj(scheme="random_prefix", with_tokenizer=True)
        rnd.append(o.batch_preprocessing_cond({"text": [WIKI_TEXT] * 3})["text_src"])
    assert rnd[0] != rnd[1]


@pytest.mark.parametrize("split", ["validation", "test"])
def test_wikipedia_eval_splits_never_blank_the_prompt(split):
    """CFG-бланк -- прием обучения. На валидации и тесте промпт обязан доходить
    до модели целиком при любом swap_cfg_coef, иначе метрики условной генерации
    считались бы частично по безусловной."""
    o = _wiki_obj(split=split, swap=1.0, with_tokenizer=True)
    out = o.batch_preprocessing_cond({"text": [WIKI_TEXT] * 8})
    assert all(s.strip() for s in out["text_src"]), (
        f"на сплите {split} промпт занулился -- обнуление разрешено только на train")


def test_wikipedia_random_prefix_stays_within_context_budget():
    """random_prefix оставлен для сравнения схем: его граница обязана лежать
    внутри бюджета промпта, иначе промпт не влезет в max_context_len и будет
    молча обрезан в collate."""
    o = _wiki_obj(scheme="random_prefix", with_tokenizer=True)
    ids = list(range(300))
    for _ in range(50):
        src, trg = o._split_ids(ids, "random_prefix")
        assert len(src) < o.max_context_len
        assert len(trg) <= o.max_sequence_len


@pytest.mark.parametrize("ids", [[], [7], [7, 8]])
def test_wikipedia_split_survives_degenerate_texts(ids):
    """Короткие и пустые абзацы не должны ронять препроцессинг: датасет
    фильтруется по длине, но фильтр считает слова, а не токены."""
    o = _wiki_obj(with_tokenizer=True)
    src, trg = o._split_ids(list(ids), "prefix_lm")
    assert isinstance(src, list) and isinstance(trg, list)
    assert len(src) + len(trg) == len(ids)
    if ids:
        assert trg, "при непустом тексте продолжение не должно быть пустым"


def _collate_obj(at):
    """DiffusionRunner только для collate_fn -- без модели, оптимизатора и GPU."""
    from diffusion_holder import DiffusionRunner
    from transformers import AutoTokenizer

    cfg = create_config(make_args(at, dataset_name="wikipedia"))
    r = DiffusionRunner.__new__(DiffusionRunner)
    r.config = cfg
    r.tokenizer = AutoTokenizer.from_pretrained(cfg.model.encoder_link)
    return r


@pytest.mark.parametrize("at", ["genie", "diffuseq"])
def test_collate_geometry_is_64_plus_64(at):
    """Итоговая геометрия батча -- та самая, о которой договорились: 64 позиции
    промпта и 64 продолжения. У diffuseq они склеиваются в одну
    последовательность, поэтому обе части обязаны быть паддингованы до
    фиксированной длины, а не до самого длинного текста в батче."""
    r = _collate_obj(at)
    # ВСЕ тексты в батче короткие: если бы длины подгонялись под самый длинный
    # пример (padding=True), ширина вышла бы заметно меньше 64 и стык промпта с
    # таргетом у diffuseq гулял бы от батча к батчу. С длинным примером в батче
    # этот тест ничего бы не проверял -- truncation обрезала бы его до тех же 64
    batch = [{"text_src": "Short prompt.", "text_trg": "Short tail."},
             {"text_src": "Another one.", "text_trg": "And its tail."}]
    out = r.collate_fn(batch)

    if at == "diffuseq":
        # latent replacement: длины фиксированы, стык промпта и таргета
        # обязан стоять на одном и том же месте во всех примерах батча
        assert out["input_ids_src"].shape[1] == 64, (
            "промпт diffuseq паддингуется не до max_context_len: "
            f"{out['input_ids_src'].shape[1]}")
        assert out["input_ids_trg"].shape[1] == 64, (
            "таргет diffuseq паддингуется не до max_sequence_len: "
            f"{out['input_ids_trg'].shape[1]}")
        total = out["input_ids_src"].shape[1] + out["input_ids_trg"].shape[1]
        assert total == 128, f"склеенная последовательность не 128 позиций: {total}"
    else:
        # genie подает промпт через cross-attention, фиксированная ширина не
        # нужна -- батч ужимается по самому длинному примеру
        assert out["input_ids_src"].shape[1] <= 64
        assert out["input_ids_trg"].shape[1] <= 64

    # длинный текст в любом режиме обрезается по бюджету
    long_out = r.collate_fn([{"text_src": " ".join(f"w{i}" for i in range(200)),
                              "text_trg": " ".join(f"v{i}" for i in range(200))}])
    assert long_out["input_ids_src"].shape[1] == 64
    assert long_out["input_ids_trg"].shape[1] == 64


def test_diffusion_target_spends_two_slots_on_special_tokens():
    """Осознанное свойство схемы (наследство tencdm, так же было на rocstories):
    у диффузии [CLS] и [SEP] занимают 2 из 64 позиций, поэтому реального
    контента в таргете 62 токена, а у gpt (add_special_tokens=False) -- все 64.
    Пары (промпт, продолжение) при этом одни и те же во всех подходах, разница
    только в том, сколько токенов строки доходит до модели.

    Тест фиксирует это, чтобы расхождение длин генерации между подходами не
    выглядело потом багом. Если решим уравнять -- менять придется здесь.
    """
    r = _collate_obj("diffuseq")
    long_trg = " ".join(f"word{i}" for i in range(200))
    out = r.collate_fn([{"text_src": long_trg, "text_trg": long_trg}])

    ids = out["input_ids_trg"][0]
    assert len(ids) == 64
    assert r.tokenizer.convert_ids_to_tokens(int(ids[0])) == "[CLS]"
    assert r.tokenizer.convert_ids_to_tokens(int(ids[-1])) == "[SEP]"

    # у gpt те же 64 позиции продолжения заняты контентом целиком
    from gpt2_holder import GPT2Runner
    from transformers import GPT2Tokenizer
    g = GPT2Runner.__new__(GPT2Runner)
    g.config = create_config(make_args("gpt", dataset_name="wikipedia"))
    g.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    g.tokenizer.pad_token = g.tokenizer.eos_token
    g_out = g.collate_fn([{"text_src": long_trg, "text_trg": long_trg}])
    assert int((g_out["labels"][0] != -100).sum()) == 64


def test_uncond_collate_matches_conditional_target_geometry():
    """guidance берет чекпоинт безусловной диффузии, поэтому таргет обеих веток
    обязан иметь одну длину -- иначе позиционные эмбеддинги не совпадут."""
    cond = _collate_obj("diffuseq").collate_fn(
        [{"text_src": "A prompt here.", "text_trg": "A continuation here."}])
    unc = _collate_obj("unconditional").collate_fn(
        [{"text_trg": "A continuation here."}])
    assert "input_ids_src" not in unc, "в безусловном режиме промпт не подается"
    assert unc["input_ids_trg"].shape[1] <= cond["input_ids_trg"].shape[1]


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


def test_classifier_tokenizes_with_pipeline_lengths():
    """На обучении классификатор обязан видеть ту же геометрию входа, что и на
    guidance-инференсе: src шириной data.max_context_len (как в collate
    диффузии), таргет шириной data.max_sequence_len (как x_t). Прежняя
    токенизация до фиксированных 80 давала лишние зашумленные PAD-латенты
    и другие BERT-позиции в блоке таргета."""
    for scheme, fn in SCHEME_FILES.items():
        src = open(fn, encoding="utf-8").read()
        assert "max_length=config.data.max_context_len" in src, scheme
        assert "max_length=config.data.max_sequence_len" in src, scheme
        assert "max_length=config.cond_encoder.max_sequence_len" not in src, (
            f"{scheme}: токенизация вернулась к фиксированной ширине 80")


def test_classifier_name_encodes_input_geometry():
    """Геометрия входа входит в имя чекпоинта: классификатор, обученный старым
    кодом с шириной 80, не должен молча переиспользоваться новым."""
    w = create_config(make_args("guidance", dataset_name="wikipedia"))
    assert "-64x64-" in w.cond_encoder.name, w.cond_encoder.name
    r = create_config(make_args("guidance"))
    assert "-45x35-" in r.cond_encoder.name, r.cond_encoder.name


def test_classifier_name_matches_configured_epochs():
    """Число эпох в имени файла не должно расходиться с реально обучаемым."""
    c = create_config(make_args("guidance"))
    assert f"-epochs-{c.cond_encoder.epochs}-" in os.path.basename(c.cond_encoder.cond_encoder_path)


# =====================================================================
# Авторегрессионный baseline
# =====================================================================

def test_gpt_tokenizer_pads_left_for_generation():
    """estimate() отрезает сгенерированное продолжение срезом
    generated[:, src_len:] -- это корректно только при left padding промптов:
    при right padding новые токены идут после [PAD]-ов короткого промпта и в
    срез попадает паддинг. Тесты выше строят раннер через __new__, минуя
    __init__, поэтому сама настройка padding_side нигде больше не проверяется."""
    import inspect
    from gpt2_holder import GPT2Runner
    src = inspect.getsource(GPT2Runner.__init__)
    assert re.search(r"padding_side\s*=\s*[\"']left[\"']", src), \
        "GPT2Runner.__init__ обязан выставлять tokenizer.padding_side='left'"


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


@needs_cuda
def test_guidance_applies_in_validation_generation_during_training():
    """Во время обучения диффузия безусловная, но генерация на валидации обязана
    идти С guidance. calc_score смотрит на training-флаг DDP-ОБЕРТКИ (score_fn
    собран с model=ddp_score_estimator), а pred_embeddings сам переводит в eval
    только внутренний score_estimator -- поэтому estimate() обязан делать eval
    именно обертке, иначе guidance молча выключится при DDP-обучении."""
    r, cfg, H = _runner("guidance")

    class DDPStub(torch.nn.Module):
        """Эмулирует DDP: отдельный модуль со СВОИМ training-флагом."""
        def __init__(self, m):
            super().__init__()
            self.module = m

        def forward(self, *a, **kw):
            return self.module(*a, **kw)

    wrap = DDPStub(r.score_estimator).cuda()
    r.ddp_score_estimator = wrap
    r.diff_eq_solver.score_fn = lambda *a, **k: r.calc_score(*a, model=wrap, **k)

    src = torch.randn(B, L_SRC, H, device="cuda")
    mask = torch.ones(B, L_SRC, dtype=torch.long, device="cuda")
    cfg.dynamic.N = r.dynamic.N = 3

    def gen_as_estimate():
        # ровно то, что estimate() делает вокруг generate_text_conditional
        r.score_estimator.eval()
        r.ddp_score_estimator.eval()
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        out = r.pred_embeddings(batch_size=B, cond_x=src, cond_mask=mask,
                                attention_mask=None)
        r.ddp_score_estimator.train()
        r.score_estimator.train()
        return out

    # сеть в train mode -- как внутри train_epoch перед вызовом estimate()
    r.ddp_score_estimator.train()
    guided = gen_as_estimate()
    assert r.ddp_score_estimator.training, "estimate обязан вернуть train mode"

    r.use_guidance, r.guidance_scale = False, 0.0
    r.ddp_score_estimator.train()
    plain = gen_as_estimate()
    r.use_guidance, r.guidance_scale = True, cfg.guidance_scale

    assert not torch.allclose(guided, plain), (
        "на валидации во время обучения guidance не применился к генерации")

    # антитеза: если eval получил только внутренний модуль, а обертка осталась
    # в train (так делает pred_embeddings сам по себе), guidance выключен --
    # именно поэтому estimate() обязан переводить в eval и обертку
    r.ddp_score_estimator.train()
    r.score_estimator.eval()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    half_eval = r.pred_embeddings(batch_size=B, cond_x=src, cond_mask=mask,
                                  attention_mask=None)
    assert torch.allclose(half_eval, plain), (
        "ожидалось, что с train-флагом на DDP-обертке guidance не применяется; "
        "если это изменилось -- обновите тест и проверьте guard в calc_score")


# =====================================================================
# Короткий проверочный прогон (SMOKE=1) не должен задевать боевой
# =====================================================================

def _artifact_paths(at):
    """Все пути и параметры, которые smoke-прогон обязан развести с боевым."""
    c = create_config(make_args(at))
    out = {
        "prefix": c.training.checkpoints_prefix,
        "iters": c.training.training_iters,
        "num_gen_texts": c.validation.num_gen_texts,
    }
    if "decoder" in c:
        out["decoder"] = c.decoder.decoder_path
        out["decoder_steps"] = c.decoder.max_train_steps
    if "cond_encoder" in c:
        out["cond_encoder"] = c.cond_encoder.cond_encoder_path
    return out


@pytest.mark.parametrize("at", ["genie", "diffuseq", "guidance", "unconditional", "gpt"])
def test_smoke_env_absent_leaves_config_untouched(at, monkeypatch):
    """Без SMOKE=1 в окружении конфиг обязан быть в точности боевым: режим
    включается только переменной окружения, никаких следов по умолчанию."""
    monkeypatch.delenv("SMOKE", raising=False)
    real = _artifact_paths(at)

    assert real["iters"] >= 50_000, "боевой прогон не должен быть коротким"
    assert real["num_gen_texts"] == 5000
    for key, value in real.items():
        assert "smoke" not in str(value), f"{key} несет след smoke-режима: {value}"
    if "decoder_steps" in real:
        assert real["decoder_steps"] is None, "боевой декодер учится полную эпоху"

    # значение, отличное от "1", тоже не включает режим (fail-safe)
    monkeypatch.setenv("SMOKE", "0")
    assert _artifact_paths(at) == real
    monkeypatch.setenv("SMOKE", "true")
    assert _artifact_paths(at) == real


@pytest.mark.parametrize("at", ["genie", "diffuseq", "guidance", "unconditional", "gpt"])
def test_smoke_run_never_shares_artifact_with_real_run(at, monkeypatch):
    """SMOKE=1 обязан развести ВСЕ артефакты с боевыми. Пересечение хотя бы по
    одному файлу означает, что короткий прогон затрет боевые веса, а боевой
    запуск потом молча продолжит обучение с недоученного чекпоинта."""
    monkeypatch.delenv("SMOKE", raising=False)
    real = _artifact_paths(at)
    monkeypatch.setenv("SMOKE", "1")
    smoke = _artifact_paths(at)

    for key in ("prefix", "decoder", "cond_encoder"):
        if key in real:
            assert smoke[key] != real[key], f"smoke и боевой прогон делят {key}: {real[key]}"
            assert smoke[key].endswith("-smoke") or "-smoke." in smoke[key]

    assert smoke["iters"] < real["iters"]
    assert smoke["num_gen_texts"] < real["num_gen_texts"]
    if "decoder_steps" in smoke:
        assert smoke["decoder_steps"] == 200


@pytest.mark.parametrize("at", ["diffuseq", "gpt"])
def test_smoke_shrinks_warmup_below_training_length(at, monkeypatch):
    """Прогрев длиннее самого прогона оставил бы lr около нуля, а eval_freq
    больше числа шагов -- ни одной генерации за прогон. Проверяем, что все три
    величины урезаны согласованно, иначе smoke не проверяет то, ради чего он."""
    monkeypatch.setenv("SMOKE", "1")
    c = create_config(make_args(at))
    assert c.optim.linear_warmup < c.training.training_iters
    assert c.training.eval_freq <= c.training.training_iters
    assert c.training.checkpoint_freq <= c.training.training_iters


def test_real_pipeline_scripts_disable_smoke():
    """run_wikipedia.sh гасит SMOKE явно: sbatch наследует окружение целиком,
    и переменная, оставшаяся в шелле, иначе урезала бы боевое обучение."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "run_wikipedia.sh"), encoding="utf-8") as f:
        assert re.search(r"^export SMOKE=0", f.read(), re.M), \
            "run_wikipedia.sh обязан явно выставлять SMOKE=0"
    with open(os.path.join(root, "smoke_test.sh"), encoding="utf-8") as f:
        assert re.search(r"^export SMOKE=1", f.read(), re.M)


if __name__ == "__main__":
    sys.exit(pytest.main([os.path.abspath(__file__), "-v", "--tb=short", "-q"]))
