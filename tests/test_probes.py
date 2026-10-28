"""Проверки диагностических прогонов (probe_*.py).

Главный тест здесь -- test_reverse_process_reproduces_pred_embeddings. Обратный
процесс в probes/common.py продублирован ради перехватов (ветвление, подмена
градиента, снимки состояний), и единственное, что защищает от его расхождения с
боевым pred_embeddings, -- побитовое сравнение. Сравнение идет с НАСТОЯЩИМ
методом DiffusionRunner, а не с копией: метод вызывается на том же объекте-заглушке.

Запуск:
    python -m pytest tests/test_probes.py -v
"""
import argparse
import inspect
import math
import os
import sys
import types
from functools import partial

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from create_config import create_config                      # noqa: E402
from diffusion_utils.dynamic import DynamicSDE               # noqa: E402
from diffusion_utils.solvers import create_solver            # noqa: E402
from probes.common import reverse_process, unigram_f1        # noqa: E402
from utils.util import set_seed                              # noqa: E402

HIDDEN = 16
B = 3


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


class StubRunner:
    """Минимальный раннер: только то, чего касаются pred_embeddings и reverse_process.

    Чекпоинты, BERT и декодер здесь не нужны -- проверяется ПОРЯДОК вычислений и
    обращений к генератору случайных чисел, а не качество модели. Поэтому
    calc_score детерминированный и случайности не потребляет: любое расхождение
    в результате означает расхождение в самом цикле.
    """

    def __init__(self, config, device="cpu"):
        from diffusion_holder import DiffusionRunner

        self.config = config
        self.device = torch.device(device)
        self.dynamic = DynamicSDE(config=config)
        self.encoder = types.SimpleNamespace(
            encoder=types.SimpleNamespace(
                config=types.SimpleNamespace(hidden_size=HIDDEN)))
        self.score_estimator = types.SimpleNamespace(eval=lambda: None)
        self.ddp_score_estimator = self.score_estimator
        self.tokenizer = types.SimpleNamespace(pad_token_id=0)
        self.build_score_estimator_input = partial(
            DiffusionRunner.build_score_estimator_input, self)
        self.pred_embeddings = partial(DiffusionRunner.pred_embeddings, self)
        self.diff_eq_solver = create_solver(config)(
            dynamic=self.dynamic,
            score_fn=partial(self.calc_score, model=self.ddp_score_estimator),
            ode_sampling=config.training.ode_sampling,
        )

    def calc_score(self, model, x_t, t, cond=None, attention_mask=None,
                   cond_mask=None, x_0_self_cond=None):
        params = self.dynamic.marginal_params(t)
        x_0 = torch.tanh(x_t) * 0.5
        eps = (x_t - params["mu"] * x_0) / params["std"]
        return {"score": -eps / params["std"], "x_0": x_0, "eps_theta": eps}


def _stub(architecture_type, n_steps=5, seq_len=8):
    config = create_config(make_args(architecture_type))
    config.data.max_sequence_len = seq_len
    config.data.max_context_len = seq_len
    config.dynamic.N = n_steps
    return StubRunner(config), config


# =====================================================================
# Страж дублирования цикла генерации
# =====================================================================

@pytest.mark.parametrize("architecture_type", ["unconditional", "genie", "diffuseq"])
def test_reverse_process_reproduces_pred_embeddings(architecture_type):
    """При выключенных перехватах reverse_process обязан дать ПОБИТОВО то же,
    что DiffusionRunner.pred_embeddings, при одном сиде.

    Порядок обращений к генератору здесь часть контракта: сначала
    prior_sampling, затем по одному randn_like на каждом шаге солвера. Лишний
    или пропущенный вызов сдвинет всю траекторию, и диагностика описывала бы
    не ту генерацию, которая стоит в таблицах статьи."""
    runner, config = _stub(architecture_type)

    cond_x = cond_mask = None
    if config.is_pipeline_conditional:
        torch.manual_seed(7)
        cond_x = torch.randn(B, config.data.max_context_len, HIDDEN)
        cond_mask = torch.ones(B, config.data.max_context_len, dtype=torch.long)

    set_seed(0)
    reference = runner.pred_embeddings(
        batch_size=B, cond_x=cond_x, cond_mask=cond_mask, attention_mask=None)

    set_seed(0)
    probe = reverse_process(
        runner, batch_size=B, cond_x=cond_x, cond_mask=cond_mask,
        attention_mask=None)

    assert reference.shape == probe.shape
    assert torch.equal(reference, probe), (
        f"{architecture_type}: reverse_process разошелся с pred_embeddings; "
        f"max |diff| = {(reference - probe).abs().max().item():.3e}")


def test_resume_does_not_touch_the_random_generator():
    """Ветвление стартует из сохраненного состояния, а не из шума, и обязано
    НЕ вызывать prior_sampling: иначе ветка потребила бы лишний случайный
    тензор и отличалась бы от базовой траектории даже при одинаковом сиде."""
    runner, config = _stub("unconditional")

    snapshots = []
    set_seed(0)
    runner.pred_embeddings(batch_size=B, cond_x=None, cond_mask=None,
                           attention_mask=None)

    set_seed(0)
    reverse_process(runner, batch_size=B, record_at=[0, 2],
                    record_out=snapshots)
    assert len(snapshots) == 2

    snap = snapshots[-1]
    # Доигрываем хвост дважды с одного сида -- результат обязан совпасть.
    set_seed(123)
    a = reverse_process(runner, batch_size=B, resume=snap)
    set_seed(123)
    b = reverse_process(runner, batch_size=B, resume=snap)
    assert torch.equal(a, b), "хвост ветвления невоспроизводим при одном сиде"

    # ...а с другого -- разойтись, иначе ветвление ничего не меряет.
    set_seed(456)
    c = reverse_process(runner, batch_size=B, resume=snap)
    assert not torch.equal(a, c), (
        "хвост не зависит от сида -- ветвление вырождено, divergence(t) "
        "окажется нулевой по построению")


def test_record_snapshot_resumes_from_the_next_step():
    """Снимок сделан ПОСЛЕ шага idx, значит продолжать надо с idx+1.

    Ошибка на единицу здесь тихая: шаг просто повторился бы, кривая
    предрешенности поехала бы на один шаг и tau_commit оказался бы смещен."""
    runner, _ = _stub("unconditional", n_steps=4)
    snaps = []
    set_seed(0)
    reverse_process(runner, batch_size=B, record_at=range(4), record_out=snaps)
    for s in snaps:
        assert s["start_idx"] == s["idx"] + 1


def test_build_only_returns_before_any_side_effect():
    """build_only обязан выйти ДО штатных веток: eval=True зовет estimate() на
    5000 текстов, eval=False готовит оптимизатор и может дозапуститься."""
    from diffusion_holder import DiffusionRunner

    assert "build_only" in inspect.signature(DiffusionRunner.__init__).parameters
    src = inspect.getsource(DiffusionRunner.__init__)
    assert "if build_only:" in src, "флаг объявлен, но нигде не проверяется"
    assert src.index("if build_only:") < src.index("if eval:"), (
        "проверка build_only стоит после ветки eval -- побочные эффекты уже "
        "случатся")


# =====================================================================
# Геометрия батча: длины обязаны совпадать с длинами генерации
# =====================================================================

def test_pad_batch_to_max_matches_generation_geometry():
    """collate_fn паддит по самому длинному тексту БАТЧА, а генерация всегда
    идет на data.max_sequence_len.

    Без выравнивания цель reconstruction guidance имела бы другую длину, чем
    x_t: либо падение на вычитании, либо -- при случайном совпадении размеров
    -- сравнение РАЗНЫХ позиций. Банк латентов из нескольких батчей разной
    длины вообще не склеивается в тензор."""
    from probes.common import pad_batch_to_max

    runner, config = _stub("genie", seq_len=8)
    config.data.max_sequence_len = 8
    config.data.max_context_len = 6

    batch = {
        "input_ids_trg": torch.full((2, 5), 7, dtype=torch.long),
        "attention_mask_trg": torch.ones(2, 5, dtype=torch.long),
        "input_ids_src": torch.full((2, 3), 9, dtype=torch.long),
        "attention_mask_src": torch.ones(2, 3, dtype=torch.long),
    }
    out = pad_batch_to_max(runner, batch)

    assert out["input_ids_trg"].shape == (2, 8)
    assert out["attention_mask_src"].shape == (2, 6)
    # хвост -- именно паддинг, и маска его закрывает
    assert torch.equal(out["input_ids_trg"][:, 5:],
                       torch.zeros(2, 3, dtype=torch.long))
    assert torch.equal(out["attention_mask_trg"][:, 5:],
                       torch.zeros(2, 3, dtype=torch.long))
    # содержательная часть не тронута
    assert torch.equal(out["input_ids_trg"][:, :5],
                       torch.full((2, 5), 7, dtype=torch.long))


def test_pad_batch_to_max_truncates_and_tolerates_missing_source():
    """Длиннее конфига -- обрезать: иначе диффузия получила бы последовательность,
    которой не бывает на генерации. Безусловный режим идет вообще без src."""
    from probes.common import pad_batch_to_max

    runner, config = _stub("unconditional", seq_len=4)
    config.data.max_sequence_len = 4

    batch = {
        "input_ids_trg": torch.arange(12).reshape(2, 6),
        "attention_mask_trg": torch.ones(2, 6, dtype=torch.long),
    }
    out = pad_batch_to_max(runner, batch)
    assert out["input_ids_trg"].shape == (2, 4)
    assert torch.equal(out["input_ids_trg"], torch.arange(12).reshape(2, 6)[:, :4])


# =====================================================================
# Статистика дырявости
# =====================================================================

def test_hole_ratio_separates_a_blob_from_clusters():
    """H обязан отличать гладкое облако от разделенных кластеров.

    Это проверка самой СТАТИСТИКИ, а не кода вокруг нее: если H не различает
    эти два синтетических случая, ее числа на реальных энкодингах ничего не
    значат, а на них держится весь тезис про дыры."""
    from probe_latent_geometry import hole_ratio

    torch.manual_seed(0)
    blob = torch.randn(4000, 8)

    centers = torch.randn(20, 8) * 12.0
    clusters = (centers.repeat_interleave(200, dim=0)
                + torch.randn(4000, 8) * 0.02)

    h_blob = hole_ratio(blob, n_anchors=128, seed=0)
    h_clust = hole_ratio(clusters, n_anchors=128, seed=0)

    b = h_blob["H"]["0.5"]
    c = h_clust["H"]["0.5"]
    assert b < 5.0, f"у гладкого облака H(0.5) = {b:.2f}, ожидался порядок единицы"
    assert c > 10.0 * b, (
        f"кластеры дали H(0.5) = {c:.2f} против {b:.2f} у облака -- "
        f"статистика не различает дыры")

    # Внутри кластера промежуток населен: на малом квантиле H обязан быть мал.
    assert h_clust["H"]["0.01"] < h_clust["H"]["0.5"], (
        "H не растет с расстоянием пары -- значит меряется не покрытие "
        "промежутка")


def test_hole_ratio_excludes_the_pair_from_the_nearest_search():
    """Концы отрезка исключены из поиска ближайшего. Без этого для близкой пары
    ответом всегда была бы она сама, и H выродился бы в константу около 0.5
    на любых данных."""
    from probe_latent_geometry import _min_dist_excluding

    bank = torch.tensor([[0.0], [1.0], [10.0]])
    mid = torch.tensor([[0.5]])                     # середина между 0 и 1
    excl = torch.tensor([[0, 1]])
    d = _min_dist_excluding(mid, bank, excl)
    assert torch.allclose(d, torch.tensor([9.5])), (
        f"ожидалось расстояние до точки 10.0 (9.5), получено {d.tolist()}")


# =====================================================================
# Reconstruction guidance
# =====================================================================

def test_recon_identity_gradient_matches_the_closed_form():
    """В режиме identity градиент обязан быть scale*(z - x_0)/(gamma^2 * mu_t).

    Это и есть приближение d x_0 / d x_t ~ (1/mu_t) I. Если множитель mu_t
    потерять, сила guidance начнет зависеть от уровня шума не так, как
    написано в статье."""
    from probe_recon_guidance import make_recon_guidance

    runner, config = _stub("unconditional")
    torch.manual_seed(0)
    x_t = torch.randn(B, config.data.max_sequence_len, HIDDEN)
    x_0 = torch.randn_like(x_t)
    target = torch.randn_like(x_t)
    t = 0.4 * torch.ones(B)

    diag = {"rel_norm": []}
    fn = make_recon_guidance(runner, scale=2.0, jacobian="identity",
                             gamma2=3.0, diag=diag, scale_mode="const")
    fn.state["target"] = target

    out = {"x_0": x_0, "score": torch.ones_like(x_t)}
    got = fn(x_t=x_t, t=t, out=out)

    mu = runner.dynamic.marginal_params(t)["mu"]
    expected = 2.0 * (target - x_0) / (3.0 * mu)
    assert torch.allclose(got, expected, atol=1e-6)
    assert diag["rel_norm"], "диагностика силы шага не заполнена"


def test_recon_scale_zero_is_exactly_unconditional():
    """При scale=0 перехват обязан вернуть None, а не нулевой тензор.

    score + 0*grad выглядит тем же самым, но заставляет обертку пересчитать
    x_0 по алгебраической формуле (x_t + std^2*score)/mu и дает расхождение на
    уровне арифметики с плавающей точкой. Базовая линия обязана быть ТОЧНО
    безусловной генерацией -- иначе ее не с чем сравнивать, а именно ради
    сравнения она и считается."""
    from probe_recon_guidance import make_recon_guidance

    runner, config = _stub("unconditional")
    torch.manual_seed(0)
    x_t = torch.randn(B, config.data.max_sequence_len, HIDDEN)
    out = {"x_0": torch.randn_like(x_t), "score": torch.randn_like(x_t)}

    fn = make_recon_guidance(runner, scale=0.0, jacobian="identity",
                             gamma2=1.0, diag={"rel_norm": []},
                             scale_mode="const")
    fn.state["target"] = torch.randn_like(x_t)
    assert fn(x_t=x_t, t=0.4 * torch.ones(B), out=out) is None


def test_recon_stage_launches_the_zero_scale_baseline():
    """Без базовой линии числа oracle не читаются: непонятно, пол это или уже
    результат. Строка guidance из основной таблицы ее не заменяет -- там другое
    число текстов и другие сиды."""
    stage = _probes_stage()
    assert "PROBE_TARGET_RATIO=0" in stage, (
        "стадия probes-recon не запускает базовую линию. Внимание: после "
        "перехода на адаптивный шаг пол задается ДОЛЕЙ 0, а не scale=0 -- "
        "PROBE_RECON_SCALE=0 при scale_mode=adaptive guidance НЕ выключает")


# =====================================================================
# Касательное пространство: измерение и метод
# =====================================================================

def _subspace_bank(n=4000, D=32, d=4, noise=1e-4, seed=0):
    """Банк точек, лежащих на известном d-мерном подпространстве R^D."""
    g = torch.Generator().manual_seed(seed)
    A = torch.linalg.qr(torch.randn(D, d, generator=g))[0].T      # [d, D]
    coef = torch.randn(n, d, generator=g)
    pts = coef @ A + noise * torch.randn(n, D, generator=g)
    return pts, A


def test_tangent_basis_recovers_a_known_subspace():
    """На точках, лежащих в d-мерном подпространстве, локальный PCA обязан
    найти именно его.

    Это проверка самой оценки касательного. Если она не восстанавливает
    известное подпространство на синтетике, то числа tan_grad на реальных
    энкодингах ничего не значат -- а на них держится и измерение, и метод."""
    from probes.manifold import LatentBank

    pts, A = _subspace_bank()
    bank = LatentBank(pts, knn=16, dim=4)
    x = pts[:64]

    inside = torch.randn(64, 4) @ A                 # вектор В подпространстве
    outside = torch.randn(64, 32)
    outside = outside - (outside @ A.T) @ A         # вектор ОРТОГОНАЛЬНО ему

    f_in = bank.tangent_fraction(inside, x).mean().item()
    f_out = bank.tangent_fraction(outside, x).mean().item()
    assert f_in > 0.99, f"вектор из подпространства дал долю {f_in:.4f}"
    assert f_out < 0.05, f"ортогональный вектор дал долю {f_out:.4f}"


def test_random_vector_hits_the_dim_over_D_baseline():
    """У случайного направления доля энергии в dim-мерном подпространстве равна
    dim/D. Это та самая база, без которой tan_grad не читается."""
    from probes.manifold import LatentBank

    pts, _ = _subspace_bank(D=32, d=4)
    bank = LatentBank(pts, knn=16, dim=4)
    assert bank.random_tangent_fraction == pytest.approx(4 / 32)

    torch.manual_seed(0)
    rnd = torch.randn(256, 32)
    f = bank.tangent_fraction(rnd, pts[:256]).mean().item()
    assert abs(f - 4 / 32) < 0.05, (
        f"эмпирическая доля случайного вектора {f:.4f} против теоретической "
        f"{4 / 32:.4f}")


def test_tangent_and_normal_parts_are_orthogonal():
    """Энергии касательной и нормальной частей обязаны складываться в целое --
    иначе режимы tangent и normal не дополняют друг друга и их сравнение
    ничего не разделяет."""
    from probes.manifold import LatentBank

    pts, _ = _subspace_bank(D=32, d=4)
    bank = LatentBank(pts, knn=16, dim=4)
    torch.manual_seed(1)
    g = torch.randn(128, 32)
    x = pts[:128]

    tan = bank.project(g, x)
    nrm = g - tan
    assert torch.allclose((tan * nrm).sum(dim=1),
                          torch.zeros(128), atol=1e-4), "части не ортогональны"
    assert torch.allclose(tan.pow(2).sum(1) + nrm.pow(2).sum(1),
                          g.pow(2).sum(1), rtol=1e-4)


def test_bank_refuses_a_degenerate_basis():
    """Соседей должно быть больше размерности касательного, иначе базис
    вырожден, проекция становится тождественной и метод молча превращается
    в raw."""
    from probes.manifold import LatentBank

    pts, _ = _subspace_bank(D=32, d=4)
    with pytest.raises(ValueError):
        LatentBank(pts, knn=8, dim=8)


def test_cached_basis_matches_recomputation():
    """basis_for считается один раз на батч и переиспользуется на 21 уровне
    шума. Результат обязан совпасть с проекцией, считающей базис заново."""
    from probes.manifold import LatentBank

    pts, _ = _subspace_bank(D=32, d=4)
    bank = LatentBank(pts, knn=16, dim=4, query_chunk=37)
    torch.manual_seed(2)
    g = torch.randn(100, 32)
    x = pts[:100]
    basis = bank.basis_for(x)
    assert torch.allclose(bank.project_with(g, basis), bank.project(g, x),
                          atol=1e-5)


def test_projected_stage_compares_modes_at_equal_step():
    """raw, tangent и normal обязаны идти при ОДНОЙ доле шага.

    Проекция уменьшает норму градиента в разы. При постоянной lambda сравнение
    режимов было бы сравнением размеров шага, а не направлений, и выигрыш
    tangent ничего бы не доказывал."""
    stage = _probes_stage()
    assert "for MODE in raw tangent normal" in stage, (
        "нет контроля normal -- без него выигрыш tangent не отличить от "
        "простого уменьшения шага")
    assert "PROBE_SCALE_MODE=adaptive" in stage, (
        "режимы сравниваются не при равной доле шага")
    assert "PROBE_LAMBDA=0" in stage, "нет пола без guidance"


def test_projected_guidance_respects_the_requested_step_fraction():
    """В режиме adaptive отданный градиент обязан давать ровно заданную долю
    ||lam*grad||/||score||: на этом держится сравнение режимов."""
    from probe_projected_guidance import make_guidance

    runner, config = _stub("unconditional")
    L, D = config.data.max_sequence_len, HIDDEN

    class FakeClassifier(torch.nn.Module):
        def forward(self, src_embeds, noisy_trg_embeds, t, src_mask=None):
            return noisy_trg_embeds.pow(2).sum(dim=(1, 2)) * 0.01

    runner.cond_encoder = FakeClassifier()
    torch.manual_seed(0)
    x_t = torch.randn(B, L, D)
    out = {"x_0": torch.randn(B, L, D), "score": torch.randn(B, L, D)}
    cond = torch.randn(B, config.data.max_context_len, D)

    diag = {"rel_norm": [], "tangent_frac": []}
    fn = make_guidance(runner, None, "raw", "adaptive", 0.25, 10.0, True, diag)
    g = fn(x_t=x_t, t=0.5 * torch.ones(B), out=out, cond=cond, cond_mask=None)

    ratio = (g.flatten(1).norm(dim=1)
             / out["score"].flatten(1).norm(dim=1))
    assert torch.allclose(ratio, torch.full((B,), 0.25), atol=1e-4), (
        f"запрошена доля 0.25, получена {ratio.tolist()}")

    # lambda=0 в режиме const -- это пол без guidance, а не нулевой тензор
    fn0 = make_guidance(runner, None, "raw", "const", 0.1, 0.0, True, diag)
    assert fn0(x_t=x_t, t=0.5 * torch.ones(B), out=out, cond=cond,
               cond_mask=None) is None


def test_summary_survives_an_empty_diagnostic():
    """Базовая линия (lambda=0) не заполняет диагностику, и прямое
    форматирование None роняло прогон УЖЕ ПОСЛЕ подсчета метрик: json не
    сохранялся, карта тратилась зря (задание 4440381)."""
    body = open("probe_projected_guidance.py", encoding="utf-8").read()
    assert "def fmt(" in body, "нет безопасного форматирования сводки"
    assert "{mean(diag['rel_norm']):.4f}" not in body, (
        "сводка снова форматирует возможный None напрямую")


def test_target_json_accepts_the_format_estimate_writes():
    """estimate() сохраняет СЛОВАРЬ {датасет: [примеры]}, а не список -- в имени
    файла это видно по len=1 (число датасетов). Двухэтапная стадия 24.10 падала
    здесь с 'unhashable type: slice' на срезе словаря, уже после загрузки
    модели и датасета."""
    import json
    import tempfile

    from probe_recon_guidance import _load_target_texts

    rows = [{"SRC": "a", "TRG": "b", "GEN": "c"}]
    with tempfile.TemporaryDirectory() as d:
        as_dict = os.path.join(d, "dict.json")
        with open(as_dict, "w", encoding="utf-8") as f:
            json.dump({"wikipedia": rows}, f)
        assert _load_target_texts(as_dict) == rows

        as_list = os.path.join(d, "list.json")
        with open(as_list, "w", encoding="utf-8") as f:
            json.dump(rows, f)
        assert _load_target_texts(as_list) == rows

        many = os.path.join(d, "many.json")
        with open(many, "w", encoding="utf-8") as f:
            json.dump({"a": rows, "b": rows}, f)
        with pytest.raises(SystemExit):
            _load_target_texts(many)


def test_sweep_searches_below_the_measured_overshoot():
    """Замер 24.10: при доле шага 0.1 все режимы оказались НИЖЕ пола, то есть
    0.1 уже за пиком. При lambda=10 из ВКР доля была 0.007-0.017 и давала
    прирост. Значит свип обязан покрывать именно этот диапазон, а не идти
    вверх от 0.1 -- иначе оптимума он не найдет."""
    stage = _probes_stage()
    assert "PROBE_RATIOS:-0.003" in stage, (
        "свип не сужен вокруг измеренного пика 0.005")
    import re
    ratios = re.search(r"PROBE_RATIOS:-([0-9. ]+)\}", stage)
    assert ratios, "диапазон свипа не задан"
    values = [float(x) for x in ratios.group(1).split()]
    assert min(values) <= 0.004 and max(values) <= 0.02, (
        f"диапазон {values} не накрывает измеренный пик 0.005")


# =====================================================================
# Ось информативности условия
# =====================================================================

def test_prefix_mask_counts_content_positions_not_raw_ones():
    """Открываем первые k СОДЕРЖАТЕЛЬНЫХ позиций, а не первые k вообще.

    Нулевая позиция -- это [CLS]. По сырым позициям prefix с k=1 не дал бы ни
    одного настоящего токена, и нижняя точка оси оказалась бы ложно нулевой."""
    from probes.degrade import prefix_mask

    #                    CLS  tok  tok  SEP  pad
    content = torch.tensor([[False, True, True, False, False],
                            [False, True, True, True, False]])
    assert torch.equal(prefix_mask(content, 0), torch.zeros_like(content))
    assert torch.equal(prefix_mask(content, 1),
                       torch.tensor([[False, True, False, False, False],
                                     [False, True, False, False, False]]))
    got = prefix_mask(content, 2)
    assert got[0].sum() == 2 and got[1].sum() == 2
    # запрошено больше, чем есть содержательных -- берем все, что есть
    assert prefix_mask(content, 99)[0].sum() == 2
    assert prefix_mask(content, 99)[1].sum() == 3


def _prefix_case(B=4, L=9, D=6, seed=0):
    """Батч с явным скелетом: [CLS] + 6 содержательных + [SEP] + паддинг."""
    torch.manual_seed(seed)
    target = torch.randn(B, L, D)
    content = torch.zeros(B, L, dtype=torch.bool)
    content[:, 1:7] = True
    return target, content


def test_prefix_display_target_is_degraded_not_the_oracle():
    """ЦЕЛЬ ДЛЯ ОТОБРАЖЕНИЯ обязана зависеть от уровня.

    Раньше prefix возвращал target как есть, и steer = decode(z) был полным
    референсом: target_recovery читался 0.998 на ВСЕХ k. Пока осью x был
    косинус, это была косметика; теперь восстановимость цели -- это и есть
    ось x, и prefix вырождался в вертикальную прямую (четыре разных качества
    при одном x). Именно на prefix держится вывод 'номинальные биты качество
    не предсказывают', так что точка без координаты x -- это вывод без опоры.
    """
    from probes.degrade import content_position_mean, degrade_target

    target, content = _prefix_case()
    fill = content_position_mean(target, content)

    def gap(k):
        z, _, _ = degrade_target("prefix", target, k, content=content,
                                 pos_mean=fill)
        return (z - target)[content].pow(2).mean().item()

    assert gap(1) > 0, "цель при k=1 совпала с оракулом -- деградации нет"
    gaps = [gap(k) for k in (1, 2, 4, 6)]
    assert gaps == sorted(gaps, reverse=True), (
        f"цель не приближается к истине с ростом k: {gaps}")
    assert gaps[-1] == pytest.approx(0.0, abs=1e-6), (
        "при k = числу содержательных позиций цель обязана стать оракулом")


def test_prefix_display_target_keeps_the_sequence_skeleton():
    """Спецтокены и паддинг не трогаем.

    Скелет одинаков на всех уровнях, поэтому восстановимость меряет
    СОДЕРЖАНИЕ, а не длину: подмена [CLS] декодировалась бы лишним словом, а
    подмена позиции [SEP] сдвигала бы обрезку в latents_to_tokens, и кривая
    поехала бы по длине."""
    from probes.degrade import degrade_target

    target, content = _prefix_case()
    for k in (0, 1, 3):
        z, _, _ = degrade_target("prefix", target, k, content=content)
        assert torch.equal(z[~content], target[~content]), (
            f"k={k}: подменены позиции вне содержательных")


def test_prefix_display_target_does_not_touch_the_gradient():
    """СТРАЖ: подмена цели для отображения обязана быть невидимой для guidance.

    make_axis_guidance домножает corr на ту же позиционную маску, поэтому
    внутри нее z совпадает с target побитово, а снаружи вклад и так обнулен.
    Если это сломается, точки оси начнут отличаться не только уровнем, и
    сравнивать их будет нельзя."""
    from probes.degrade import degrade_target

    target, content = _prefix_case()
    x_0 = torch.randn_like(target)
    for k in (1, 2, 5):
        z, mask, _ = degrade_target("prefix", target, k, content=content)
        m = mask.unsqueeze(-1)
        assert torch.equal((z - x_0) * m, (target - x_0) * m), (
            f"k={k}: градиент изменился -- цель для отображения протекла в него")


def test_prefix_fill_is_a_position_mean_that_ignores_padding():
    """Якорь закрытых позиций -- позиционное среднее по СОДЕРЖАТЕЛЬНЫМ примерам.

    Encoder кладет в паддинг zero_emb. Усреднение вместе с ним тянуло бы хвост
    последовательности к нулю тем сильнее, чем позиция дальше, то есть якорь
    нес бы информацию о ДЛИНЕ -- ровно то, что эта ось мерить не должна."""
    from probes.degrade import content_position_mean

    target = torch.zeros(4, 3, 2)
    content = torch.zeros(4, 3, dtype=torch.bool)
    # В позиции 1 настоящие токены только у двух примеров из четырех,
    # у остальных там "паддинг" с огромным значением.
    content[:2, 1] = True
    target[:2, 1] = torch.tensor([[1.0, 3.0], [3.0, 5.0]])
    target[2:, 1] = 1000.0
    content[:, 0] = True

    got = content_position_mean(target, content)
    assert got.shape == (3, 2), (
        f"ожидалось позиционное среднее [L, D], получено {tuple(got.shape)} -- "
        f"общее среднее по батчу и позициям схлопывает структуру позиций")
    assert got[1].tolist() == pytest.approx([2.0, 4.0]), (
        f"паддинг попал в среднее: {got[1].tolist()}")


def test_signal_survival_measures_alignment_in_x0_space():
    """Классификатор и оракульные деградации обязаны мериться ОДНОЙ линейкой.

    probe_information_axis перешел на cos_x0 после того, как замер показал,
    что косинус в пространстве x_t измеряет не то: при t -> 1 mu ~ 5e-9, и
    mu*true - x_t вырождается в -x_t, то есть в 'убрать шум'. Если
    probe_signal_survival останется на снятой оси, 0.009 у классификатора и
    косинусы оракульных деградаций будут числами с разных шкал, и ставить их
    в одно предложение нельзя."""
    import probe_signal_survival

    src = inspect.getsource(probe_signal_survival.main)
    assert 'trg_x - out["x_0"]' in src, (
        "нет направления к истине в пространстве x_0")
    for key in ("cos_x0_grad", "cos_x0_score", "cos_x0_random_rms"):
        assert key in src, f"не считается {key}"
    # Снятая ось остается рядом: расхождение двух линеек -- результат, а не
    # предположение, и выбрасывать старую величину нельзя.
    assert "dir_true = mu * trg_x - x_t" in src, (
        "снятая ось удалена -- расхождение с ней больше нечем показать")


def test_signed_random_cosine_cannot_be_a_baseline():
    """Почему рядом с косинусом печатается именно RMS, а не среднее.

    Косинус случайного вектора симметричен, поэтому его знаковое среднее
    сходится к нулю с ростом выборки: отношение 'во сколько раз градиент лучше
    случайного' по такому знаменателю можно сделать любым, просто взяв больше
    примеров. Так и появилось '0.009 против 0.0001, в 90 раз лучше'. Честный
    масштаб -- СКО, и оно равно 1/sqrt(dim)."""
    torch.manual_seed(0)
    dim, n = 4096, 4000
    ref = torch.randn(dim)
    ref = ref / ref.norm()
    v = torch.randn(n, dim)
    cos = (v @ ref) / v.norm(dim=1)

    signed_mean = cos.mean().abs().item()
    root_mean_square = cos.pow(2).mean().sqrt().item()
    expected = dim ** -0.5

    assert root_mean_square == pytest.approx(expected, rel=0.1), (
        f"СКО случайного косинуса {root_mean_square:.5f} "
        f"против 1/sqrt(dim) = {expected:.5f}")
    assert signed_mean < root_mean_square / 10, (
        "знаковое среднее не сходится к нулю -- тест не воспроизводит эффект")


def test_position_stats_keep_the_position_structure():
    """dilute и noisy обязаны опираться на ПОЗИЦИОННЫЕ статистики.

    Докстринг всегда обещал позиционное среднее, а код считал общее по батчу и
    позициям. При alpha=0 это делало якорь не «средним продолжением», а
    постоянной последовательностью -- и портило именно нижние уровни dilute
    (стадия пускает 1%, 2%, 5%), ради которых режим и вводился."""
    from probes.degrade import content_position_stats, degrade_target

    torch.manual_seed(0)
    target = torch.randn(8, 5, 4)
    # Позиции заведомо разные по масштабу: так общее среднее от позиционного
    # отличается, и подмена одного другим видна.
    target[:, 0] += 50.0
    content = torch.ones(8, 5, dtype=torch.bool)

    mean, std = content_position_stats(target, content)
    assert mean.shape == (5, 4) and std.shape == (5, 4)
    assert mean[0].mean() > 40, "позиционное среднее схлопнуто в общее"

    z, _, _ = degrade_target("dilute", target, 0, content=content)
    assert torch.allclose(z, mean.unsqueeze(0).expand_as(z), atol=1e-4), (
        "alpha=0 обязан давать СРЕДНЕЕ ПРОДОЛЖЕНИЕ, а не постоянный вектор")
    assert not torch.allclose(z[0, 0], z[0, 1], atol=1.0), (
        "все позиции совпали -- структура позиций потеряна")

    full, _, _ = degrade_target("dilute", target, 100, content=content)
    assert torch.allclose(full, target, atol=1e-3), "alpha=1 обязан быть оракулом"


def test_commitment_has_a_deterministic_drift_branch():
    """Ветвление по шуму не отвечает на вопрос про guidance.

    Пересев дает возмущение НУЛЕВОГО СРЕДНЕГО, которое вдоль траектории
    частично гасится; guidance -- согласованный дрейф, он копится. Поэтому
    divergence по шуму занижает управляемость, причем в сторону исходной
    гипотезы работы. Нужна вторая ветка: тот же сид (значит тот же шум
    побитово) плюс дрейф той же доли шага."""
    import probe_commitment

    src = inspect.getsource(probe_commitment)
    assert "def make_drift" in src, "нет ветки детерминированного дрейфа"
    assert "divergence_drift" in src and "tau_commit_drift" in src, (
        "дрейф не доходит до результата")
    # Ключевое: обе ветки стартуют с ОДНОГО сида, иначе шум разный и
    # расхождение нельзя приписать дрейфу.
    body = inspect.getsource(probe_commitment.main)
    assert body.count("set_seed(branch_seed)") == 2, (
        "ветки с дрейфом и без идут по разному шуму -- сравнивать нечего")
    # И дрейф не должен трогать глобальный генератор.
    assert "generator=g_cpu" in body, (
        "случайное направление берется из глобального RNG -- оно сдвинет шум "
        "одной из веток")


def test_geometry_has_a_matched_gaussian_null():
    """H(q) без нуля не читается.

    Это отношение двух медиан расстояний, и в высокой размерности сама
    концентрация меры загоняет середины случайных пар в разреженные области.
    whitened -- те же точки, только декоррелированные (проверка анизотропии),
    images -- D=4 против 768. Нужен гауссиан тех же N и D."""
    import probe_latent_geometry

    src = inspect.getsource(probe_latent_geometry.main)
    assert "matched_gaussian" in src, "нет гауссова нуля"
    assert '"gauss"' in src
    assert "encodings,gauss" in src, "нуль не считается по умолчанию"


def _topic_bank(n_topics=4, per=24, L=5, D=12, seed=0):
    """Банк, где темы РАЗЛИЧИМЫ по смыслу, но порядок слов случаен.

    Так и устроены реальные данные: тему несёт средняя семантика абзаца, а не
    конкретная последовательность. Банк, собранный иначе, не проверил бы то,
    ради чего кластеризация переехала на пулированные векторы."""
    from probes.manifold import LatentBank

    torch.manual_seed(seed)
    centres = torch.randn(n_topics, D) * 20.0
    # Тема -- ПЛОТНО ВЫБРАННАЯ КРИВАЯ (кольцо) вокруг центра, а не гауссов шар
    # и не разреженная оболочка. Это и есть геометрия, ради которой всё
    # затевалось: середина кольца пуста, то есть среднее кластера лежит вне
    # данных по построению, и удалённость от многообразия растёт монотонно,
    # когда цель едет от кольца к центру.
    #
    # Шар не годится -- там центроид попадает в самую плотную область.
    # Разреженная оболочка в высокой размерности тоже: соседи по ней дальше,
    # чем её радиус, и зазор сначала ПАДАЕТ при движении внутрь. Оба варианта
    # проверены и отвергнуты.
    R = 2.5
    idx = torch.arange(n_topics * per).float()
    phase = (idx % per) / per * 2 * math.pi
    ang = phase.unsqueeze(1) + torch.arange(L).float().unsqueeze(0) * 0.3
    seqs = centres.repeat_interleave(per, dim=0).unsqueeze(1).expand(-1, L, -1).clone()
    seqs[:, :, 0] += R * torch.cos(ang)
    seqs[:, :, 1] += R * torch.sin(ang)
    return LatentBank(seqs.reshape(-1, D).contiguous(), knn=8, dim=4,
                      sequences=seqs.contiguous()), centres


def test_sequence_clustering_survives_high_dimension():
    """ПРИЧИНА ПОЧИНКИ. k-means по развёрнутым последовательностям вырождался.

    Прогон 25.10: K = 4, 16, 64, 256 дали bert 0.3882 / 0.3943 / 0.3957 /
    0.3973, то есть рост числа тем в 64 раза не изменил ничего, а три разных
    примера получили один и тот же медоид. 1024 точки в 49152 измерениях почти
    равноудалены, и кластеры схлопываются. Кластеризация переехала на
    пулированные векторы; тест требует, чтобы темы РАЗЛИЧАЛИСЬ."""
    from probes.degrade import cluster_bundle, sequence_codebook

    bank, centres = _topic_bank()
    centers, assign, pooled = sequence_codebook(bank, 4)
    assert len(set(assign.tolist())) == 4, (
        f"кластеры схлопнулись: занято {len(set(assign.tolist()))} из 4")

    # Две цели из РАЗНЫХ тем обязаны получить разные медоиды. Именно это и
    # сломалось на боевом прогоне.
    a = bank.sequences[0:1]
    b = bank.sequences[-1:]
    _, med_a, _ = cluster_bundle(bank, a, 4)
    _, med_b, _ = cluster_bundle(bank, b, 4)
    assert not torch.allclose(med_a, med_b, atol=1e-3), (
        "цели из разных тем получили один медоид -- кластеризация вырождена")


def test_medoid_is_real_data_and_centroid_is_not():
    """Вся конструкция держится на том, что медоид лежит НА многообразии, а
    центроид -- нет. Если это перестанет быть так, цена схода с многообразия
    измеряет не то."""
    from probes.degrade import cluster_bundle

    bank, _ = _topic_bank()
    target = bank.sequences[5:9]
    centroid, medoid, _ = cluster_bundle(bank, target, 4)

    # Медоид обязан СОВПАДАТЬ с каким-то элементом банка.
    d = (bank.sequences.unsqueeze(0) - medoid.unsqueeze(1)).pow(2).flatten(2).sum(2)
    assert d.min(dim=1).values.max() < 1e-6, "медоид не является реальной точкой"
    # Центроид -- нет.
    dc = (bank.sequences.unsqueeze(0) - centroid.unsqueeze(1)).pow(2).flatten(2).sum(2)
    assert dc.min(dim=1).values.min() > 1e-3, "центроид совпал с точкой банка"


def test_candidates_are_spread_not_collapsed():
    """Почему mixture дал ноль на боевом прогоне.

    Кандидаты брались как четыре ближайших к центроиду и почти совпадали, так
    что STEER у medoid и mixture выходил дословно одинаковым: множество было
    неотличимо от точки. Теперь дальняя выборка, и набор обязан покрывать
    кластер шире, чем его ближайшая окрестность."""
    from probes.degrade import cluster_bundle

    bank, _ = _topic_bank()
    target = bank.sequences[3:5]
    _, medoid, cand = cluster_bundle(bank, target, 4, members=4)

    spread = (cand - medoid.unsqueeze(1)).pow(2).flatten(2).sum(2)   # [B, M]
    assert spread[:, 0].max() < 1e-6, "первый кандидат обязан быть медоидом"
    assert spread[:, 1:].min() > 1e-2, (
        "кандидаты схлопнулись в медоид -- множества снова нет")
    # И они должны отличаться друг от друга, а не только от медоида.
    pair = (cand[:, 1:2] - cand[:, 2:3]).pow(2).flatten(2).sum(2)
    assert pair.min() > 1e-3, "кандидаты совпали между собой"


def test_offmanifold_sweep_keeps_information_and_moves_off_data():
    """РЕШАЮЩИЙ ИНВАРИАНТ. beta меняет ТОЛЬКО удалённость цели от данных.

    Концы свипа -- медоид и центроид одного кластера, то есть обе точки, как и
    любая промежуточная, определяются ровно меткой кластера. Если beta начнёт
    менять информацию, свип перестанет отвечать на свой вопрос."""
    from probes.degrade import cluster_bundle, offmanifold_mix

    bank, _ = _topic_bank()
    target = bank.sequences[7:11]
    centroid, medoid, _ = cluster_bundle(bank, target, 4)

    assert torch.allclose(offmanifold_mix(bank, target, 4, 0.0), medoid, atol=1e-5)
    assert torch.allclose(offmanifold_mix(bank, target, 4, 1.0), centroid, atol=1e-5)

    # Удалённость от данных обязана расти монотонно по beta -- это ось x замера.
    gaps = [bank.manifold_gap(offmanifold_mix(bank, target, 4, b)).mean().item()
            for b in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert gaps == sorted(gaps), f"удалённость не растёт по beta: {gaps}"
    assert gaps[-1] > gaps[0] * 1.5, (
        f"центроид не дальше от данных, чем медоид: {gaps[0]:.3f} -> {gaps[-1]:.3f}")


def test_manifold_gap_is_one_for_real_data():
    """Шкала обязана быть безразмерной: у реальной точки данных зазор около 1.

    Иначе число нельзя сравнивать между прогонами и чекпоинтами, и оно
    повторит судьбу косинуса -- величина без базы."""
    from probes.manifold import LatentBank

    torch.manual_seed(0)
    vecs = torch.randn(600, 8)
    bank = LatentBank(vecs, knn=8, dim=3)
    real = vecs[:16].reshape(4, 4, 8)

    gap = bank.manifold_gap(real)
    assert 0.5 < gap.median().item() < 2.0, (
        f"зазор у реальных точек {gap.median().item():.3f}, ожидалось около 1")
    far = bank.manifold_gap(real + 50.0)
    assert far.min() > 10 * gap.max(), "точка в пустоте не отличается от данных"


def _filler_bank(n=120, twins=3, L=8, D=6, n_content=2, seed=1,
                 core_j=0.002, shell_j=0.05):
    """Банк, где ПЛОТНОЕ ЯДРО позиционного облака лежит там, куда попадает
    поэлементное среднее последовательностей.

    Так устроен текст: большинство позиций занимают частые токены ("the",
    запятая, точка), они лежат тесной кучей около общего среднего, а
    содержательные слова редки и разбросаны. Поэтому среднее сотен
    последовательностей попадает в САМУЮ ПЛОТНУЮ область позиционного
    облака, оставаясь при этом невозможной последовательностью.

    Отличие от _topic_bank намеренное. Там середина кластера пуста, и оба
    зазора растут согласованно; здесь они обязаны разойтись, и именно это
    расхождение наблюдалось на боевых данных."""
    from probes.manifold import LatentBank

    torch.manual_seed(seed)
    seqs = []
    for _ in range(n):
        base = torch.zeros(L, D)
        w = torch.randn(n_content, D)
        base[:n_content] = w / w.norm(dim=1, keepdim=True)
        for _ in range(twins):
            # Близнецы с новым дрожанием: без них у банка нет собственного
            # масштаба -- ближайший сосед оказывался бы в другой теме, и
            # нормировка зазора потеряла бы смысл.
            t = base.clone()
            t[:n_content] += torch.randn(n_content, D) * shell_j
            t[n_content:] += torch.randn(L - n_content, D) * core_j
            seqs.append(t)
    seqs = torch.stack(seqs)
    return LatentBank(seqs.reshape(-1, D).contiguous(), knn=8, dim=3,
                      sequences=seqs.contiguous())


def test_position_gap_and_sequence_gap_disagree_on_the_mean():
    """ПРИЧИНА, ПО КОТОРОЙ ГЛАВНОЙ ОСЬЮ СТАЛ sequence_gap.

    Свип 26.10 (offmanifold, beta ведёт цель от медоида к центроиду) дал по
    позиционному зазору 1.110 / 1.052 / 0.976 / 0.881 / 0.819 / 0.797 / 0.795,
    то есть монотонно ВНИЗ: центроид вышел ближе к данным, чем реальная
    последовательность. Ошибки в счёте там нет, мерится не то многообразие --
    усреднение сотен последовательностей сносит каждую позицию в плотное ядро
    позиционного облака. В тексте это видно прямо: STEER центроида
    декодируется в "s the the the the", то есть каждая позиция правдоподобна,
    а последовательность невозможна.

    Тест закрепляет оба факта разом: на такой геометрии позиционный зазор у
    средней точки МЕНЬШЕ, чем у реальной, а зазор по последовательностям
    кратно больше. Если sequence_gap когда-нибудь поведёт себя как
    позиционный, главная ось работы молча развернётся."""
    bank = _filler_bank()
    real = bank.sequences[:3]
    mean_point = torch.zeros_like(real)        # плотное ядро, но не текст
    m = torch.ones(real.shape[:2], dtype=torch.bool)

    pos_real = bank.manifold_gap(real).median().item()
    pos_mean = bank.manifold_gap(mean_point).median().item()
    seq_real = bank.sequence_gap(real, m).median().item()
    seq_mean = bank.sequence_gap(mean_point, m).median().item()

    assert pos_mean < 0.9 * pos_real, (
        f"позиционный зазор не упал: {pos_real:.3f} -> {pos_mean:.3f}; "
        f"геометрия теста перестала воспроизводить наблюдение 26.10")
    assert seq_mean > 3 * seq_real, (
        f"зазор по последовательностям не вырос: {seq_real:.3f} -> "
        f"{seq_mean:.3f}, то есть главная ось не отличает среднее от данных")


def test_sequence_gap_is_dimensionless_and_grows_along_the_sweep():
    """Главная ось обязана быть безразмерной и расти вдоль свипа.

    У реального элемента банка значение около 1 -- иначе число нельзя
    сравнивать между прогонами и чекпоинтами, и оно повторит судьбу косинуса.

    Ровно на beta=0 зазор равен 1 по соглашению: цель совпадает с элементом
    банка, сам он из поиска исключён (иначе ноль), и ответом служит его
    собственный ближайший сосед. Поэтому чуть сдвинутая цель может дать чуть
    МЕНЬШЕ 1 -- она ещё ближе к медоиду, чем медоид к своему соседу. Рост
    требуем с того места, где цель уже вышла за это расстояние."""
    from probes.degrade import offmanifold_mix

    bank, _ = _topic_bank()
    target = bank.sequences[7:11]
    m = torch.ones(target.shape[:2], dtype=torch.bool)

    real = bank.sequence_gap(bank.sequences[3:7], m).median().item()
    assert 0.5 < real < 2.0, (
        f"зазор у реальных последовательностей {real:.3f}, ожидалось около 1")

    gaps = [bank.sequence_gap(offmanifold_mix(bank, target, 4, b), m)
            .mean().item() for b in (0.25, 0.5, 0.75, 1.0)]
    assert gaps == sorted(gaps), f"удалённость не растёт по beta: {gaps}"
    assert gaps[-1] > 2.5 * real, (
        f"центроид не дальше от данных, чем реальная последовательность: "
        f"{real:.3f} -> {gaps[-1]:.3f}")


def test_set_temperature_interpolates_on_concentrated_distances():
    """ПОЧЕМУ МНОЖЕСТВО НЕ ПОЛУЧАЛОСЬ ДВА ПРОГОНА ПОДРЯД.

    25.10 при sigma 0.5 энтропия веса вышла 1.239 из 1.386 -- почти
    равномерно, то есть цель была средним кандидатов. 26.10 при sigma 0.05 она
    вышла 0.00003 -- жёсткий argmin, цель стала дословно медоидом: STEER
    совпал с медоидным прогоном побуквенно, а bert разошёлся на 0.0002. Между
    этими двумя значениями середины не было вовсе.

    Причина -- концентрация расстояний: в 49152 измерениях их разброс много
    меньше их величины, а показатель брался как d2/sigma^2 с sigma от
    ВЕЛИЧИНЫ. Теперь температура берётся от разброса, и sigma читается как
    обратный размах логитов в натах. Тест требует, чтобы на таких числах
    существовала настоящая середина."""
    from probes.degrade import set_weights

    # Расстояния как они выходят на боевых данных: огромная общая величина,
    # маленький разброс по кандидатам.
    d2 = 1e5 + torch.tensor([[0.0, 1.0, 2.0, 3.0]])

    def ent(s):
        w = set_weights(d2, s)
        return (-(w.clamp_min(1e-12).log() * w).sum(dim=1).item()
                / math.log(d2.shape[1]))

    assert ent(0.02) < 0.05, "малая sigma обязана давать argmin"
    assert ent(3.0) > 0.95, "большая sigma обязана давать равномерный вес"
    mid = ent(0.3)
    assert 0.3 < mid < 0.85, (
        f"при sigma 0.3 вес ни argmin, ни равномерный: энтропия {mid:.3f} от "
        f"равномерной -- значит множества снова нет")
    es = [ent(s) for s in (0.02, 0.05, 0.1, 0.3, 1.0, 3.0)]
    assert es == sorted(es), f"энтропия не растёт по sigma: {es}"
    # Абсолютная величина расстояний не должна влиять вообще -- только
    # разброс. Иначе sigma опять не перенесётся между чекпоинтами.
    assert torch.allclose(set_weights(d2, 0.3), set_weights(d2 + 1e7, 0.3),
                          atol=1e-6), (
        "вес зависит от абсолютной величины расстояний, а не от их разброса")


def test_axis_probe_reports_fidelity_and_sequence_gap():
    """Верность цели -- отдельное число, и без неё свип не читается.

    На beta 0.9 и выше падают СРАЗУ две вещи: цель перестаёт декодироваться в
    текст ("s the the the") и сэмплер перестаёт её отслеживать. По одному
    GEN vs TRG их не различить, а это два разных утверждения о guidance."""
    body = open("probe_information_axis.py", encoding="utf-8").read()
    assert "score(gen_texts, steer_texts" in body, (
        "нет верности цели: по одному GEN vs TRG не отличить «сэмплер не "
        "дошёл до цели» от «цель перестала быть текстом»")
    assert "bank.sequence_gap(" in body, (
        "главная ось геометрии не снимается в свипе")
    for key in ('"target_fidelity"', '"sequence_gap"'):
        assert key in body, f"{key} не попадает в json -- число потеряется"


def test_sigma_stage_sweeps_the_set_temperature():
    """Второй, независимый способ сойти с многообразия -- через температуру.

    probes-offmanifold двигает цель по отрезку руками; здесь она смещается
    сама, через веса кандидатов, при той же информации. Если оба свипа дадут
    одно падение по sequence_gap, закон держится на двух разных механизмах."""
    import re

    body = open("run_wikipedia.sh", encoding="utf-8").read()
    assert "probes-sigma" in body, "нет стадии свипа по температуре множества"
    m = re.search(r"PROBE_SIGMAS:-([0-9 .]+)\}", body)
    assert m, "в стадии нет сетки по sigma"
    sigmas = [float(x) for x in m.group(1).split()]
    assert min(sigmas) <= 0.05 and max(sigmas) >= 2.0, (
        f"свип обязан доходить до обоих концов -- argmin и равномерного "
        f"веса, получено {sigmas}")


def test_offmanifold_stage_sweeps_beta_at_fixed_information():
    """Свип обязан быть в оркестраторе, с обоими концами и сеткой у излома.

    Прогон 26.10 на сетке 0 10 25 50 75 90 100 дал bert 0.5059 / 0.5058 /
    0.5055 / 0.5036 / 0.4956 / 0.4266 / 0.4019: до beta=0.5 не происходит
    ничего, весь обвал лежит между 0.75 и 0.9. Половина заданий ушла на
    плоский участок, а сам излом остался неразрешённым -- по двум точкам
    нельзя сказать, где он и насколько резкий. Требуем сгущения именно там."""
    import re

    stage = _probes_stage()
    assert "probes-offmanifold" in stage, "нет стадии цены схода с многообразия"
    assert "PROBE_DEGRADE=offmanifold" in stage
    m = re.search(r"PROBE_BETAS:-([0-9 .]+)\}", stage)
    assert m, "в стадии нет сетки по beta"
    betas = [float(x) for x in m.group(1).split()]
    assert 0 in betas and 100 in betas, (
        f"свип обязан включать оба конца: beta=0 это медоид, 100 -- центроид, "
        f"получено {betas}")
    knee = [b for b in betas if 75 < b < 100]
    assert len(knee) >= 3, (
        f"излом между 0.75 и 0.9 требует не меньше трёх точек внутри, "
        f"получено {knee}")
    assert "PROBE_TOPICS_LEVEL" in stage, (
        "число тем задаётся отдельно от beta, иначе информация поедет вдоль свипа")


def test_kmeans_recovers_known_clusters():
    """Если k-means не находит очевидные кластеры на синтетике, числа режима
    quantize на реальных латентах ничего не значат."""
    from probes.degrade import kmeans

    torch.manual_seed(0)
    centers = torch.tensor([[-5.0, 0.0], [5.0, 0.0], [0.0, 5.0]])
    x = centers.repeat_interleave(200, dim=0) + torch.randn(600, 2) * 0.1
    got = kmeans(x, 3, seed=0)

    for c in centers:
        assert (got - c).norm(dim=1).min() < 0.3, (
            f"кластер {c.tolist()} не найден: {got.tolist()}")


def test_degradation_is_monotone_in_level():
    """ГЛАВНЫЙ ИНВАРИАНТ ОСИ: чем выше уровень, тем ближе деградированная цель
    к истинной.

    Если это не так, ось информативности бессмысленна: точки на ней
    расположены не по количеству информации, и кривая 'качество от косинуса'
    ничего не доказывает."""
    from probes.degrade import degrade_target
    from probes.manifold import LatentBank

    torch.manual_seed(0)
    D, L, M = 24, 6, 32
    # Банк и цель берутся из ОДНОГО распределения -- так же, как в боевом
    # прогоне, где и банк, и цель это латенты реальных продолжений. Если бы
    # цель была независима от банка, квантизация не переносила бы информацию
    # вовсе: ближайший центроид чужого облака не приближается к цели с ростом K.
    centers = torch.randn(M, D) * 4.0
    bank_vecs = (centers.repeat_interleave(40, dim=0)
                 + torch.randn(M * 40, D) * 0.3)
    bank = LatentBank(bank_vecs, knn=8, dim=4,
                      sequences=bank_vecs[:240].reshape(40, L, D))
    pick = torch.randint(M, (4 * L,))
    target = (centers[pick] + torch.randn(4 * L, D) * 0.3).reshape(4, L, D)

    def closeness(z):
        return -(z - target).pow(2).mean().item()

    for mode, levels in (("quantize", [2, 8, 32]), ("rank", [1, 4, 12, 24])):
        scores = [closeness(degrade_target(mode, target, lv, bank=bank)[0])
                  for lv in levels]
        assert scores == sorted(scores), (
            f"{mode}: близость к цели не растет с уровнем: {scores}")

    # Полная размерность и полный ранг обязаны вернуть цель без искажения
    full, _, _ = degrade_target("rank", target, D, bank=bank)
    assert torch.allclose(full, target, atol=1e-4)


def test_none_mode_is_the_untouched_oracle():
    """Верхняя точка оси обязана быть ТОЧНО оракулом: иначе вся шкала съезжает."""
    from probes.degrade import degrade_target

    torch.manual_seed(0)
    target = torch.randn(3, 5, 8)
    z, mask, _ = degrade_target("none", target, 0)
    assert z is target and mask is None


def test_seqquant_needs_sequence_level_bank():
    """Квантизация по темам работает с латентами целых продолжений. Банк
    позиций для нее не годится, и молча подставлять его нельзя."""
    from probes.degrade import degrade_target
    from probes.manifold import LatentBank

    bank = LatentBank(torch.randn(100, 8), knn=8, dim=4)
    with pytest.raises(ValueError):
        degrade_target("seqquant", torch.randn(2, 4, 8), 4, bank=bank)


def test_infoaxis_stage_runs_two_different_degradation_axes():
    """Две оси нужны именно разные по характеру: prefix дает точную информацию
    о немногих позициях, quantize -- грубую обо всех. Совпадение их кривых и
    есть доказательство, что дело в количестве информации, а не в том, как она
    распределена. С одной осью доказательства нет."""
    stage = _probes_stage()
    assert "PROBE_DEGRADE=prefix" in stage, "нет оси точной локальной информации"
    assert "PROBE_DEGRADE=quantize" in stage, "нет оси грубой глобальной информации"
    assert "PROBE_DEGRADE=seqquant" in stage, "нет точки 'веди к теме'"
    assert "PROBE_DEGRADE=none" in stage, "нет верхней точки оси (оракул)"


def test_sigmoid_factor_changes_the_gradient():
    """PROBE_SIGMOID=0 обязан убирать множитель (1 - sigma(f)), а не быть
    декоративным флагом: на нем держится проверка факта, что guidance гаснет
    там, где классификатор уверен."""
    from probe_projected_guidance import classifier_grad

    runner, config = _stub("unconditional")
    L, D = config.data.max_sequence_len, HIDDEN

    class FakeClassifier(torch.nn.Module):
        def forward(self, src_embeds, noisy_trg_embeds, t, src_mask=None):
            # большой положительный логит -> sigma(f) ~ 1 -> множитель ~ 0
            return noisy_trg_embeds.pow(2).sum(dim=(1, 2)) + 50.0

    runner.cond_encoder = FakeClassifier()
    torch.manual_seed(0)
    x_t = torch.randn(B, L, D)
    cond = torch.randn(B, config.data.max_context_len, D)
    t = 0.5 * torch.ones(B)

    with_sig = classifier_grad(runner, x_t, cond, t, None, sigmoid_factor=True)
    without = classifier_grad(runner, x_t, cond, t, None, sigmoid_factor=False)
    assert without.norm() > 0
    assert with_sig.norm() < 1e-6 * without.norm(), (
        "множитель (1-sigma) не гасит градиент при уверенном классификаторе -- "
        "значит флаг ничего не меняет")


def test_recon_guidance_without_target_is_a_noop():
    """Пока цель не выставлена, перехват обязан вернуть None и оставить score
    безусловным -- иначе первый батч считался бы с мусорной целью."""
    from probe_recon_guidance import make_recon_guidance

    runner, config = _stub("unconditional")
    fn = make_recon_guidance(runner, scale=1.0, jacobian="identity",
                             gamma2=1.0, diag={"rel_norm": []},
                             scale_mode="const")
    x_t = torch.zeros(B, config.data.max_sequence_len, HIDDEN)
    out = {"x_0": torch.zeros_like(x_t), "score": torch.zeros_like(x_t)}
    assert fn(x_t=x_t, t=0.5 * torch.ones(B), out=out) is None


def test_guidance_fn_none_keeps_the_production_score_fn():
    """Без перехвата обертка обязана быть ТЕМ ЖЕ calc_score, а не копией:
    иначе диагностика мерила бы другую функцию, чем боевая генерация."""
    from probes.common import _wrap_score_fn

    runner, _ = _stub("unconditional")
    fn = _wrap_score_fn(runner, None)
    # сравнение по == , а не is: обращение к связанному методу каждый раз
    # создает новый объект-обертку, при этом равный по (функция, объект)
    assert fn.func == runner.calc_score
    assert fn.keywords["model"] is runner.ddp_score_estimator


# =====================================================================
# Мера совпадения текстов
# =====================================================================

def test_unigram_f1_properties():
    assert unigram_f1([1, 2, 3], [1, 2, 3]) == 1.0
    assert unigram_f1([], []) == 1.0
    assert unigram_f1([1, 2], []) == 0.0
    assert unigram_f1([1, 2, 3], [4, 5, 6]) == 0.0
    # кратность учитывается: повтор не делает тексты одинаковыми
    assert unigram_f1([1, 1, 1], [1]) == pytest.approx(0.5)
    assert 0.0 < unigram_f1([1, 2, 3, 4], [1, 2, 9, 9]) < 1.0


def test_auc_handles_ties_and_the_degenerate_classifier():
    """AUC считается через ранги со средним рангом для связок.

    Связки -- не экзотика: при большом шуме классификатор насыщается и выдает
    почти одинаковые логиты на всем батче. Без усреднения рангов вырожденный
    классификатор показал бы AUC 0 или 1 вместо честных 0.5, и кривая
    accuracy(t) соврала бы ровно там, где делается главный вывод."""
    from probe_signal_survival import _auc

    assert _auc([3.0, 2.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert _auc([1.0, 0.0], [3.0, 2.0]) == pytest.approx(0.0)
    # полностью вырожденный выход -- ровно 0.5, а не крайнее значение
    assert _auc([1.0, 1.0, 1.0], [1.0, 1.0, 1.0]) == pytest.approx(0.5)
    assert 0.0 < _auc([2.0, 0.0], [1.0, 1.0]) < 1.0


def test_t_grid_covers_the_full_noise_range():
    """Сетка обязана доходить до t=1: именно там, по гипотезе работы, сигнал
    классификатора уже мертв, и обрезать этот конец значило бы не увидеть
    главного."""
    from probes.common import EPS_T, t_grid

    g = t_grid(21)
    assert len(g) == 21
    assert g[0] == pytest.approx(EPS_T)
    assert g[-1] == pytest.approx(1.0)
    assert all(a < b for a, b in zip(g, g[1:]))


# =====================================================================
# Стадия оркестратора
# =====================================================================

def _probes_stage():
    """Тело ветки стадий probes. Метка ветки ищется по началу, а не
    сравнивается целиком: список подстадий растет, и тесты не должны
    падать от добавления новой."""
    body = open("run_wikipedia.sh", encoding="utf-8").read()
    branch = body.index("\n    probes|") + 1
    return body[branch:body.index(";;", branch)]


def test_stages_read_the_stashed_checkpoint_step():
    """Шаг сравнения берется из _CKPT_STEP, а не из CHECKPOINT_STEP.

    Шапка run_wikipedia.sh делает unset CHECKPOINT_STEP и прячет значение в
    _CKPT_STEP -- иначе обучающие стадии дозапускались бы с указанного шага.
    Стадия, которая читает голый CHECKPOINT_STEP, всегда видит ПУСТО: ровно на
    этом стадия probes падала с требованием задать переменную, которая была
    задана. Проверяем все стадии разом, чтобы следующая не наступила туда же."""
    body = open("run_wikipedia.sh", encoding="utf-8").read()
    exports = [ln for ln in body.splitlines()
               if ln.strip().startswith("export CHECKPOINT_STEP")]
    assert exports, "ни одна стадия не пинит шаг сравнения"
    for ln in exports:
        assert "_CKPT_STEP" in ln, (
            f"стадия берет шаг не из _CKPT_STEP, а он к этому моменту стерт: "
            f"{ln.strip()}")
    assert "export CHECKPOINT_STEP" in _probes_stage(), (
        "стадия probes не пинит шаг -- restore_parameters возьмет "
        "max(нумерованных), а он разный у разных подходов")


def test_probes_signal_substage_can_pick_one_scheme():
    """Первый заход считает одну схему, а не три: PROBE_SCHEMES=augmented.

    Нужна именно подстадия, а не ручной sbatch -- тот уходит без -A и падает
    с Invalid account, потому что проект Slurm выставляет оркестратор."""
    stage = _probes_stage()
    assert "PROBE_SCHEMES:-shuffled augmented combined" in stage, (
        "нельзя выбрать подмножество схем классификатора")
    assert "probes-signal" in stage, "нет подстадии только для accuracy(t)"


def test_probes_window_runs_all_three_classifier_schemes():
    """Совпадение кривых accuracy(t) у трех РАЗНЫХ классификаторов -- это и есть
    довод, что дело не в конкретном обучении. С одной схемой довода нет."""
    stage = _probes_stage()
    for aug in ("shuffled", "augmented", "combined"):
        assert aug in stage, f"схема {aug} не запускается в probes-window"
    # genie -- положительный контроль кривой предрешенности
    assert "probe-commit-genie" in stage, (
        "нет положительного контроля: без genie непустое окно не на чем "
        "проверить, и нулевой результат по unconditional ничего не доказывает")


def test_long_probes_save_partial_results():
    """Диагностика, снятая по лимиту времени, обязана оставить хоть что-то.

    Усреднение по батчам устроено так, что половина батчей уже дает читаемую
    кривую; терять ее из-за среза -- значит платить за прогон дважды."""
    for name in ("probe_signal_survival.py", "probe_commitment.py"):
        body = open(name, encoding="utf-8").read()
        assert "partial=(b_idx + 1 < n_batches)" in body, (
            f"{name}: нет досохранения после батча")
        assert "partial=False" in body, f"{name}: нет финального сохранения"


def test_probe_time_limits_leave_room_for_loading():
    """Лимит должен многократно перекрывать оценку, иначе прогон срежется на
    загрузке датасета -- она на wikipedia занимает десятки минут и в оценку
    счета не входит. SLURM считает фактически потраченное время, а не лимит,
    поэтому запас не стоит GPU-часов."""
    import glob
    import re

    for script in sorted(glob.glob("probe_*.sh")):
        body = open(script, encoding="utf-8").read()
        m = re.search(r"#SBATCH --time=(\d+):", body)
        assert m, f"{script}: не задан лимит времени"
        assert int(m.group(1)) >= 2, (
            f"{script}: лимит {m.group(1)} ч -- меньше двух часов не хватит "
            f"на загрузку датасета и чекпоинтов")


def test_shell_scripts_are_executable_in_the_index():
    """Бит запуска хранится в индексе git, а не на диске.

    Репозиторий на кластере клонируется гитом (заливать с винды нельзя -- CRLF
    ломает .sh). Если файл лежит в индексе как 100644, после каждого клона
    приходится руками делать chmod +x, причем ровно в тот момент, когда хочется
    запускать прогоны. Новый .sh без бита ловится здесь, а не на кластере."""
    import subprocess

    out = subprocess.run(["git", "ls-files", "-s", "--", "*.sh"],
                         capture_output=True, text=True, check=True).stdout
    bad = [line.split("\t")[-1] for line in out.splitlines()
           if line and not line.startswith("100755")]
    assert not bad, (
        "в индексе git нет бита запуска у: " + ", ".join(bad) +
        "  --  поправить: git update-index --chmod=+x <файл>")


def test_probe_scripts_ask_for_one_gpu():
    """Диагностики -- инференс по одной карте. Четыре карты здесь означали бы,
    что кто-то скопировал шапку обучения, и бюджет проекта сгорел бы вчетверо
    быстрее при той же пользе."""
    import glob
    import re

    for script in sorted(glob.glob("probe_*.sh")):
        body = open(script, encoding="utf-8").read()
        m = re.search(r"#SBATCH --gpus-per-task=(\d+)", body)
        assert m, f"{script}: не задано число карт"
        assert int(m.group(1)) == 1, (
            f"{script}: запрошено {m.group(1)} карт, диагностике нужна одна")
        assert "source run_flags.sh" in body, (
            f"{script}: не подключен run_flags.sh -- ARCH_FLAGS/DATA_FLAGS "
            f"не соберутся, и прогон пойдет на конфиге по умолчанию")


def test_dilute_endpoints_and_monotonicity():
    """Сжатие к среднему: alpha=1 обязано вернуть оракула, alpha=0 -- среднее.

    Это КОНЦЫ оси. Если они съехали, вся непрерывная шкала смещена, и точка
    реального классификатора сравнивается не с тем."""
    from probes.degrade import degrade_target, dilute_target

    torch.manual_seed(0)
    target = torch.randn(4, 6, 12)
    # Среднее ПОЗИЦИОННОЕ, [L, D]: якорь нулевой информации -- это среднее
    # продолжение, а не один вектор на все позиции.
    mean = target.mean(dim=0)
    assert mean.shape == (6, 12)

    assert torch.allclose(dilute_target(target, 1.0), target, atol=1e-5)
    z0 = dilute_target(target, 0.0)
    assert torch.allclose(z0, mean.unsqueeze(0).expand_as(z0), atol=1e-5)

    def closeness(z):
        return -(z - target).pow(2).mean().item()

    scores = [closeness(degrade_target("dilute", target, lv)[0])
              for lv in (1, 5, 20, 50, 100)]
    assert scores == sorted(scores), f"dilute не монотонен: {scores}"


def test_noisy_is_monotone_and_reproducible():
    """Шумовая ось обязана быть монотонной и ВОСПРОИЗВОДИМОЙ.

    Невоспроизводимая порча означала бы, что две точки оси отличаются не только
    уровнем, и наклон кривой мерился бы вместе с шумом сэмплирования."""
    from probes.degrade import degrade_target

    torch.manual_seed(0)
    target = torch.randn(4, 6, 12)

    def closeness(z):
        return -(z - target).pow(2).mean().item()

    scores = [closeness(degrade_target("noisy", target, lv, seed=0)[0])
              for lv in (150, 100, 50, 25)]
    assert scores == sorted(scores), f"noisy не монотонен: {scores}"

    a = degrade_target("noisy", target, 50, seed=3)[0]
    b = degrade_target("noisy", target, 50, seed=3)[0]
    assert torch.equal(a, b), "один сид дает разную порчу"
    c = degrade_target("noisy", target, 50, seed=4)[0]
    assert not torch.equal(a, c), "сид не влияет на порчу"


def _clustered_bank(n_clusters=8, per=12, L=4, D=10, spread=0.2):
    from probes.manifold import LatentBank

    torch.manual_seed(0)
    centers = torch.randn(n_clusters, L * D) * 6.0
    seqs = (centers.repeat_interleave(per, dim=0)
            + torch.randn(n_clusters * per, L * D) * spread)
    return LatentBank(seqs[:, :D].contiguous(), knn=8, dim=4,
                      sequences=seqs.reshape(-1, L, D)), centers


def test_cluster_bundle_leaks_nothing_beyond_the_cluster_label():
    """РЕШАЮЩИЙ ИНВАРИАНТ сравнения форм условия.

    centroid, medoid и кандидаты несут РОВНО метку кластера. Если бы медоид
    выбирался по близости к истине, он нес бы больше бита и сравнение
    'множество против точки при равной информации' развалилось бы: множество
    выигрывало бы просто потому, что ему подсказали ответ.

    Проверяем прямо: две разные цели из одного кластера обязаны получить
    одинаковые центроид, медоид и набор кандидатов."""
    from probes.degrade import cluster_bundle

    bank, centers = _clustered_bank()
    L, D = bank.sequences.shape[1], bank.sequences.shape[2]
    base = centers[3].reshape(1, L, D)
    a = base + torch.randn(1, L, D) * 0.05
    b = base + torch.randn(1, L, D) * 0.05
    pair = torch.cat([a, b])

    cen, med, cand = cluster_bundle(bank, pair, 8, members=3)
    assert torch.allclose(cen[0], cen[1], atol=1e-5), "центроид зависит от цели"
    assert torch.allclose(med[0], med[1], atol=1e-5), (
        "медоид зависит от цели внутри кластера -- это утечка информации "
        "сверх метки кластера, сравнение форм станет нечестным")
    assert torch.allclose(cand[0], cand[1], atol=1e-5), (
        "набор кандидатов зависит от цели внутри кластера")


def test_cluster_bundle_candidates_are_real_points_of_the_manifold():
    """Медоид и кандидаты -- настоящие элементы банка, а не средние.

    В этом весь смысл разложения: centroid лежит ВНЕ многообразия, medoid и
    кандидаты -- на нем. Разность между ними и измеряет цену схода с
    многообразия. Если кандидаты окажутся усреднениями, измеряться будет
    ничто."""
    from probes.degrade import cluster_bundle

    bank, _ = _clustered_bank()
    L, D = bank.sequences.shape[1], bank.sequences.shape[2]
    flat_bank = bank.sequences.reshape(bank.sequences.shape[0], -1)
    target = bank.sequences[[5, 40]] + torch.randn(2, L, D) * 0.02

    cen, med, cand = cluster_bundle(bank, target, 8, members=4)
    for i in range(2):
        d = (flat_bank - med[i].reshape(-1)).pow(2).sum(dim=1)
        assert d.min() < 1e-6, "медоид не совпал ни с одним элементом банка"
        for j in range(cand.shape[1]):
            d = (flat_bank - cand[i, j].reshape(-1)).pow(2).sum(dim=1)
            assert d.min() < 1e-6, f"кандидат {j} не из банка"
    # Центроид усреднен, то есть почти наверняка НЕ элемент банка
    d = (flat_bank - cen[0].reshape(-1)).pow(2).sum(dim=1)
    assert d.min() > 1e-8, "центроид совпал с элементом банка -- кластер из одного"


def test_codebook_is_cached_so_levels_stay_comparable():
    """Кодовая книга считается один раз на банк и уровень.

    Две причины. Во-первых, k-means на каждом батче при K=512 это минуты на
    пустом месте. Во-вторых и важнее: точки оси обязаны отличаться ТОЛЬКО
    уровнем, а пересчет с другим состоянием генератора дал бы разные центроиды
    на разных батчах одного прогона."""
    from probes.degrade import sequence_codebook

    bank, _ = _clustered_bank()
    first = sequence_codebook(bank, 4)
    second = sequence_codebook(bank, 4)
    assert first[0] is second[0] and first[1] is second[1], (
        "кодовая книга пересчитывается -- центроиды поедут между батчами")


def test_dilute_stage_reaches_the_region_where_the_classifier_lives():
    """Непрерывная ось нужна именно низкими уровнями.

    Реальный классификатор несет один бит: восстановимость его условия на
    уровне пола 0.4605. Дискретные режимы в этот регион не попали (seqquant
    дал 0.307-0.320, то есть НИЖЕ пола, quantize -- 0.355 и выше). Без низких
    уровней dilute сравнивать классификатор по-прежнему не с чем."""
    stage = _probes_stage()
    assert "probes-dilute" in stage, "нет стадии непрерывной оси"
    assert "PROBE_DEGRADE=dilute" in stage
    assert "PROBE_DILUTE:-1 2 5" in stage, (
        "в свипе нет низких уровней -- ось не доходит до региона "
        "классификатора")
    assert "PROBE_DEGRADE=noisy" in stage, (
        "нет второй непрерывной оси: совпадение двух РАЗНЫХ по характеру "
        "порчей и есть доказательство, что важен объем информации")


def test_set_stage_compares_three_shapes_at_matched_information():
    """Стадия probes-set обязана гонять все три формы на ОДНОМ уровне.

    centroid -> medoid измеряет цену схода с многообразия, medoid -> mixture --
    выигрыш множественной формы. Без любой из трех точек разложение не
    считается, и объяснить, почему один бит классификатора обыгрывает два бита
    точки-цели, нечем."""
    stage = _probes_stage()
    assert "probes-set" in stage, "нет стадии сравнения форм условия"
    for shape in ("centroid", "medoid", "mixture"):
        assert shape in stage, f"форма {shape} не запускается"
    assert "PROBE_SET=${SHAPE}" in stage
    assert "PROBE_SEQ_BANK" in stage, (
        "банк последовательностей не задан: при бюджете в позициях их "
        "набирается 192, и K=256 падает на k-means")


def test_every_seqquant_launch_sizes_the_sequence_bank():
    """Квантизация по темам требует банка БОЛЬШЕ числа тем.

    Прогон 24.10: seqquant K=256 упал с 'центроидов (256) не меньше точек банка
    (192)' -- бюджет банка задан в позициях, а последовательностей из него
    выходит в сорок раз меньше. Проверяем все запуски seqquant разом.

    Склеиваем продолжения строк: запуск может быть разнесён обратными слешами,
    и построчная проверка ловила бы перевод строки, а не отсутствие банка."""
    stage = _probes_stage()
    for line in stage.replace("\\\n", " ").splitlines():
        if "PROBE_DEGRADE=seqquant" in line:
            assert "PROBE_SEQ_BANK" in line, (
                "запуск seqquant без размера банка последовательностей: "
                + line.strip())


def test_help_covers_the_whole_header():
    """Справка печатается диапазоном строк, и этот диапазон обязан покрывать
    шапку целиком.

    Стадию легко добавить и забыть про диапазон -- тогда она есть, работает, но
    в справке ее не видно, и узнать о ней можно только из кода."""
    import re

    body = open("run_wikipedia.sh", encoding="utf-8").read()
    lines = body.splitlines()
    m = re.search(r"sed -n '2,(\d+)p'", body)
    assert m, "справка печатается не диапазоном строк -- поправьте тест"
    shown = int(m.group(1))
    head_end = next(i for i, l in enumerate(lines) if l.startswith("set -e"))
    assert shown >= head_end - 1, (
        f"справка печатает {shown} строк, а шапка кончается на {head_end}: "
        f"последние стадии в нее не попадут")
    for stage in ("probes-dilute", "probes-set", "probes-infoaxis"):
        assert any(stage in l for l in lines[1:shown]), (
            f"стадии {stage} нет в печатаемой справке")


def _write_dump(tmp_path, name, refs, as_dict=True):
    import json

    d = tmp_path / name
    d.mkdir()
    rows = [{"TRG": t, "GEN": "x", "SRC": "y"} for t in refs]
    payload = {"wikipedia": rows} if as_dict else rows
    f = d / "87500-N=50-seed=0-len=1-gs=0.0.json"
    f.write_text(json.dumps(payload), encoding="utf-8")
    return str(f)


def test_eval_alignment_check_catches_different_example_sets(tmp_path):
    """СРАВНИМОСТЬ ТАБЛИЦЫ. Подходы обязаны быть оценены на одних примерах.

    Дампы 24.10: genie против diffuseq -- 5000/5000 позиционно, genie против
    guidance -- 1500/5000 при пересечении множеств 4500. Причина в сборе
    генераций: ранг останавливается на границе батча за порогом
    num_gen_texts // world_size, а у guidance validation.batch_size=128 вместо
    500. На выводы это не влияет (сдвиг среднего ~0.0005 при эффектах в сотых),
    но таблицу статьи так строить нельзя, и узнать об этом надо ДО прогона на
    20 сидов."""
    import subprocess

    same_a = _write_dump(tmp_path, "approach-a", [f"ref {i}" for i in range(20)])
    same_b = _write_dump(tmp_path, "approach-b", [f"ref {i}" for i in range(20)],
                         as_dict=False)
    shifted = _write_dump(tmp_path, "approach-c",
                          [f"ref {i}" for i in range(2, 22)])

    def run(*paths):
        return subprocess.run(
            [sys.executable, "check_eval_alignment.py", *paths],
            capture_output=True, text=True, encoding="utf-8", errors="replace")

    ok = run(same_a, same_b)
    assert ok.returncode == 0, (
        "одинаковые наборы объявлены разными: " + ok.stdout)

    bad = run(same_a, shifted)
    assert bad.returncode == 1, (
        "разные наборы приняты за одинаковые -- проверка бесполезна: "
        + bad.stdout)
    assert "GUIDANCE_EVAL_BS" in bad.stdout, (
        "в отчете нет причины расхождения, и починить его по нему нельзя")


if __name__ == "__main__":
    sys.exit(pytest.main([os.path.abspath(__file__), "-v"]))


def test_snap_pulls_to_real_objects_of_the_bank():
    """Притягивание обязано давать РЕАЛЬНЫЕ объекты, и именно ближайшие.

    Весь метод держится на том, что цель после притягивания лежит на данных.
    Если snap вернёт что-то своё, метод проверяет не то, что заявлено."""
    from probes.manifold import LatentBank

    torch.manual_seed(0)
    vecs = torch.randn(200, 6)
    seqs = torch.randn(30, 4, 6)
    bank = LatentBank(vecs, knn=8, dim=3, sequences=seqs)
    x = torch.randn(3, 4, 6)

    pos = bank.snap_positions(x)
    brute = vecs[torch.cdist(x.reshape(-1, 6), vecs).argmin(dim=1)]
    assert torch.allclose(pos, brute.reshape(x.shape)), (
        "позиционное притягивание вернуло не ближайший вектор банка")

    seq = bank.snap_sequence(x, torch.ones(3, 4, dtype=torch.bool))
    d = (seqs.unsqueeze(0) - seq.unsqueeze(1)).pow(2).flatten(2).sum(2)
    assert d.min(dim=1).values.max() < 1e-5, (
        "притягивание последовательности вернуло не элемент банка")


def test_snap_alpha_zero_is_raw_and_alpha_keeps_the_step():
    """ДВА ИНВАРИАНТА МЕТОДА, без которых свип по alpha не читается.

    alpha=0 обязан воспроизводить raw ПОБИТОВО: это встроенный контроль на то,
    что свип меряет притягивание к данным, а не побочный эффект переписанного
    кода. И доля шага обязана сохраняться при любом alpha: смешиваются
    НАПРАВЛЕНИЯ, иначе alpha менял бы заодно размер шага, и сравнение
    показывало бы не «куда идти», а «как далеко»."""
    from probe_projected_guidance import _blend_with_manifold
    from probes.manifold import LatentBank

    torch.manual_seed(0)
    bank = LatentBank(torch.randn(200, 6), knn=8, dim=3,
                      sequences=torch.randn(30, 4, 6))
    g = torch.randn(3, 4, 6)
    out = {"x_0": torch.randn(3, 4, 6)}
    score_norm = torch.full((3,), 2.0)
    ratio = 0.1
    diag = {"snap_dist": [], "cos_cls_manifold": []}

    same = _blend_with_manifold(bank, "snap-pos", 0.0, g, out, score_norm,
                                ratio, diag)
    assert same is g, "alpha=0 изменил градиент -- контроль сломан"
    # Но диагностики при alpha=0 обязаны считаться: прогон 26.10 вернул в них
    # «нет (guidance выключен)» ровно в той точке, ради которой контроль и
    # запускался первым. Они не зависят от того, применяем ли притягивание, и
    # в контроле нужны больше всего -- это замер НЕТРОНУТОГО guidance.
    assert diag["cos_cls_manifold"] and diag["snap_dist"], (
        "при alpha=0 не померены удалённость от данных и косинус с ними")

    for mode in ("snap-pos", "snap-seq"):
        for alpha in (0.25, 0.5, 1.0):
            res = _blend_with_manifold(bank, mode, alpha, g, out, score_norm,
                                       ratio, diag)
            step = res.flatten(1).norm(dim=1)
            assert torch.allclose(step, ratio * score_norm, atol=1e-5), (
                f"{mode} при alpha={alpha} изменил долю шага: {step.tolist()}")


def test_snap_stage_sweeps_alpha_from_the_control():
    """Стадия обязана включать alpha=0 -- без него свип не с чем сравнивать."""
    import re

    stage = _probes_stage()
    assert "probes-snap" in stage, "нет стадии притягивания цели к данным"
    assert "PROBE_GRAD_MODE=${MODE}" in stage
    m = re.search(r"PROBE_SNAP_ALPHAS:-([0-9 .]+)\}", stage)
    assert m, "в стадии нет сетки по alpha"
    alphas = [float(x) for x in m.group(1).split()]
    assert 0.0 in alphas, f"нет контрольной точки alpha=0: {alphas}"
    assert max(alphas) >= 1.0, f"свип не доходит до полного притягивания: {alphas}"

    # Доля шага обязана быть рабочей. Свип 24.10 нашёл пик raw при 0.005, а при
    # 0.1 ВСЕ режимы ниже пола; контрольный прогон 26.10 при 0.1 дал 0.4499 и
    # пустые генерации. Чинить guidance поверх уже разрушенного шага -- значит
    # мерить починку поверх поломки.
    # Привязываемся к ТЕЛУ стадии: имя probes-snap встречается и раньше, в
    # перечне стадий ветки, и срез оттуда поймал бы долю шага у probes-projected.
    body = stage[stage.index('if [ "$1" = "probes-snap" ]'):]
    r = re.search(r"PROBE_TARGET_RATIO:-([0-9.]+)\}", body)
    assert r and float(r.group(1)) <= 0.01, (
        f"стадия snap идёт на доле шага {r.group(1) if r else '?'} -- "
        f"guidance там уже разрушен")


def test_snap_bank_comes_from_train_not_test():
    """Банк МЕТОДА берётся из train.

    Притягивать генерации к латентам тестового сплита -- значит подглядывать в
    ответ, и работа снимается на этом одном предложении. У диагностик (tangent)
    остаётся test с пропуском оцениваемых примеров, там это допустимо."""
    body = open("probe_projected_guidance.py", encoding="utf-8").read()
    assert '"PROBE_BANK_SPLIT", "train" if mode.startswith("snap") else "test"' \
        in body, "snap-режимы берут банк не из train"
    assert 'skip_examples=0 if bank_split != "test" else num_texts' in body, (
        "пропуск примеров применяется не только к test")


def test_bank_cache_round_trips_and_refuses_a_stale_file(tmp_path):
    """Кэш банка обязан читаться и ОТКАЗЫВАТЬСЯ при несовпадении.

    Препроцессинг сплита в data/dataset_wiki.py идёт с
    load_from_cache_file=False, то есть повторяется в каждом задании: на train
    это 1 час 22 минуты (замер 26.10). Свип из десяти заданий отдал бы этому 13
    GPU-часов, поэтому кэшируется результат -- сам банк.

    Молча принять чужой кэш нельзя: банк, снятый под другой чекпоинт или сплит,
    описывал бы не ту модель и не те данные, а прогон выглядел бы успешным."""
    from probes.manifold import LatentBank

    torch.manual_seed(0)
    path = str(tmp_path / "bank.pt")
    bank = torch.randn(64, 6)
    seqs = torch.randn(8, 4, 6)
    masks = torch.ones(8, 4, dtype=torch.bool)
    meta = {"split": "train", "size": 64, "skip": 0,
            "min_sequences": 8, "checkpoint": "ck/87500.pth"}
    LatentBank._save_cache(path, bank, seqs, masks, meta)

    got = LatentBank._load_cache(path, dict(meta), knn=8, dim=3)
    assert got is not None and torch.allclose(got.bank.cpu(), bank), (
        "кэш не прочитался или вернул другие векторы")

    assert LatentBank._load_cache(str(tmp_path / "нет.pt"), dict(meta)) is None, (
        "отсутствующий кэш должен молча означать «собрать заново»")

    for key, bad in (("checkpoint", "ck/50000.pth"), ("split", "test"),
                     ("size", 128)):
        spoiled = dict(meta)
        spoiled[key] = bad
        with pytest.raises(RuntimeError):
            LatentBank._load_cache(path, spoiled, knn=8, dim=3)

    # Для числа последовательностей условие односторонее: большего банка
    # хватает, меньшего -- нет.
    more = dict(meta)
    more["min_sequences"] = 16
    with pytest.raises(RuntimeError):
        LatentBank._load_cache(path, more, knn=8, dim=3)
    less = dict(meta)
    less["min_sequences"] = 4
    assert LatentBank._load_cache(path, less, knn=8, dim=3) is not None, (
        "кэш с ИЗБЫТКОМ последовательностей отвергать не за что")


def test_diagnostics_are_resolved_by_noise_level():
    """Среднее по всей траектории отвечает не на тот вопрос.

    Замер 27.10 дал удалённость 1.2617 и косинус -0.0656 в среднем по всем
    шагам солвера. Но при t -> 1 оценка x_0 заведомо мусорна, уводить её
    некуда, и среднее определяется именно этими шагами. Текст решается при
    МАЛОМ t, и содержателен сход с данных именно там: если он сосредоточен при
    большом t, притягивание надо включать не на всей траектории."""
    from probe_projected_guidance import bins_by_t

    ts = [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    vals = [1.0, 3.0, 10.0, 20.0, 30.0, 40.0, 50.0]
    rows = bins_by_t(ts, vals)
    assert [r[2] for r in rows] == [2, 1, 1, 1, 2], (
        f"замеры разложены по корзинам неверно: {[r[2] for r in rows]}")
    assert rows[0][3] == 2.0, "среднее в первой корзине посчитано неверно"
    # t ровно 1.0 обязан попасть в последнюю корзину, а не выпасть: это самый
    # шумный конец, где и живёт артефакт усреднения.
    assert rows[-1][3] == 45.0, "t=1.0 выпал из разбивки"

    assert bins_by_t([], []) == [], "пустой замер не должен падать"
    assert bins_by_t([0.5], [1.0, 2.0]) == [], (
        "рассогласованные длины обязаны давать пусто, а не молча считать")
    # Пустая корзина -- это None, а не ноль: ноль читался бы как измеренное
    # значение.
    assert bins_by_t([0.9], [1.0])[0][3] is None


def test_conditional_lift_is_zero_without_conditioning():
    """УСЛОВНЫЙ ПРИРОСТ обязан быть нулевым у несвязанных генераций.

    bert не различает "улучшилась обусловленность" и "улучшилась беглость": при
    alpha=1, где условного сигнала нет вообще, он вышел 0.46596 против пола
    0.4605. Прирост F1(своя цель) - F1(чужая цель) от беглости свободен -- всё,
    что улучшает текст вообще, поднимает оба слагаемых и из разности уходит."""
    from analyze_paired_lift import content_words, f1, per_example

    assert content_words("The cat and THE dog") == ["cat", "dog"], (
        "служебные слова не отфильтрованы -- они совпадают у любых двух "
        "английских текстов и глушат сигнал")
    assert f1(["a"], []) == 0.0 and f1([], ["a"]) == 0.0

    # Генерации не связаны со своими целями: прирост обязан быть около нуля.
    texts = [{"GEN": "alpha beta", "TRG": "gamma delta"},
             {"GEN": "gamma delta", "TRG": "alpha beta"}]
    own, other = per_example(texts)
    assert sum(own) / 2 == 0.0, "несвязанные генерации дали ненулевое совпадение"

    # Генерации совпадают со своими целями: прирост обязан быть большим.
    texts = [{"GEN": "alpha beta", "TRG": "alpha beta"},
             {"GEN": "gamma delta", "TRG": "gamma delta"}]
    own, other = per_example(texts)
    lift = sum(o - x for o, x in zip(own, other)) / 2
    assert lift > 0.9, f"прирост у точного совпадения всего {lift:.3f}"


def test_paired_analysis_refuses_runs_without_full_texts():
    """Прогон, снятый до дампа всех текстов, парную статистику не даёт.

    Молча посчитать по двадцати примерам из поля examples -- значит выдать
    доверительный интервал, который в семь раз уже настоящего."""
    import json as _json

    from analyze_paired_lift import load

    p = "probe_results/_тест_без_текстов.json"
    os.makedirs("probe_results", exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        _json.dump({"metrics": {"bert-score": 0.5},
                    "examples": [{"GEN": "a", "TRG": "b"}]}, f)
    try:
        with pytest.raises(SystemExit):
            load(p)
    finally:
        os.remove(p)


# =====================================================================
# Парная статистика: отдельное задание поверх уже снятых дампов
# =====================================================================

def test_lift_stage_launches_the_analysis_without_a_gpu():
    """Стадия probes-lift ничего не считает заново и не занимает карту.

    Задание читает json-ы, уже записанные probes-snap, и делает униграммный F1
    -- секунды чистого питона. Запрос GPU здесь означал бы, что решающий для
    рамки статьи ответ стоит в очереди за обучением.
    """
    stage = _probes_stage()
    assert 'if [ "$1" = "probes-lift" ]' in stage, "нет стадии парной статистики"
    body = stage[stage.index('if [ "$1" = "probes-lift" ]'):]
    body = body[:body.index("fi")]
    assert "analyze_paired_lift.sh" in body

    sbatch_directives = [
        ln for ln in open("analyze_paired_lift.sh", encoding="utf-8")
        if ln.startswith("#SBATCH")]
    assert not any("gpus" in ln or "gres" in ln for ln in sbatch_directives), (
        "задание просит карту, хотя считает на CPU")
    # escape-люк: аргумент sbatch перебивает директивы файла, если очередь
    # вообще не принимает задания без карты
    assert "LIFT_GPU" in body, "нет способа запросить карту, если очередь требует"


def test_lift_job_pairs_only_comparable_runs():
    """Шаблон отбора прогонов приколот к датасету И к доле шага.

    Парная разность считается по примерам: прогоны обязаны идти на одних
    промптах. Свести в одну таблицу wikipedia с rocstories или ratio 0.005 с
    0.1 -- значит вычитать друг из друга разные наборы примеров.
    """
    src = open("analyze_paired_lift.sh", encoding="utf-8").read()
    glob_line = next(ln for ln in src.splitlines() if ln.startswith("LIFT_GLOB="))
    assert "${DATASET:-wikipedia}" in glob_line, "шаблон не приколот к датасету"
    assert "${PROBE_TARGET_RATIO:-0.005}" in glob_line, "шаблон не приколот к доле шага"


def test_lift_job_fails_loudly_when_there_is_nothing_to_pair():
    """Пустой шаблон обязан быть ошибкой, а не пустой таблицей.

    nullglob превращает непопавший шаблон в НОЛЬ аргументов, и analyze_paired_lift.py
    молча взял бы свой собственный дефолтный glob -- то есть не те прогоны.
    """
    src = open("analyze_paired_lift.sh", encoding="utf-8").read()
    assert "nullglob" in src
    assert "exit 1" in src, "задание не падает, когда json-ов нет"
    assert "ls -1 probe_results/" in src, "в логе не видно, что вообще лежит"
    assert "tee" in src, "результат не остаётся нигде, кроме лога"
