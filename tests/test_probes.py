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


def test_sweep_searches_below_the_measured_overshoot():
    """Замер 24.10: при доле шага 0.1 все режимы оказались НИЖЕ пола, то есть
    0.1 уже за пиком. При lambda=10 из ВКР доля была 0.007-0.017 и давала
    прирост. Значит свип обязан покрывать именно этот диапазон, а не идти
    вверх от 0.1 -- иначе оптимума он не найдет."""
    stage = _probes_stage()
    assert "PROBE_RATIOS:-0.002" in stage, (
        "свип не спускается ниже измеренного перелета")
    import re
    ratios = re.search(r"PROBE_RATIOS:-([0-9. ]+)\}", stage)
    assert ratios, "диапазон свипа не задан"
    values = [float(x) for x in ratios.group(1).split()]
    assert min(values) <= 0.005 and max(values) <= 0.05, (
        f"диапазон {values} не накрывает измеренную рабочую область 0.007-0.017")


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


if __name__ == "__main__":
    sys.exit(pytest.main([os.path.abspath(__file__), "-v"]))
