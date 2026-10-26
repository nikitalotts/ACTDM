"""Локальное касательное пространство многообразия энкодингов.

ЗАЧЕМ
-----
Замер probe_signal_survival дал главный факт работы: классификатор различает
правильное продолжение от чужого с AUC 0.99 на всем диапазоне шума, но косинус
его градиента с направлением к истинному продолжению равен 0.009 при случайной
базе 0.0001. Эту пару чисел нельзя читать как отношение: cos_random --
среднее ЗНАКОВОЙ величины, оно сходится к нулю с ростом выборки, и "во сколько
раз лучше" можно сделать любым, просто взяв больше примеров. Честная база --
СКО случайного косинуса: в пространстве 64x768 это 1/sqrt(49152) = 0.0045.
Значит 0.009 -- это ДВЕ случайные СКО (на тысяче примеров это десятки сигм от
нуля, то есть сигнал есть, но он слабый), и примерно 8e-05 энергии градиента
лежит в нужном направлении. Остальное ортогонально тому, куда надо идти.

Гипотеза работы: ортогональная часть -- это движение ПРОЧЬ С МНОГООБРАЗИЯ.
Энкодинги BERT лежат тонким кластерным множеством, его локальная размерность
много меньше 768, и почти любое направление из точки данных уводит в пустоту,
где ни декодер, ни классификатор не определены. Отсюда и невозможность поднять
lambda: ортогональная компонента разрушает сэмпл раньше, чем касательная
накопит полезный сдвиг.

Модуль дает две вещи:
  * ИЗМЕРЕНИЕ -- какая доля энергии градиента лежит в касательном
    подпространстве (с честной базой: у случайного вектора эта доля равна
    dim/D, и без нее число не читается);
  * МЕТОД -- проекцию градиента на касательное подпространство, после которой
    lambda можно поднимать, не сходя с многообразия.

КАК СЧИТАЕТСЯ КАСАТЕЛЬНОЕ
-------------------------
Локальный PCA по k ближайшим соседям из банка чистых энкодингов. Базис берется
через грам-матрицу [k, k], а не через SVD матрицы [k, 768]: k много меньше 768,
и собственные векторы грама дают тот же базис в разы дешевле. Собственные
векторы u_i грам-матрицы C C^T дают строки V_i = u_i^T C, ортогональные с
нормой sqrt(lambda_i), -- после нормировки это ортонормированный базис.

Касательное оценивается в x_0 (текущая оценка чистого состояния), а НЕ в x_t:
x_t зашумлен и лежит вне многообразия по построению, локальные соседи там
бессмысленны.
"""
from typing import Optional, Sequence

import torch

from probes.common import (content_mask, encode_batch, log, masked_norm,
                           probe_loader)


class LatentBank:
    """Банк чистых энкодингов плюс локальное касательное в произвольной точке."""

    def __init__(self, vectors: torch.Tensor, knn: int = 32, dim: int = 16,
                 query_chunk: int = 1024, bank_chunk: int = 8192,
                 sequences: Optional[torch.Tensor] = None,
                 sequence_masks: Optional[torch.Tensor] = None):
        if vectors.dim() != 2:
            raise ValueError(f"банк ожидается [N, D], получено {tuple(vectors.shape)}")
        if knn <= dim:
            raise ValueError(
                f"соседей ({knn}) должно быть больше размерности касательного "
                f"({dim}): иначе базис вырожден и проекция тождественна")
        self.bank = vectors
        self.knn = knn
        self.dim = dim
        self.query_chunk = query_chunk
        self.bank_chunk = bank_chunk
        # Латенты целых продолжений [N, L, D]. Нужны только оси информативности
        # (probes/degrade.py, режим seqquant -- квантизация по темам); для
        # касательного пространства работаем на уровне позиций.
        self.sequences = sequences
        # Содержательные позиции последовательностей банка [N, L]. Нужны
        # sequence_gap: расстояние меряется по позициям запроса, и у
        # последовательностей банка паддинг обязан исключаться по тому же
        # правилу, иначе числитель и знаменатель нормировки считаются по
        # разным правилам и отношение теряет смысл.
        if sequences is not None and sequence_masks is None:
            sequence_masks = torch.ones(sequences.shape[:2], dtype=torch.bool,
                                        device=sequences.device)
        self.sequence_masks = sequence_masks

    @property
    def D(self) -> int:
        return self.bank.shape[1]

    @property
    def random_tangent_fraction(self) -> float:
        """Доля энергии СЛУЧАЙНОГО вектора в касательном подпространстве.

        Для равномерно случайного направления в R^D ожидание доли энергии в
        любом фиксированном dim-мерном подпространстве равно dim/D. Это и есть
        база сравнения: без нее «доля 0.1» ничего не значит.
        """
        return self.dim / self.D

    @classmethod
    def from_runner(cls, runner, size: int = 8192, skip_examples: int = 0,
                    split: str = "test", batch_size: int = 64,
                    min_sequences: int = 0, **kw) -> "LatentBank":
        """Собрать банк позиционных векторов чистых энкодингов.

        skip_examples пропускает начало сплита -- те примеры, на которых потом
        считается генерация. Иначе касательное в точке оценивалось бы по ее же
        соседям из того же текста, и проекция льстила бы себе.

        min_sequences -- сколько целых последовательностей набрать минимум.
        Бюджета в векторах для этого мало: 8192 позиции при ~43 содержательных
        на пример -- это всего 192 последовательности, и seqquant с level=256
        падал на «центроидов не меньше точек банка». Квантизация по темам
        требует банка заметно больше числа тем.
        """
        log(f"собираю банк латентов: {size} векторов "
            f"(пропускаю первые {skip_examples} примеров)")
        loader = probe_loader(runner, split=split, batch_size=batch_size)
        chunks, seqs, seq_masks, seen, taken = [], [], [], 0, 0
        for batch in loader:
            n = len(batch["text_trg"])
            if seen + n <= skip_examples:
                seen += n
                continue
            seen += n
            batch, _, trg_x = encode_batch(runner, batch)
            mask = content_mask(runner, batch["input_ids_trg"])
            v = trg_x[mask].detach().float()
            chunks.append(v)
            seqs.append(trg_x.detach().float())
            seq_masks.append(mask.detach())
            taken += v.shape[0]
            got_seq = sum(c.shape[0] for c in seqs)
            if taken >= size and got_seq >= min_sequences:
                break
        if not chunks:
            raise RuntimeError(
                "банк пуст: skip_examples больше размера сплита")
        bank = torch.cat(chunks)[:size].contiguous()
        sequences = torch.cat(seqs).contiguous()
        sequence_masks = torch.cat(seq_masks).contiguous()
        log(f"банк готов: позиции {tuple(bank.shape)}, "
            f"последовательности {tuple(sequences.shape)}")
        return cls(bank, sequences=sequences,
                   sequence_masks=sequence_masks, **kw)

    def _knn_index(self, x: torch.Tensor) -> torch.Tensor:
        """Индексы k ближайших точек банка. x: [M, D] -> [M, k]."""
        best_d = None
        best_i = None
        for start in range(0, self.bank.shape[0], self.bank_chunk):
            sl = self.bank[start:start + self.bank_chunk]
            d = torch.cdist(x, sl)
            k = min(self.knn, sl.shape[0])
            dv, di = torch.topk(d, k, dim=1, largest=False)
            di = di + start
            if best_d is None:
                best_d, best_i = dv, di
            else:
                best_d = torch.cat([best_d, dv], dim=1)
                best_i = torch.cat([best_i, di], dim=1)
                dv, order = torch.topk(best_d, self.knn, dim=1, largest=False)
                best_d = dv
                best_i = torch.gather(best_i, 1, order)
        return best_i[:, :self.knn]

    def tangent_basis(self, x: torch.Tensor) -> torch.Tensor:
        """Ортонормированный базис касательного в каждой точке. [M, D] -> [M, dim, D]."""
        idx = self._knn_index(x)
        nbr = self.bank[idx]                                  # [M, k, D]
        c = nbr - nbr.mean(dim=1, keepdim=True)
        gram = c @ c.transpose(1, 2)                          # [M, k, k]
        # eigh отдает по возрастанию -- берем последние dim, это главные оси
        _, evecs = torch.linalg.eigh(gram.float())
        top = evecs[:, :, -self.dim:]                         # [M, k, dim]
        basis = top.transpose(1, 2) @ c                       # [M, dim, D]
        return basis / basis.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    def basis_for(self, x: torch.Tensor) -> torch.Tensor:
        """Базисы касательного для всех точек x: [..., D] -> [M, dim, D].

        Вынесено отдельно, потому что базис зависит ТОЛЬКО от точки. В
        probe_signal_survival один и тот же набор чистых латентов проецируется
        на 21 уровне шума: считать kNN и собственные векторы заново на каждом
        уровне -- в 21 раз дороже без всякой разницы в результате.
        """
        xx = x.reshape(-1, x.shape[-1]).float()
        parts = []
        for s in range(0, xx.shape[0], self.query_chunk):
            parts.append(self.tangent_basis(xx[s:s + self.query_chunk]))
        return torch.cat(parts)

    @staticmethod
    def project_with(g: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
        """Проекция g на уже посчитанный базис. g: [..., D], basis: [M, dim, D]."""
        shape = g.shape
        gg = g.reshape(-1, shape[-1]).float()
        coef = torch.einsum("mkd,md->mk", basis, gg)
        out = torch.einsum("mk,mkd->md", coef, basis)
        return out.reshape(shape).to(g.dtype)

    def project(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Проекция g на касательное в x. Обе формы [..., D], результат той же формы."""
        return self.project_with(g, self.basis_for(x))

    @staticmethod
    def fraction(g: torch.Tensor, proj: torch.Tensor,
                 mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Доля энергии g, попавшая в проекцию. Возвращает [B] по примерам.

        Сравнивать надо с random_tangent_fraction = dim/D. Доля ВЫШЕ нее --
        градиент хоть сколько-то следует многообразию; доля НА НЕЙ -- он
        неотличим от случайного направления, то есть просто уводит в пустоту.
        """
        if mask is not None:
            m = mask.unsqueeze(-1).to(g.dtype)
            g, proj = g * m, proj * m
        num = proj.flatten(1).pow(2).sum(dim=1)
        den = g.flatten(1).pow(2).sum(dim=1).clamp_min(1e-20)
        return num / den

    def tangent_fraction(self, g: torch.Tensor, x: torch.Tensor,
                         mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.fraction(g, self.project(g, x), mask)

    @property
    def nn_scale(self) -> float:
        """Медианное расстояние от точки банка до ближайшего соседа.

        Собственный масштаб данных. Все расстояния до многообразия делятся на
        него, чтобы число было безразмерным и сравнивалось между прогонами и
        чекпоинтами: 1.0 означает «так же далеко от данных, как типичная точка
        данных от своего соседа».
        """
        if getattr(self, "_nn_scale", None) is None:
            sub = self.bank[:min(2048, self.bank.shape[0])]
            d = torch.cdist(sub, self.bank)
            d.scatter_(1, torch.arange(sub.shape[0], device=d.device)
                       .unsqueeze(1), float("inf"))
            self._nn_scale = float(d.min(dim=1).values.median().item())
        return self._nn_scale

    def manifold_gap(self, x: torch.Tensor,
                     mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Насколько точка удалена от многообразия данных. [B,L,D] -> [B].

        ЗАЧЕМ ЭТО НУЖНО. Замер 25.10 дал главный факт работы: при РОВНО
        одинаковой информации (метка кластера) цель-центроид даёт bert 0.394, а
        реальный элемент того же кластера -- 0.496. Но это ОДНА пара точек, и
        как закон она не читается: нужна непрерывная величина, по которой
        падение качества предсказуемо. Вот она.

        Считается как медианное по содержательным позициям расстояние до
        ближайшего вектора банка, делённое на собственный масштаб данных
        (nn_scale). У реальной точки данных значение около 1, у точки в
        пустоте -- заметно больше.

        ВНИМАНИЕ: это зазор по ОТДЕЛЬНЫМ ПОЗИЦИЯМ, и как главная ось он не
        годится -- см. sequence_gap, где разобрано, почему на свипе он едет в
        обратную сторону. Здесь он оставлен второй колонкой: он отвечает на
        другой вопрос -- правдоподобна ли каждая позиция сама по себе.
        """
        xx = x.reshape(-1, x.shape[-1]).float()
        # Держим ДВА ближайших, а не одно. Медоид -- это буквально элемент
        # банка, и по одному ближайшему его зазор вышел бы нулевым, то есть
        # нижний конец свипа схлопнулся бы в ноль и ось перестала бы работать.
        # Та же причина, по которой hole_ratio исключает концы отрезка из
        # поиска соседа. Закреплено тестом.
        two = None
        for s in range(0, self.bank.shape[0], self.bank_chunk):
            d = torch.cdist(xx, self.bank[s:s + self.bank_chunk])
            k = min(2, d.shape[1])
            dv = torch.topk(d, k, dim=1, largest=False).values
            if dv.shape[1] < 2:
                dv = torch.cat([dv, dv[:, -1:]], dim=1)
            two = dv if two is None else torch.topk(
                torch.cat([two, dv], dim=1), 2, dim=1, largest=False).values
        scale = max(self.nn_scale, 1e-12)
        eps = 1e-4 * scale
        best = torch.where(two[:, 0] > eps, two[:, 0], two[:, 1])
        best = best.reshape(x.shape[0], x.shape[1]) / scale
        if mask is None:
            return best.median(dim=1).values
        # Позиции паддинга исключаем: там у всех примеров один и тот же
        # zero_emb, расстояние до него вырождено и занижало бы медиану.
        out = torch.empty(x.shape[0], device=x.device, dtype=best.dtype)
        for i in range(x.shape[0]):
            m = mask[i]
            out[i] = best[i][m].median() if m.any() else best[i].median()
        return out

    # --- зазор в пространстве ПОСЛЕДОВАТЕЛЬНОСТЕЙ ------------------------

    @property
    def _seq_sqnorm(self) -> torch.Tensor:
        """Квадраты норм позиционных векторов банка. [N, L]."""
        if getattr(self, "_seq_sqnorm_cache", None) is None:
            self._seq_sqnorm_cache = self.sequences.pow(2).sum(dim=-1)
        return self._seq_sqnorm_cache

    def _seq_dist(self, x: torch.Tensor,
                  mask: torch.Tensor) -> torch.Tensor:
        """Расстояния от x до последовательностей банка. [B,L,D] -> [B, N].

        Нормировано на одну содержательную позицию: иначе сравнивались бы
        суммы по разному числу позиций, и у длинного примера зазор выходил бы
        больше просто из-за длины.

        Маска запроса применяется к ОБЕИМ сторонам, но банк при этом не
        копируется: раскрываем |x-s|^2 = |x|^2 - 2<x,s> + |s|^2 и маскируем
        слагаемые по отдельности. Прямой путь требовал бы маскированной копии
        банка на каждый пример -- 6040 x 49152 float, около 1.2 ГБ.
        """
        if self.sequences is None:
            raise RuntimeError(
                "банк собран без последовательностей: зазор по "
                "последовательностям мерить нечем (нужен min_sequences > 0)")
        n, L, D = self.sequences.shape
        if tuple(x.shape[1:]) != (L, D):
            raise ValueError(
                f"ожидается [B, {L}, {D}], получено {tuple(x.shape)}")
        xf = x.float()
        m = mask.to(xf.dtype)                                   # [B, L]
        cross = (xf * m.unsqueeze(-1)).reshape(x.shape[0], L * D) \
            @ self.sequences.reshape(n, L * D).t()              # [B, N]
        a = (xf.pow(2).sum(dim=-1) * m).sum(dim=-1)             # [B]
        b = m @ self._seq_sqnorm.t()                            # [B, N]
        d2 = (a.unsqueeze(1) + b - 2 * cross).clamp_min(0)
        cnt = m.sum(dim=1).clamp_min(1.0).unsqueeze(1)
        return (d2 / cnt).sqrt()

    @property
    def seq_nn_scale(self) -> float:
        """Медианное расстояние от последовательности банка до ближайшей.

        Собственный масштаб данных на уровне последовательностей -- ровно то
        же, чем nn_scale служит для позиций. Считается по подвыборке банка,
        каждая со своей маской: числитель зазора тоже маскируется маской
        запроса, и правило должно быть одно.
        """
        if getattr(self, "_seq_nn_scale", None) is None:
            n = self.sequences.shape[0]
            k = min(256, n)
            idx = torch.arange(0, n, max(1, n // k),
                               device=self.sequences.device)[:k]
            d = self._seq_dist(self.sequences[idx], self.sequence_masks[idx])
            d.scatter_(1, idx.unsqueeze(1), float("inf"))
            self._seq_nn_scale = float(d.min(dim=1).values.median().item())
        return self._seq_nn_scale

    def sequence_gap(self, x: torch.Tensor,
                     mask: torch.Tensor) -> torch.Tensor:
        """Удалённость ЦЕЛОЙ последовательности от данных. [B,L,D] -> [B].

        ГЛАВНАЯ ось геометрии работы, и именно она, а не manifold_gap.

        ПОЧЕМУ ПОЗИЦИОННОГО ЗАЗОРА НЕ ХВАТИЛО. Свип 26.10 (offmanifold, beta
        ведёт цель от медоида к центроиду) дал по manifold_gap падение
        1.110 -> 0.795: центроид вышел БЛИЖЕ к данным, чем реальная
        последовательность. Это не ошибка счёта, а не то многообразие.
        Центроид усредняет сотни последовательностей поэлементно, поэтому
        каждый его позиционный вектор съезжает к среднему по позиции, то есть
        в САМУЮ ПЛОТНУЮ область позиционного облака -- туда, где живут "the",
        запятая и точка. Каждая позиция по отдельности правдоподобна; а
        последовательность невозможна, и это прямо видно в тексте: STEER
        центроида декодируется в "s the the the the". Сеть шумоподавления
        определена на целой последовательности, значит и многообразие, с
        которого цель сходит, -- многообразие последовательностей.

        Считается как расстояние до ближайшей РЕАЛЬНОЙ последовательности
        банка, делённое на собственный масштаб данных (seq_nn_scale). У
        реального элемента банка значение около 1, у точки в пустоте больше.
        """
        d = self._seq_dist(x, mask)
        two = torch.topk(d, min(2, d.shape[1]), dim=1, largest=False).values
        if two.shape[1] < 2:
            two = torch.cat([two, two[:, -1:]], dim=1)
        scale = max(self.seq_nn_scale, 1e-12)
        eps = 1e-4 * scale
        # Медоид -- буквально элемент банка, и по одному ближайшему его зазор
        # вышел бы нулевым: нижний конец свипа схлопнулся бы, а с ним и ось.
        # Та же причина, по которой hole_ratio исключает концы отрезка.
        return torch.where(two[:, 0] > eps, two[:, 0], two[:, 1]) / scale


def denoiser_gap(runner, z: torch.Tensor, reference: torch.Tensor,
                 mask: torch.Tensor,
                 times: Sequence[float] = (0.1, 0.2, 0.4),
                 seed: int = 0) -> torch.Tensor:
    """Насколько цель НЕВОЗМОЖНА по мнению самой модели. [B,L,D] -> [B].

    ПОЧЕМУ ГЕОМЕТРИЮ ПРИШЛОСЬ ОСТАВИТЬ. Свип 26.10 опроверг оба зазора по
    расстояниям. Позиционный поехал вниз (1.110 -> 0.795). Зазор по
    последовательностям, написанный ему на замену, оказался немонотонным:
    beta 0 / .25 / .5 / .75 / 1 дали 0.625 / 0.184 / 0.368 / 0.552 / 0.657,
    и он НИ РАЗУ не превысил 1, то есть центроид формально ближе к данным,
    чем реальные последовательности друг к другу.

    Это не ошибка нормировки, а свойство размерности. Среднее N независимых
    точек лежит от каждой из них на расстоянии примерно 1/sqrt(2) типичного
    межточечного, а минимум по банку в 49152 измерениях почти равен типичному
    расстоянию -- расстояния концентрируются. Поэтому центроид обязан выйти
    БЛИЖЕ к данным, чем сами данные друг к другу, при любой нормировке: весь
    отрезок медоид->центроид (0.74 от nn) короче одного шага между соседями.
    Евклидово расстояние до ближайшего соседа не отличает возможный текст от
    невозможного, ни по позициям, ни по последовательностям.

    Спрашиваем тогда не геометрию, а МОДЕЛЬ -- она и есть оценка плотности
    данных. Шумим цель и смотрим, куда сеть тянет её обратно:
        pull(z) = || x_0(mu*z + std*eps, t) - z ||
    Для возможной последовательности сеть возвращает почти её же, pull мал.
    Для невозможной тянет прочь, к многообразию, и pull велик. Делим на то же
    у настоящей цели батча:
        gap = pull(z) / pull(trg)
    Величина безразмерна, равна 1 для реальных данных по построению, не
    зависит от масштаба латентов и от размерности, и измеряет именно то, что
    видит сэмплер. Шум eps берётся ОДИН И ТОТ ЖЕ для z и для trg -- парное
    сравнение, иначе разброс по шуму съедает разницу.
    """
    if z.shape != reference.shape:
        raise ValueError(
            f"цель {tuple(z.shape)} и образец {tuple(reference.shape)} "
            f"должны совпадать по форме: сравнение парное")
    gen = torch.Generator(device="cpu").manual_seed(seed)
    ratios = []
    for t_val in times:
        t = torch.full((z.shape[0],), float(t_val),
                       device=z.device, dtype=z.dtype)
        params = runner.dynamic.marginal_params(t)
        mu, std = params["mu"], params["std"]
        eps = torch.randn(z.shape, generator=gen, dtype=torch.float32)
        eps = eps.to(device=z.device, dtype=z.dtype)

        def pull(x):
            out = runner.calc_score(
                model=runner.ddp_score_estimator,
                x_t=mu * x + std * eps, t=t,
                attention_mask=None,
                x_0_self_cond=torch.zeros_like(x))
            return masked_norm(out["x_0"] - x, mask)

        num = pull(z)
        den = pull(reference).clamp_min(1e-12)
        ratios.append(num / den)
    return torch.stack(ratios).mean(dim=0)
