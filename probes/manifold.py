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
from typing import Optional

import torch

from probes.common import content_mask, encode_batch, log, probe_loader


class LatentBank:
    """Банк чистых энкодингов плюс локальное касательное в произвольной точке."""

    def __init__(self, vectors: torch.Tensor, knn: int = 32, dim: int = 16,
                 query_chunk: int = 1024, bank_chunk: int = 8192,
                 sequences: Optional[torch.Tensor] = None):
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
        chunks, seqs, seen, taken = [], [], 0, 0
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
            taken += v.shape[0]
            got_seq = sum(c.shape[0] for c in seqs)
            if taken >= size and got_seq >= min_sequences:
                break
        if not chunks:
            raise RuntimeError(
                "банк пуст: skip_examples больше размера сплита")
        bank = torch.cat(chunks)[:size].contiguous()
        sequences = torch.cat(seqs).contiguous()
        log(f"банк готов: позиции {tuple(bank.shape)}, "
            f"последовательности {tuple(sequences.shape)}")
        return cls(bank, sequences=sequences, **kw)

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
