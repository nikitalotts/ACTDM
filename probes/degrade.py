"""Управляемая деградация целевого латента: ось информативности условия.

ЗАЧЕМ
-----
Замеры 24.10 дали две крайние точки одной шкалы, измеренные на ОДНОЙ и той же
замороженной безусловной диффузии:

  * бинарный классификатор «корректное продолжение?» -- AUC 0.99, но косинус
    градиента с направлением к истинному продолжению 0.009; закрывает 52%
    семантического разрыва до cross-attention и 18% лексического;
  * точка-цель (оракул) -- косинус 1 по построению; bleu 0.938, bert 0.980.

Между ними пропасть, и объяснять ее нечем: две точки -- это не закон. Нужен
спектр условий с КОНТРОЛИРУЕМЫМ количеством информации о цели, измеренный тем
же косинусом и той же метрикой качества.

Модуль строит такой спектр, деградируя оракульную цель тремя способами. Ни один
из них не требует обучения -- это ключевое: ось получается целиком на инференсе.

  prefix   -- guidance только по первым k содержательным позициям продолжения,
              остальные свободны. Информация точная, но локальная.
  quantize -- цель заменяется ближайшим из K центроидов (k-means по банку
              чистых латентов). Информация приближенная, но по всем позициям.
              На уровне ПОСЛЕДОВАТЕЛЬНОСТИ это ровно «веди к теме c»: оракульный
              классификатор темы дал бы градиент в центроид своего кластера.
              Поэтому отдельный классификатор темы обучать не нужно.
  rank     -- цель проецируется на r главных компонент банка.

Две оси деградации намеренно разные по ХАРАКТЕРУ: prefix дает точную информацию
о немногих позициях, quantize -- грубую обо всех. Если обе ложатся на одну
кривую «качество от косинуса», значит качество определяется количеством
информации, а не тем, как она распределена. Это и есть проверяемая форма
закона.
"""
from typing import Optional

import torch


def kmeans(x: torch.Tensor, k: int, iters: int = 25, seed: int = 0,
           chunk: int = 8192) -> torch.Tensor:
    """Простой k-means на GPU. x: [N, D] -> центроиды [k, D].

    Своя реализация, а не sklearn: данные уже на карте, и гонять 8192x768 через
    CPU дороже, чем посчитать два десятка итераций Ллойда прямо здесь.
    Инициализация детерминированная (первые k точек случайной перестановки) --
    точки оси должны воспроизводиться между прогонами.
    """
    if k >= x.shape[0]:
        raise ValueError(f"центроидов ({k}) не меньше точек банка ({x.shape[0]})")
    g = torch.Generator(device="cpu").manual_seed(seed)

    # Инициализация k-means++, а не случайные k точек. На синтетике из трех
    # разнесенных кластеров случайная инициализация с сидом 0 поставила два
    # центроида в один кластер, и третий кластер слился с соседним -- то есть
    # уровень K=3 дал бы фактически две группы. Для оси информативности это
    # хуже, чем просто шум: уровни перестают соответствовать объему информации.
    first = int(torch.randint(x.shape[0], (1,), generator=g).item())
    centers = [x[first]]
    d2 = ((x - centers[0]) ** 2).sum(dim=1)
    for _ in range(1, k):
        probs = d2.clamp_min(0)
        total = probs.sum()
        if total <= 0:
            idx = int(torch.randint(x.shape[0], (1,), generator=g).item())
        else:
            idx = int(torch.multinomial(
                (probs / total).cpu(), 1, generator=g).item())
        centers.append(x[idx])
        d2 = torch.minimum(d2, ((x - centers[-1]) ** 2).sum(dim=1))
    centers = torch.stack(centers).clone()

    for _ in range(iters):
        assign = torch.empty(x.shape[0], dtype=torch.long, device=x.device)
        for s in range(0, x.shape[0], chunk):
            assign[s:s + chunk] = torch.cdist(x[s:s + chunk], centers).argmin(dim=1)
        new = torch.zeros_like(centers)
        counts = torch.zeros(k, device=x.device)
        new.index_add_(0, assign, x)
        counts.index_add_(0, assign, torch.ones_like(assign, dtype=x.dtype))
        # Пустой кластер оставляем на месте: обнулять его центроид нельзя,
        # иначе к нему притянется половина точек на следующей итерации.
        alive = counts > 0
        centers[alive] = new[alive] / counts[alive].unsqueeze(1)
    return centers


def prefix_mask(content: torch.Tensor, k: int) -> torch.Tensor:
    """Маска первых k СОДЕРЖАТЕЛЬНЫХ позиций. content: [B, L] -> [B, L] bool.

    Считаем по содержательным, а не по сырым позициям: нулевая позиция это
    [CLS], и prefix с k=1 по сырым позициям не дал бы ни одного настоящего
    токена -- точка оси оказалась бы ложно нулевой.
    """
    if k <= 0:
        return torch.zeros_like(content, dtype=torch.bool)
    order = content.cumsum(dim=1)
    return content & (order <= k)


def degrade_target(mode: str, target: torch.Tensor, level: int,
                   bank=None, content: Optional[torch.Tensor] = None):
    """Вернуть (деградированная цель, маска позиций или None, описание).

    mode:
      none     -- цель как есть (оракул, верхняя точка оси)
      prefix   -- level = число открытых позиций
      quantize -- level = число центроидов (по позициям)
      seqquant -- level = число центроидов (по последовательностям); это и есть
                  «веди к теме»
      rank     -- level = число главных компонент
    """
    if mode == "none":
        return target, None, "оракул (полная цель)"

    if mode == "prefix":
        if content is None:
            raise ValueError("prefix требует маску содержательных позиций")
        return target, prefix_mask(content, level), f"первые {level} позиций"

    if mode == "quantize":
        if bank is None:
            raise ValueError("quantize требует банк латентов")
        centers = kmeans(bank.bank, level)
        flat = target.reshape(-1, target.shape[-1]).float()
        idx = torch.cdist(flat, centers).argmin(dim=1)
        z = centers[idx].reshape(target.shape).to(target.dtype)
        return z, None, f"{level} центроидов по позициям"

    if mode == "seqquant":
        if bank is None:
            raise ValueError("seqquant требует банк латентов")
        seq = bank.sequences
        if seq is None:
            raise ValueError("для seqquant банк должен хранить последовательности")
        flat_bank = seq.reshape(seq.shape[0], -1).float()
        centers = kmeans(flat_bank, level)
        flat = target.reshape(target.shape[0], -1).float()
        idx = torch.cdist(flat, centers).argmin(dim=1)
        z = centers[idx].reshape(target.shape).to(target.dtype)
        return z, None, f"{level} тем (центроиды последовательностей)"

    if mode == "rank":
        if bank is None:
            raise ValueError("rank требует банк латентов")
        x = bank.bank.float()
        mu = x.mean(dim=0, keepdim=True)
        # Главные компоненты банка; level штук оставляем, остальное отбрасываем
        _, _, vh = torch.linalg.svd(x - mu, full_matrices=False)
        basis = vh[:level]                                   # [r, D]
        flat = target.reshape(-1, target.shape[-1]).float() - mu
        z = (flat @ basis.T) @ basis + mu
        return z.reshape(target.shape).to(target.dtype), None, f"{level} компонент"

    raise ValueError(f"неизвестный режим деградации: {mode}")


def estimated_bits(mode: str, level: int, seq_len: int) -> Optional[float]:
    """Грубая оценка числа бит о цели. Вторичная ось, помечается как оценка.

    Главная ось -- ИЗМЕРЕННЫЙ косинус градиента с направлением к истинному
    продолжению; эта оценка нужна только чтобы подписать точки по-человечески.
    Для prefix считаем ~10 бит на позицию (энтропия токена wordpiece на
    wikipedia того же порядка), для квантизации -- log2(K) на единицу.
    """
    import math
    if mode == "none":
        return None
    if mode == "prefix":
        return 10.0 * level
    if mode == "quantize":
        return seq_len * math.log2(level)
    if mode == "seqquant":
        return math.log2(level)
    if mode == "rank":
        return None
    return None
