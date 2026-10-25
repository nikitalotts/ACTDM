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
    Инициализация -- k-means++ с фиксированным сидом (см. комментарий ниже):
    детерминированная, то есть точки оси воспроизводятся между прогонами.
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


def content_position_mean(target: torch.Tensor,
                          content: torch.Tensor) -> torch.Tensor:
    """Позиционное среднее по СОДЕРЖАТЕЛЬНЫМ позициям батча. [B,L,D] -> [L,D].

    Это якорь «про эту позицию условие не говорит ничего»: маргинальное
    среднее латентов в ней. Усреднять надо только по тем примерам, у которых
    в этой позиции стоит настоящий токен: Encoder кладет в паддинг zero_emb, и
    общее среднее тянуло бы хвост последовательности к нулю тем сильнее, чем
    позиция дальше, то есть якорь нес бы информацию о ДЛИНЕ. Позиции, где
    содержательных примеров не нашлось вовсе, получают общее среднее.
    """
    return content_position_stats(target, content)[0]


def content_position_stats(target: torch.Tensor, content: torch.Tensor):
    """Позиционные среднее и СКО по содержательным позициям. -> ([L,D], [L,D]).

    ПОЧЕМУ ПОЗИЦИОННЫЕ, А НЕ ОБЩИЕ. Позиционная статистика энкодингов BERT
    сильно неоднородна: нулевая позиция это всегда [CLS], последняя
    содержательная соседствует с [SEP], пунктуация тяготеет к концу
    предложения. Общее среднее по батчу и позициям схлопывает эту структуру в
    одну точку, и якорь «нулевой информации» перестает быть средним
    продолжением -- он становится постоянной последовательностью, которая
    декодируется вырожденным повтором. Для dilute это портит именно НИЖНИЕ
    уровни (стадия пускает 1%, 2%, 5%), ради которых режим и вводился.

    Паддинг исключен по той же причине, что и в среднем: иначе статистика
    хвоста несла бы информацию о длине.
    """
    m = content.unsqueeze(-1).to(torch.float32)
    x = target.float() * m
    n = m.sum(dim=0)                                      # [L, 1]
    s = x.sum(dim=0)                                      # [L, D]
    total = m.sum().clamp_min(1.0)

    glob_mean = x.flatten(0, 1).sum(dim=0) / total        # [D]
    mean = torch.where(n > 0, s / n.clamp_min(1.0),
                       glob_mean.unsqueeze(0).expand_as(s))

    # Дисперсия тоже только по содержательным: (E[x^2] - E[x]^2), смещенная
    # оценка -- на позиции может оказаться один пример, и несмещенная там
    # делилась бы на ноль.
    s2 = (x * target.float() * m).sum(dim=0)              # [L, D]
    var = torch.where(n > 0, s2 / n.clamp_min(1.0) - mean ** 2,
                      torch.zeros_like(mean))
    glob_std = (x.flatten(0, 1).pow(2).sum(dim=0) / total
                - glob_mean ** 2).clamp_min(0).sqrt()     # [D]
    std = torch.where(n > 0, var.clamp_min(0).sqrt(),
                      glob_std.unsqueeze(0).expand_as(mean))

    return mean.to(target.dtype), std.to(target.dtype)


def degrade_target(mode: str, target: torch.Tensor, level: int,
                   bank=None, content: Optional[torch.Tensor] = None,
                   seed: int = 0, pos_mean: Optional[torch.Tensor] = None,
                   pos_std: Optional[torch.Tensor] = None,
                   topic_level: Optional[int] = None):
    """Вернуть (деградированная цель, маска позиций или None, описание).

    mode:
      none     -- цель как есть (оракул, верхняя точка оси)
      prefix   -- level = число открытых позиций
      quantize -- level = число центроидов (по позициям)
      seqquant -- level = число центроидов (по последовательностям); это и есть
                  «веди к теме»
      dilute   -- level = проценты: z = mean + (level/100) (target - mean)
      noisy    -- level = проценты std: z = target + (level/100) std eps
      rank     -- level = число главных компонент
    """
    if mode == "none":
        return target, None, "оракул (полная цель)"

    if mode == "prefix":
        if content is None:
            raise ValueError("prefix требует маску содержательных позиций")
        mask = prefix_mask(content, level)
        # ЦЕЛЬ ДЛЯ ОТОБРАЖЕНИЯ обязана быть деградированной, а не оракульной.
        # Раньше здесь возвращался target как есть, и steer = decode(z)
        # оказывался полным референсом: target_recovery читался 0.998 на ВСЕХ
        # уровнях k. Пока осью x был косинус, это была косметика. Теперь ось x
        # -- это и есть восстановимость цели, и prefix вырождался в
        # вертикальную прямую: четыре разных качества при одном x. А именно
        # prefix несет методический вывод «номинальные биты качество не
        # предсказывают» (prefix-32 дает bert 0.722 против 0.470 у quantize-32
        # при тех же ~320 битах), то есть точка без координаты x -- это вывод
        # без опоры.
        #
        # Открытые позиции берем из истины, остальные СОДЕРЖАТЕЛЬНЫЕ -- из
        # позиционного среднего: про них условие и правда не говорит ничего.
        # Спецтокены и паддинг не трогаем -- скелет последовательности на всех
        # уровнях одинаков, поэтому восстановимость меряет СОДЕРЖАНИЕ, а не
        # длину. Утечка длины при этом постоянна вдоль оси и тренд дать не
        # может.
        if pos_mean is None:
            pos_mean = content_position_mean(target, content)
        keep = (mask | ~content).unsqueeze(-1)
        z = torch.where(keep, target, pos_mean.unsqueeze(0).to(target.dtype))
        # На градиент подмена не влияет: corr домножается на ту же mask, а
        # внутри нее z совпадает с target побитово. Закреплено тестом
        # test_prefix_display_target_does_not_touch_the_gradient.
        return z, mask, f"первые {level} позиций"

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
        if bank.sequences is None:
            raise ValueError("для seqquant банк должен хранить последовательности")
        # Идёт через ту же кодовую книгу, что и cluster_bundle: кластеризация
        # по пулированным векторам, цель -- среднее ПОЛНЫХ последовательностей
        # кластера. Так seqquant и форма условия сравнимы между собой, и обе
        # свободны от вырождения k-means в 49152 измерениях.
        centroid, _, _ = cluster_bundle(bank, target, level, members=1)
        return centroid, None, f"{level} тем (среднее кластера)"

    if mode == "offmanifold":
        if bank is None or bank.sequences is None:
            raise ValueError("offmanifold требует банк с последовательностями")
        # level -- проценты: 0 это медоид (на многообразии), 100 -- центроид
        # кластера (вне его). Информация постоянна на всём свипе.
        b = level / 100.0
        topics = int(topic_level or 16)
        return (offmanifold_mix(bank, target, topics, b), None,
                f"медоид->центроид, beta={b:g} при {topics} темах")

    if mode == "dilute":
        # level -- проценты: 100 это оракул, 0 это среднее (нулевая информация)
        a = level / 100.0
        if pos_mean is None and content is not None:
            pos_mean = content_position_mean(target, content)
        return (dilute_target(target, a, mean=pos_mean), None,
                f"сжатие к среднему, alpha={a:g}")

    if mode == "noisy":
        # level -- проценты от собственного std данных
        f = level / 100.0
        if pos_std is None and content is not None:
            pos_std = content_position_stats(target, content)[1]
        return (noisy_target(target, f, seed=seed, std=pos_std), None,
                f"изотропный шум, {level}% std")

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
    if mode == "offmanifold":
        # Информация постоянна вдоль свипа -- в этом весь смысл режима.
        return math.log2(max(2, level if level > 1 else 2))
    if mode in ("dilute", "noisy", "rank"):
        # Для непрерывных осей номинальных бит нет, и это к лучшему: замеры
        # 24.10 показали, что номинальные биты качество НЕ предсказывают --
        # prefix-32 и quantize-32 обе дают ~320 бит, а bert 0.722 против 0.470.
        # Рабочая ось -- измеренная восстановимость цели (target_recovery).
        return None
    return None


# --------------------------------------------------------------------------
# Непрерывные оси информативности
# --------------------------------------------------------------------------
# Дискретные режимы выше дали монотонную картину, но КОСИНУС как ось x оказался
# непригоден (см. probe_information_axis.py, раздел «почему cos_to_true лжет»).
# Рабочая ось -- измеренная восстановимость самой цели (STEER vs TRG). Чтобы
# пройти по ней непрерывно и попасть в тот диапазон, где живет реальный
# классификатор (bert 0.5081 при поле 0.4605), нужны режимы с плавным уровнем.
#
# dilute -- сжатие цели к среднему: z = mean + a (target - mean). Это ровно та
#   форма, которую имеет условное среднее истинной цели при частичной
#   информации: байесовская оценка по b битам -- сжатая копия истины. Поэтому
#   dilute -- не произвольная порча, а каноническая форма «условие несет долю
#   информации».
# noisy -- изотропный шум: z = target + s * std * eps. Другая по характеру
#   порча при той же непрерывности. Если dilute и noisy лягут на одну кривую
#   «качество от восстановимости цели», значит важен ОБЪЕМ информации, а не
#   способ ее порчи. Это проверяемая форма закона, и в отличие от пары
#   prefix/quantize здесь обе оси непрерывны и перекрываются по диапазону.


def dilute_target(target: torch.Tensor, alpha: float,
                  mean: Optional[torch.Tensor] = None) -> torch.Tensor:
    """z = mean + alpha * (target - mean). alpha=1 -- оракул, alpha=0 -- среднее.

    mean ожидается ПОЗИЦИОННЫМ, [L, D]: якорь нулевой информации -- это среднее
    продолжение, а не постоянный вектор. Считать его надо через
    content_position_stats (исключает паддинг) и ОДИН раз на прогон, иначе
    якорь дрожит между батчами. Откат без него -- позиционное среднее по
    батчу: паддинг в него попадает, но структура позиций хотя бы сохраняется.
    """
    if mean is None:
        mean = target.float().mean(dim=0)                 # [L, D]
    return (mean + alpha * (target.float() - mean)).to(target.dtype)


def noisy_target(target: torch.Tensor, frac: float, seed: int = 0,
                 std: Optional[torch.Tensor] = None) -> torch.Tensor:
    """z = target + frac * std * eps, std -- ПОЗИЦИОННЫЙ масштаб самих данных.

    std ожидается [L, D] по той же причине, что и mean у dilute: разброс
    энкодингов на позиции [CLS] и в середине продолжения отличается на
    порядок, и один поканальный масштаб портил бы одни позиции и недошумлял
    другие. Откат без него -- позиционное СКО по батчу.
    """
    if std is None:
        std = target.float().std(dim=0)                   # [L, D]
    g = torch.Generator(device="cpu").manual_seed(seed)
    eps = torch.randn(target.shape, generator=g).to(target.device)
    return (target.float() + frac * std * eps).to(target.dtype)


# --------------------------------------------------------------------------
# Множество против точки при РАВНОЙ информации
# --------------------------------------------------------------------------
# Решающий эксперимент. Реальный классификатор несет один бит и дает +0.048 к
# полу; точка-цель сравнимой информативности (seqquant) дает -0.07, то есть
# ХУЖЕ отсутствия guidance. Значит дело не только в объеме информации, но и в
# ФОРМЕ условия: классификатор говорит «будь во множестве допустимых», а не
# «иди в эту точку».
#
# Чтобы отделить форму от объема, нужны три условия с ОДИНАКОВОЙ информацией
# (метка кластера, log2 K бит) и разной формой:
#   centroid -- среднее кластера: точка, которой нет на многообразии;
#   medoid   -- самый центральный РЕАЛЬНЫЙ элемент кластера: точка на
#               многообразии (выбран по центроиду, не по истине, -- иначе
#               утечка информации сверх метки кластера);
#   mixture  -- мягкий минимум по M центральным элементам: множество на
#               многообразии, модель сама выбирает, к какому идти.
# Разность centroid->medoid измеряет цену схода с многообразия, medoid->mixture
# -- выигрыш от множественной формы. Обучать ничего не нужно.

_CODEBOOK_CACHE = {}


def pool_sequences(seq: torch.Tensor) -> torch.Tensor:
    """Последовательности -> по вектору на каждую. [N,L,D] -> [N,D].

    ПОЧЕМУ НЕ СЫРЫЕ 49152 ИЗМЕРЕНИЯ. Прогон 25.10 показал, что k-means по
    развёрнутым последовательностям вырождается: уровни K=4, 16, 64 и 256 дали
    bert 0.3882 / 0.3943 / 0.3957 / 0.3973 -- рост числа тем в 64 раза не
    изменил ничего, три разных примера получили один и тот же медоид, а все
    центроиды декодировались в «s the the the the». Причина не в том, что тема
    не несёт информации, а в том, что 1024 точки в 49152 измерениях почти
    равноудалены, и кластеры схлопываются в один.

    Усреднение по позициям даёт 768 измерений и ровно ту величину, которая
    отвечает за тему: средняя семантика токенов абзаца. Порядок слов при этом
    теряется, и это правильно -- метка кластера и должна нести тему, а не
    синтаксис.
    """
    return seq.reshape(seq.shape[0], -1, seq.shape[-1]).float().mean(dim=1)


def sequence_codebook(bank, level: int):
    """Кодовая книга тем: (центры [K,D], приписки [N]) в ПУЛИРОВАННОМ пространстве.

    Кластеризация идёт по pool_sequences, а не по сырым последовательностям --
    см. его докстринг. Сами цели остаются полноразмерными: кластеризуем по
    теме, а ведём к последовательности.

    Без кеша k-means пересчитывался бы на каждом батче: тот же банк, тот же
    сид, тот же ответ -- но при level=512 это минуты на пустом месте.
    """
    seq = bank.sequences
    if seq is None:
        raise ValueError("нужен банк с последовательностями")
    key = (id(bank), "seq-pooled", level)
    if key not in _CODEBOOK_CACHE:
        pooled = pool_sequences(seq)
        centers = kmeans(pooled, level)
        assign = torch.cdist(pooled, centers).argmin(dim=1)
        _CODEBOOK_CACHE[key] = (centers, assign, pooled)
    return _CODEBOOK_CACHE[key]


def cluster_bundle(bank, target: torch.Tensor, level: int, members: int = 4):
    """Для каждого элемента батча -- (центроид, медоид, M кандидатов).

    Все три несут РОВНО метку кластера: медоид и кандидаты выбираются по
    близости к центроиду, истина при выборе не используется.
    Возвращает centroid [B,L,D], medoid [B,L,D], cand [B,M,L,D].
    """
    seq = bank.sequences
    centers, assign, pooled = sequence_codebook(bank, level)
    B, L, D = target.shape
    mine = torch.cdist(pool_sequences(target), centers).argmin(dim=1)   # [B]

    centroid = torch.empty(B, L, D, device=target.device, dtype=torch.float32)
    medoid = torch.empty(B, L, D, device=target.device, dtype=torch.float32)
    cand = torch.empty(B, members, L, D, device=target.device,
                       dtype=torch.float32)
    seqf = seq.float()

    for i in range(B):
        c = int(mine[i].item())
        idx = (assign == c).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            # Пустой кластер: кандидатов нет. Честнее отступить к глобальному
            # среднему, чем молча подменить кластер соседним.
            centroid[i] = seqf.mean(dim=0)
            medoid[i] = seqf.mean(dim=0)
            cand[i] = seqf.mean(dim=0).unsqueeze(0).expand(members, -1, -1)
            continue

        # ЦЕНТРОИД -- среднее ПОЛНЫХ последовательностей кластера. Это точка
        # ВНЕ многообразия по построению: усреднение реальных текстов текстом
        # не является. Раньше здесь стоял центр k-means из 49152-мерного
        # пространства, который вырождался в глобальное среднее и
        # декодировался в «the the the» независимо от темы.
        centroid[i] = seqf[idx].mean(dim=0)

        # МЕДОИД -- самая центральная РЕАЛЬНАЯ последовательность кластера.
        # Выбор по пулированному вектору, то есть по той же метке кластера;
        # истина при выборе не используется, утечки сверх метки нет.
        d = (pooled[idx] - centers[c]).pow(2).sum(dim=1)
        medoid_pos = int(d.argmin().item())
        medoid[i] = seqf[idx[medoid_pos]]

        # КАНДИДАТЫ -- РАЗНЕСЁННЫЕ по кластеру, а не ближайшие к его центру.
        # Прогон 25.10: четыре ближайших к центроиду почти совпадали, STEER у
        # medoid и mixture выходил дословно одинаковым, и «множество» было
        # неотличимо от точки. Берём жадную дальнюю выборку от медоида -- так
        # набор покрывает кластер, то есть действительно задаёт МНОЖЕСТВО
        # допустимых продолжений при той же метке.
        chosen = [medoid_pos]
        far = (pooled[idx] - pooled[idx][medoid_pos]).pow(2).sum(dim=1)
        while len(chosen) < members and len(chosen) < idx.numel():
            nxt = int(far.argmax().item())
            chosen.append(nxt)
            far = torch.minimum(
                far, (pooled[idx] - pooled[idx][nxt]).pow(2).sum(dim=1))
            far[torch.tensor(chosen, device=far.device)] = -1.0
        while len(chosen) < members:
            # Кластер меньше M: добираем повторами. Повтор не добавляет
            # информации и не смещает softmax -- он лишь делит вес кандидата.
            chosen.append(chosen[0])
        cand[i] = seqf[idx[torch.tensor(chosen, device=idx.device)]]

    return (centroid.to(target.dtype), medoid.to(target.dtype),
            cand.to(target.dtype))


def offmanifold_mix(bank, target: torch.Tensor, level: int,
                    beta: float):
    """Интерполяция медоид -> центроид при НЕИЗМЕННОЙ информации.

    РЕШАЮЩИЙ ЗАМЕР. Прогон 25.10 дал главный факт работы: при одной и той же
    метке кластера цель-центроид даёт bert 0.394, а реальный элемент того же
    кластера -- 0.496. Но это ОДНА пара точек, и законом она не является:
    рецензент справедливо спросит, наблюдение это или правило.

    Здесь beta непрерывно ведёт цель с многообразия в пустоту:
        z(beta) = medoid + beta * (centroid - medoid)
    beta=0 -- реальная последовательность кластера, beta=1 -- его среднее.
    Информация при этом ПОСТОЯННА на всём свипе: обе крайние точки, как и
    любая промежуточная, определяются ровно меткой кластера (log2 K бит) и
    ничем больше. Меняется только удалённость цели от данных, и её измеряет
    LatentBank.manifold_gap.

    Если качество падает монотонно по beta и предсказывается этим
    расстоянием -- «цель вне многообразия» становится измеримым законом, а не
    наблюдением из двух точек.
    """
    centroid, medoid, _ = cluster_bundle(bank, target, level, members=1)
    return medoid + beta * (centroid - medoid)
