# Wikipedia: полный прогон экспериментов

Порядок действий от чистого кластера до финальных метрик. Все команды — из
корня репозитория. Стадии `run_wikipedia.sh` зависят друг от друга через
артефакты на диске: каждую следующую запускать после того, как отработала
предыдущая (что где появляется — написано ниже).

## Раскладка по дискам (Харизма: home 3 мес / scratch 1 мес, NVMe)

Принцип: на scratch — только то, что дешево скачать заново; всё, без чего
прогон встанет, — в home. Раскладку задает `hf_env.sh`, подключенный всеми
точками входа; настраивать руками ничего не нужно, но при старте каждый
скрипт печатает `HF_HOME=... HF_DATASETS_CACHE=...` — стоит сверить, что
прогрев и задания видят одни и те же пути.

- `HF_HOME=~/hf_cache` (**home**, ~13 GB): модели, токенизаторы и скрипты
  метрик. Scratch чистят ежемесячно, а без этих кэшей eval падает уже после
  дней обучения — поэтому они лежат там, где переживут чистку.
- `HF_DATASETS_CACHE=/scratch/$USER/hf_datasets` (**scratch**, ~30 GB): сырой
  кэш дампа wikipedia. Нужен только на стадии `data`, после нее удаляется.
- `<репо>/datasets/wikipedia` (**home**): нарезанный датасет, статистики
  энкодера, декодеры, классификаторы. Симлинк на scratch не делать — при
  чистке пришлось бы переобучать декодеры и классификаторы.
- `checkpoints/`, `generated_texts/`, `wandb/` (**home**): чекпоинты
  невосстановимы без повторного обучения. Место под них: у диффузии
  save_top_k=5, у gpt 2, плюс служебный `last.pth` для дозапуска — порядка
  15 GB на диффузию и 13 GB на gpt.
- Репу на кластер класть **гитом** (`git clone`), а не заливать с Windows:
  CRLF ломает `.sh`.
- Переопределение любого пути — обычный `export` перед запуском: заданные
  снаружи значения ничем не перетираются.
- home тоже не вечен (3 мес): финальные json из `generated_texts/` и лучшие
  чекпоинты по завершении забирать к себе.

## 0. Локально: закоммитить и запушить

Должны уехать: `run_wikipedia.sh`, `run_flags.sh`, `prefetch_offline.py`,
`disk_report.sh`, `diffusion_holder.py`, `RUNBOOK.md`.

## 1. Login-нода (интернет есть): место и кэши

```bash
git pull && chmod +x run_wikipedia.sh disk_report.sh
```

```bash
./disk_report.sh          # посмотреть, что реально ест место
```

Чистка (пока ничего не запущено): `rm -rf ~/.cache`, `conda clean --all`,
старые чекпоинты/`wandb/`/логи slurm — по результатам отчета.

```bash
python prefetch_offline.py    # ~13 GB: BERT, gpt2-large, gpt-neo, deberta, метрики
```

Без прогрева eval упадет на mauve/ppl/bert-score уже после дней обучения —
на compute-нодах интернета нет.

## 2. Смоук-прогон (рекомендуется, ~полдня)

Проверяет весь пайплайн до первых метрик, прежде чем жечь GPU-дни. Смоук-режим
(`SMOKE=1`) режет обучение до 200 шагов и 200 генерируемых текстов, а все
артефакты помечает суффиксом `-smoke`, поэтому боевые он не трогает и не
подхватывает.

```bash
NUM_TEXTS=200000 ./run_wikipedia.sh data
```

```bash
./run_wikipedia.sh stats        # дождаться datasets/wikipedia/statistics/*.pt
```

```bash
./smoke_test.sh decoder         # дождаться decoder-*-smoke.pth
```

```bash
./smoke_test.sh diffuseq        # ~30 мин
```

```bash
./smoke_test.sh gpt             # ~30 мин, ни от чего не зависит
```

В slurm-логе проверить: SOURCE/TARGET режутся по 64 токена, примеры генерации
не мусор, метрики посчитались, лосс падает.

**Обязательная уборка после смоука** — иначе полный прогон подхватит
смоук-датасет (у него нет суффикса, в отличие от остальных артефактов):

```bash
rm -rf datasets/wikipedia checkpoints/*-smoke datasets/wikipedia/*-smoke.pth generated_texts/*-smoke
```

## 3. Полный прогон

### 3.1 Данные (login-нода, интернет)

```bash
./run_wikipedia.sh data
```

Появится `datasets/wikipedia/{train,validation,test}`. Сырой кэш дампа больше
не нужен: `rm -rf ~/.cache/huggingface/datasets/wikimedia___wikipedia*`

### 3.2 Статистики энкодера

```bash
./run_wikipedia.sh stats
```

Готово, когда появились `datasets/wikipedia/statistics/encodings-bert-base-cased-{mean,std}.pt`.

### 3.3 Декодеры (2 задания параллельно)

```bash
./run_wikipedia.sh decoders
```

Готово: два `datasets/wikipedia/decoder-*.pth` (один с `-conditional` для
genie, один безусловный для остальных).

### 3.4 Диффузии и GPT (4 обучения параллельно)

```bash
./run_wikipedia.sh diffusion    # genie, diffuseq, unconditional
```

```bash
./run_wikipedia.sh gpt          # можно было запустить сразу после 3.1
```

Диффузии можно пускать и по одной — `./run_wikipedia.sh diffuseq` (то же для
`genie` и `unconditional`), когда не нужно занимать очередь всеми тремя сразу
или когда надо перезапустить только одну.

Это самая долгая стадия: у всех четырех подходов 150k оптимизаторных шагов при
эффективном батче 512, то есть одни и те же 76.8 млн увиденных примеров.
По замеру на 4xV100-32GB: gpt ~224 ч, diffuseq ~68 ч, genie ~52 ч,
unconditional ~37 ч. Лимит задания — 75 ч, поэтому gpt потребует 3 дозапуска:
просто отправить `./run_wikipedia.sh gpt` заново, обучение продолжится с
`checkpoints/<prefix>/last.pth` (он пишется на каждом checkpoint_freq
независимо от метрики). Прогресс и промежуточные метрики — в slurm-логах
каждые 12500 шагов (GPT — 2500). guidance отдельно не обучается: он
использует чекпоинт unconditional.

### 3.5 Классификаторы guidance (после unconditional-диффузии)

```bash
./run_wikipedia.sh classifiers
```

Три схемы (shuffled/augmented/combined), каждая пишет свой
`datasets/wikipedia/conditional-encoder-*-64x64-*.pth`. augmented и combined
загружают последний чекпоинт безусловной диффузии — поэтому только после 3.4.
Вариант с TIME_SCALE: `TIME_SCALE=1000 ./run_wikipedia.sh classifiers`
(тогда и eval запускать с тем же TIME_SCALE; несовпадение поймается ошибкой).

## 4. Тестирование

### 4.1 Одиночный eval всех подходов

```bash
./run_wikipedia.sh eval
```

Запускает: genie, diffuseq, unconditional, guidance×3 схемы (CG_SCALE=10.0 по
умолчанию, поменять: `CG_SCALE=5.0 ./run_wikipedia.sh eval`), gpt.
Метрики — в slurm-логах, тексты и json — в `generated_texts/<prefix>/`
(у guidance в имени файла схема классификатора и time_scale).

Условные режимы меряются по bleu/bert-score/rouge против референсного
продолжения, unconditional — по mauve/div/ppl.

### 4.2 Для статьи: статистическая оценка (std и 95% CI по 20 сидам)

```bash
DATASET=wikipedia ARCH_TYPE=genie sbatch eval_diffusion_stat.sh
```

```bash
DATASET=wikipedia ARCH_TYPE=diffuseq sbatch eval_diffusion_stat.sh
```

```bash
DATASET=wikipedia ARCH_TYPE=unconditional sbatch eval_diffusion_stat.sh
```

```bash
DATASET=wikipedia ARCH_TYPE=guidance AUG_SCHEME=shuffled sbatch eval_diffusion_stat.sh
```

(то же с `AUG_SCHEME=augmented` и `AUG_SCHEME=combined`; сила guidance —
через `CG_SCALE=...`)

```bash
DATASET=wikipedia ARCH_TYPE=gpt sbatch eval_gpt2_stat.sh
```

Агрегированные mean ± std и CI сохраняются в
`generated_texts/<prefix>/statistical_eval-*.json`.

## Если что-то упало

- `FileNotFoundError` на декодере/статистиках/классификаторе — не дождались
  предыдущей стадии, порядок в шапке `./run_wikipedia.sh`.
- Ошибка про time_scale при загрузке классификатора — TIME_SCALE на eval не
  совпадает с тем, с которым обучали (это защита, а не баг).
- Сетевые таймауты/`OfflineModeIsEnabled` — модель не попала в прогрев;
  дозапустить `python prefetch_offline.py` на ноде с интернетом.
