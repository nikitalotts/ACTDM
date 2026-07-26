# Адаптация непрерывных текстовых диффузионных моделей для генерации текста по промпту

Репозиторий выпускной квалификационной работы, посвященной задаче генерации текста по промпту на основе непрерывных диффузионных моделей. В работе адаптируются и сравниваются три способа введения условия в диффузионный процесс — classifier guidance, cross-attention и latent replacement, — а в качестве авторегрессионного baseline используется GPT-2-medium. Эксперименты проводятся на ROCStories (пилотные) и на английской Википедии (large-scale).

Исходно каждый подход жил в отдельной папке со своей копией кода. Подходы отличаются только способом подачи условия, поэтому теперь они собраны в один проект и выбираются параметром `--architecture_type`.

## Режимы работы

| `--architecture_type` | Как подается условие | Модель | Раздел работы |
| --- | --- | --- | --- |
| `genie` | cross-attention в каждый блок denoising network | условная диффузия | 5.2.3 |
| `diffuseq` | latent replacement: латенты промпта фиксируются на каждом шаге обратного процесса | условная диффузия | 5.2.4 |
| `guidance` | градиент внешнего бинарного классификатора на этапе генерации | **безусловная** диффузия | 5.2.2 |
| `unconditional` | никак, промпт не используется | безусловная диффузия | — |
| `gpt` | промпт как префикс последовательности | GPT-2-medium с нуля | 5.2.1 |

`genie` адаптирован из метода Genie, `diffuseq` — из DiffuSeq, за основу диффузионной части взята реализация [TEncDM](https://github.com/M0RJIQUE/tencdm).

Режим `unconditional` нужен, чтобы измерить безусловную диффузию саму по себе; `guidance` — это ровно та же обученная модель плюс классификатор на генерации, отдельно ее переобучать не нужно.

`gpt` — авторегрессионный baseline: у него нет ни диффузии, ни декодера, ни энкодера латентов, поэтому у него отдельные entrypoint-ы (`train_gpt2.py`, `eval_gpt2.py`) и своя ветка в `create_config.py`.

## Быстрый старт

```
conda create --name actdm python=3.9
conda activate actdm
conda install pip
pip install -r requirements.txt
python -m spacy download en
```

Режим задается переменной окружения `ARCH_TYPE`, ее разбирает [`run_flags.sh`](./run_flags.sh), который подключен во все `.sh`-скрипты:

```
ARCH_TYPE=diffuseq bash train_diffusion.sh
DATASET=wikipedia ARCH_TYPE=diffuseq bash train_diffusion.sh
ARCH_TYPE=guidance CG_SCALE=10.0 bash eval_diffusion.sh
```

## Настройки данных и латентов

Задаются теми же переменными окружения и **обязаны совпадать** у декодера, диффузии и классификатора: они определяют и нарезку данных, и пространство латентов.

| Переменная | Значения | Что задает |
| --- | --- | --- |
| `ARCH_TYPE` | `genie`, `diffuseq`, `guidance`, `unconditional`, `gpt` | Режим работы |
| `CG_SCALE` | число > 0 (по умолчанию `10.0`) | Сила classifier guidance, только для `ARCH_TYPE=guidance` |
| `DATASET` | `rocstories` (по умолчанию), `wikipedia` | На чем учимся |
| `SPLIT_SCHEME` | см. ниже | Как текст режется на промпт и продолжение |
| `NORMALIZE` | `1` (по умолчанию), `0` | Нормализовать ли энкодинги статистиками датасета. К `gpt` неприменимо |

Схемы разбиения зависят от датасета; несовместимую пару конфиг отвергает.

**ROCStories** — по предложениям, разбиение делается при скачивании (`split_story` в [`data/load.py`](./data/load.py)), промпт 45 / таргет 35 токенов:

- `half` (по умолчанию) — предложения 1,2,3 → 4,5;
- `last_sentence` — предложения 1-4 → 5;
- `sliding` — 1→2,3 | 1,2→3,4 | 1,2,3→4,5, три пары на историю.

**Wikipedia** — по токенам, разбиение делается на лету загрузчиком (`batch_preprocessing_cond` в [`data/dataset_wiki.py`](./data/dataset_wiki.py)), 128 токенов суммарно → промпт 64 / таргет 64:

- `prefix_lm` (по умолчанию) — **ровно 50%** токенов в промпт, остальное в продолжение. Схема prefix LM из TESS-2. Граница фиксирована, поэтому позиционная статистика таргета не размазывается, а у `diffuseq` стык промпта и продолжения в конкатенированной последовательности всегда стоит на позиции 64 — только так три способа подачи условия сравниваются на одинаковой геометрии входа;
- `random_prefix` — длина промпта случайна (прежнее поведение). Оставлено, чтобы можно было воспроизвести сравнение схем: на этом коде случайная граница уже давала качество хуже фиксированной.

Для ROCStories датасет надо скачать с той же схемой, с которой потом запускается обучение. Для Wikipedia схема применяется во время обучения, перекачивать датасет при ее смене не нужно.

## Порядок запуска

### 1. Датасет

```
python -m data.load --dataset_name=rocstories
python -m data.load --dataset_name=wikipedia
```
ROCStories: для нестандартной нарезки добавьте `--split_scheme=sliding`. Повторный запуск с другой схемой перезаписывает каталог `datasets/rocstories`, поэтому для нескольких нарезок одновременно используйте разный `--dataset_path`.

Wikipedia: берется стандартный дамп [`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia) конфига `20231101.en`, статьи режутся на абзацы, короткие отбрасываются. Полезные флаги: `--min_tokens=128` (минимальная длина абзаца в словах), `--num_texts=200000` (ограничить объем для пробного прогона). Разбиение на промпт/продолжение здесь не фиксируется — оно делается на лету по `SPLIT_SCHEME`.

### 2. Статистики энкодера

Нужны всем диффузионным режимам (для `gpt` — пропустить):
```
bash make_statistics.sh
```

### 3. Декодер

Восстанавливает токены по энкодингам. У `genie` декодер условный, у остальных — безусловный, это разные файлы:
```
ARCH_TYPE=genie         bash train_decoder.sh
ARCH_TYPE=unconditional bash train_decoder.sh   # подходит для diffuseq / guidance / unconditional
```

### 4. Диффузионная модель

```
ARCH_TYPE=genie         bash train_diffusion.sh
ARCH_TYPE=diffuseq      bash train_diffusion.sh
ARCH_TYPE=unconditional bash train_diffusion.sh   # общий чекпоинт для unconditional и guidance
```

### 5. Классификатор (только для `guidance`)

Обучается поверх готовой безусловной диффузии. Три схемы генерации негативных примеров, схема входит в имя файла классификатора:
```
bash train_conditional_encoder_shuffled.sh    # негативы: перемешанные пары промпт/продолжение
bash train_conditional_encoder_augmented.sh   # негативы: продолжения, прогнанные через диффузию
bash train_conditional_encoder_combined.sh    # объединение двух схем выше
```

### 6. Генерация и метрики

```
ARCH_TYPE=genie                  bash eval_diffusion.sh
ARCH_TYPE=diffuseq               bash eval_diffusion.sh
ARCH_TYPE=guidance CG_SCALE=10.0 bash eval_diffusion.sh
ARCH_TYPE=unconditional          bash eval_diffusion.sh

ARCH_TYPE=guidance CG_SCALE=10.0 bash eval_diffusion_stat.sh   # усреднение метрик по нескольким сидам
```

В условных режимах, в `guidance` и в `gpt` считаются `bleu / bert-score / rouge`, в `unconditional` — `mauve / div / ppl`. Это верно для обоих датасетов.

### Авторегрессионный baseline

`gpt` не использует ни статистики, ни декодер, ни классификатор — достаточно скачанного датасета:
```
bash train_gpt2.sh

bash eval_gpt2.sh        # одиночный прогон
bash eval_gpt2_stat.sh   # усреднение метрик по сидам
```
Параметры декодирования (`--decoding greedy|sampling`, `--temperature`, `--top_p`, `--top_k`) относятся только к этому режиму. При `--decoding greedy` разные сиды дают одинаковый результат, поэтому усреднение по сидам осмысленно только с `sampling`.

## Имена артефактов

Чекпоинты, декодеры и классификаторы разных режимов не должны попадать в один файл, поэтому имена несут суффиксы:

- `genie` — без суффикса; `diffuseq` — `-diffuseq`; `guidance` и `unconditional` — общий `-uncond` (это одна и та же модель);
- `gpt` — отдельный префикс `gpt2-medium-...`;
- декодер `genie` дополнительно помечается `-conditional`;
- классификатор несет имя схемы негативов (`-shuffled`, `-augmented`, `-combined`);
- нестандартная `SPLIT_SCHEME` и `NORMALIZE=0` добавляют суффикс (`-sliding`, `-random_prefix`, `-unnorm`); схема по умолчанию для датасета суффикса не дает;
- имя датасета уже входит в префикс, так что артефакты ROCStories и Wikipedia не пересекаются.

Значения по умолчанию дают пустой суффикс, поэтому старые имена остаются валидными.

Все пути вычисляются из конфига — захардкоженных путей в коде нет. Сохранение и загрузка используют одну и ту же формулу:

- чекпоинты: `checkpoints/<checkpoints_prefix>/<step>.pth`, при пустом `checkpoint_name` берется последний по номеру шага;
- декодер: `datasets/<dataset>/<decoder.name>.pth`;
- классификатор: `datasets/<dataset>/<cond_encoder.name>.pth`;
- сгенерированные тексты: `generated_texts/<checkpoints_prefix>/`.

Если артефакт для выбранного режима еще не обучен, запуск падает с сообщением, в каком режиме его нужно обучить.

## Структура

```
create_config.py           конфиг, единая точка выбора режима по architecture_type
run_flags.sh               ARCH_TYPE / CG_SCALE / SPLIT_SCHEME / NORMALIZE -> флаги CLI
diffusion_holder.py        обучение и генерация для genie / diffuseq / guidance / unconditional
gpt2_holder.py             обучение и генерация для gpt
train_*.py, eval_*.py      entrypoint-ы; *.sh -- обертки под SLURM

data/       загрузка и препроцессинг: ROCStories (split_story в load.py),
            Wikipedia (batch_preprocessing_cond в dataset_wiki.py)
model/      энкодер, декодер, score estimator, нормализатор, классификатор для guidance
utils/      парсер аргументов, названия схем (schemes.py), EMA, вспомогательные функции
diffusion_utils/  динамика SDE, шедулеры, солверы
estimation_utils/ метрики (bleu, bert-score, rouge, mauve, div, ppl)
```

## Результаты экспериментов

Логи обучения и оценки лежат в `logs/`, разделенные на `training/` и `evaluation/`. По ним можно проверить значения автоматических метрик, приведенные в тексте работы.
