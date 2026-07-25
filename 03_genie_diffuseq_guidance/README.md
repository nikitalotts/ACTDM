## Три подхода к условной генерации в одном проекте

Здесь собраны три способа связать промпт с непрерывной текстовой диффузией. Они отличаются только этим, поэтому реализованы в одном проекте и выбираются параметром `--architecture_type`:

| `--architecture_type` | Условие подается | Диффузия |
| --- | --- | --- |
| `genie` | через cross-attention в каждый блок denoising network | условная |
| `diffuseq` | через latent replacement: латенты промпта фиксируются на каждом шаге обратного процесса | условная |
| `guidance` | через градиент внешнего бинарного классификатора на этапе генерации | **безусловная** |
| `unconditional` | никак, промпт не используется | безусловная |

`genie` адаптирован из метода Genie, `diffuseq` -- из DiffuSeq, `guidance` -- classifier guidance поверх безусловной модели. Режим `unconditional` нужен, чтобы мерить безусловную диффузию саму по себе; `guidance` -- это ровно она же плюс классификатор на генерации.

Во всех скриптах архитектура задается переменной окружения `ARCH_TYPE` (см. [run_flags.sh](./run_flags.sh)), для `guidance` силу задает `CG_SCALE`:
```
ARCH_TYPE=diffuseq bash train_diffusion.sh
ARCH_TYPE=guidance CG_SCALE=10.0 bash eval_diffusion.sh
```

Что важно знать про чекпоинты:

- `guidance` и `unconditional` используют **один и тот же чекпоинт** диффузии -- это одна модель, guidance добавляется только на генерации. К префиксу обоих добавляется `-uncond`.
- У `diffuseq` префикс получает `-diffuseq`, у `genie` суффикса нет.
- Декодер у `genie` условный (`-conditional` в имени), у остальных -- безусловный.

## Данные и пространство латентов

Еще две настройки задаются там же и обязаны совпадать у декодера, диффузии и классификатора:

| Переменная | Значения | Что задает |
| --- | --- | --- |
| `SPLIT_SCHEME` | `half` (по умолчанию), `last_sentence`, `sliding` | Как история rocstories режется на промпт и продолжение |
| `NORMALIZE` | `1` (по умолчанию), `0` | Нормализовать ли энкодинги статистиками датасета (`EncNormalizer`) |

Схемы разбиения (единственное место, где они заданы -- `split_story` в [data/load.py](./data/load.py)):

- `half` -- 1,2,3 -> 4,5, одна пара на историю;
- `last_sentence` -- предложения 1-4 -> предложение 5, одна пара на историю;
- `sliding` -- 1->2,3 | 1,2->3,4 | 1,2,3->4,5, три пары на историю.

Датасет надо скачать с той же схемой, с которой потом запускается обучение:
```
python -m data.load --dataset_name=rocstories --split_scheme=sliding
```

Нестандартная схема и `NORMALIZE=0` дают суффиксы (`-sliding`, `-unnorm`) в именах декодера, классификатора и чекпоинтов -- значения по умолчанию оставляют имена прежними.

## Зависимости

Список используемых библиотек приведен в [requirements.txt](./requirements.txt). Создать и активировать окружение можно с помощью Miniconda:
- `conda create --name actdm python=3.9`
- `conda activate actdm`
- `conda install pip`
- `pip install -r requirements.txt`
- `python -m spacy download en`

## Загрузка датасета

```
python -m data.load --dataset_name=rocstories
```

## Подсчет статистик

Для корректного обучения диффузионной модели необходимо предварительно посчитать статистики:
```
bash make_statistics.sh
```

## Обучение декодера

Необходимо обучить декодер, восстанавливающий токены по энкодингам:
```
ARCH_TYPE=genie         bash train_decoder.sh   # условный декодер
ARCH_TYPE=unconditional bash train_decoder.sh   # для diffuseq / guidance / unconditional
```

## Обучение диффузионной модели

После подсчета статистик и обучения декодера запустите обучение:
```
ARCH_TYPE=genie         bash train_diffusion.sh
ARCH_TYPE=diffuseq      bash train_diffusion.sh
ARCH_TYPE=unconditional bash train_diffusion.sh   # общий чекпоинт для unconditional и guidance
```

## Обучение классификатора (conditional encoder)

Нужен только для `ARCH_TYPE=guidance`. Обучается поверх готовой безусловной диффузии.
Три варианта генерации негативных примеров, схема входит в имя файла классификатора:
```
bash train_conditional_encoder_shuffled.sh    # негативы: перемешанные пары
bash train_conditional_encoder_augmented.sh   # негативы: продолжения, прогнанные через диффузию
bash train_conditional_encoder_combined.sh    # объединение двух схем выше
```

## Оценка модели

Для генерации продолжений и подсчета метрик используйте:
```
ARCH_TYPE=genie                  bash eval_diffusion.sh
ARCH_TYPE=diffuseq               bash eval_diffusion.sh
ARCH_TYPE=guidance CG_SCALE=10.0 bash eval_diffusion.sh
ARCH_TYPE=unconditional          bash eval_diffusion.sh

ARCH_TYPE=guidance CG_SCALE=10.0 bash eval_diffusion_stat.sh   # усреднение метрик по сидам
```

В условных режимах и в `guidance` считаются `bleu/bert-score/rouge`, в `unconditional` -- `mauve/div/ppl`.
