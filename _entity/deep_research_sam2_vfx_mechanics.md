# DEEP RESEARCH: SAM2 в VFX-студиях — полная механика

> Дата: 2026-04-03
> Метод: WebSearch + WebFetch + GitHub API (OSINT)

---

## 1. IMAGE ENGINE — DigiPro 2025 Paper (ПОЛНЫЙ РАЗБОР)

**Paper:** "Automated Video Segmentation Machine Learning Pipeline"
**Авторы:** Johannes Merz, Lucien Fostier — **Image Engine Design Inc.**, Vancouver
**Конференция:** DigiPro 2025 (Digital Production Symposium), August 2025
**arXiv:** [2507.07242](https://arxiv.org/abs/2507.07242) | [HTML](https://arxiv.org/html/2507.07242v1)
**ACM:** [10.1145/3744199.3744635](https://dl.acm.org/doi/10.1145/3744199.3744635)

### 1.1 Три стадии pipeline

```
Stage 1: GroundingDINO (detection) → bounding boxes + class labels
Stage 2: SAM2 (segmentation) → per-frame instance masks
Stage 3: SAM2 Video Predictor (tracking) → temporal consistency
```

Стадии запускаются **ПОСЛЕДОВАТЕЛЬНО** (не параллельно) — чтобы уместиться в 16 GB VRAM.

### 1.2 GroundingDINO параметры

- Модель: GroundingDINO (Liu et al., 2024)
- **box_threshold, text_threshold: НЕ УКАЗАНЫ в paper** (скорее всего дефолтные)
- Промпт по умолчанию (автоматический режим): `"person"`
- Примеры кастомных промптов: `"person, car"`, `"eye, hair, necklace, clothing, mouth"`
- Детекция запускается на КАЖДОМ кадре (не через интервалы)

### 1.3 SAM2 параметры

- **Конкретная модель (hiera_l/b+): НЕ УКАЗАНА**
- SAM2 Video Predictor с ограничениями:
  - **Max resolution: 1024px по длинной стороне** (масштабирование с сохранением aspect ratio)
  - Причина: GPU memory constraints видео-трекинга
- Tracking window: `sT = 20` кадров (длина окна трекинга)
- Prompt refresh step: `sP = 5` кадров
- Эти значения — результат экспериментов, "ideal balance between performance and quality"

### 1.4 Asymmetric Overlap Metric

**Формула:**
```
Sim(A, B) = |A ∩ B| / |A|
```

**Почему асимметричная, а не IoU:**
- IoU = |A∩B| / |A∪B| — симметричная, не показывает направление
- Sim(A,B) отвечает на вопрос: "Какая доля маски A содержится в маске B?"
- Это позволяет определить **вложенность** (enclosure relationships)
- Пример: маска руки целиком внутри маски тела → Sim(рука, тело) = 1.0, но Sim(тело, рука) = 0.1
- Критически важно для merge и false positive removal

### 1.5 Post-processing pipeline

**Шаг 1 — False Positive Removal:**
- Если маска перекрывается с **≥2 другими масками** при threshold **≥0.1** → УДАЛИТЬ
- Логика: реальные объекты обычно перекрываются с 0-1 другими, шум перекрывается со многими

**Шаг 2 — Merge:**
- Если маска имеет **единственного партнёра** с overlap **≥0.6** → ОБЪЕДИНИТЬ
- Высокий порог 0.6 гарантирует, что мержатся только части одного объекта
- Merge выполняется последовательно (sequential, не parallel)

### 1.6 ObjectId EXR Format

- **Отличие от Cryptomatte:** ObjectId поддерживает ПРОИЗВОЛЬНОЕ количество samples per pixel (vs hash-based Cryptomatte)
- **Кодировка:** Каждый пиксель хранит numeric ID + coverage (alpha)
- **Manifest:** В метаданных файла хранится маппинг numeric ID → human-readable name
- **Фильтрация:** Пользователь задаёт фильтр по identifiers → финальная маска = накопленный alpha
- **Историческая справка:** ObjectId (2012) предшествует Cryptomatte (SIGGRAPH 2015)
- Конкретная структура каналов EXR в paper НЕ описана

### 1.7 Hardware и деплоймент

- **GPU:** NVIDIA A4000 16GB
- **Контейнеризация:** Singularity/Apptainer (НЕ Docker)
  - Причина: не требует superuser на render farm
  - Лучшая поддержка GPU
- ML-инженеры самостоятельно собирают container images
- Быстрый update моделей через новые контейнеры

### 1.8 Скорость

| Метрика | Значение |
|---------|----------|
| Средний throughput | **~10 frames/min** (все 3 стадии суммарно) |
| 100-frame shot | ~10 минут |
| 283 frames | 23:26 мин |
| 173 frames | 11:40 мин |
| Самое тяжёлое | Стадии 2 (segmentation) и 3 (tracking) |

### 1.9 Продакшн-статистика

- **12 шоу** (2024-2025), конкретные названия НЕ указаны
- **1241 шот** обработано
- Downstream teams потребовали ручную правку только в **10-15% случаев**

### 1.10 Взаимодействие с артистами

1. **Automatic mode:** Только текстовый промпт (через preview browser tool)
2. **Interactive mode:** Browser-app с positive/negative point clicks → SAM2 propagation
3. **Manual fallback:** Ручные маски из традиционных VFX tools → подача в tracking систему
4. Preview browser показывает результаты detection + segmentation до запуска полного pipeline

### 1.11 Ограничения

- **Layer dropout:** Post-processing может удалить хорошие маски ради стабильности остальных
- **Далёкие объекты:** Мелкие люди на фоне не сегментируются из-за overlap interference
- **Max 1024px:** GPU memory SAM2 Video Predictor не позволяет больше
- **Ambiguous bounding boxes:** Очень большие или мелкие объекты → проблемы
- **Нет real-time:** 10 fps/min — это batch processing

---

## 2. ДРУГИЕ СТУДИИ И КОМПАНИИ

### 2.1 MARZ (Monsters Aliens Robots Zombies)

- **Тип:** Полностью AI-enabled VFX студия
- **Продукт:** AVX (Automated Visual Effects) платформа
- **Roto:** Полностью автоматизированный pipeline, **30-40% дешевле** традиционных студий
- **Vanity AI:** End-to-end aging/de-aging, **300x быстрее** manual pipeline
- **Infra:** NVIDIA GPUs + AWS + Weights & Biases для мониторинга
- **Статус:** Активно используется в production (Hollywood)
- **Детали архитектуры:** НЕ публичные (proprietary)
- [MARZ + W&B case study](https://wandb.ai/site/customers/marz/)

### 2.2 Batch (batch.film)

- **Тип:** Сервис automated rotoscoping for film/TV
- **Основатели:** Zak Mulligan, Seth Ricart (colorists)
- **Технология:** "Patented machine learning technology" (детали закрыты)
- **Выход:** Multi-channel EXR или ProRes файлы
- **Два качества:** MATTE COLOR (для цветокоррекции) и MATTE VFX (для compositing)
- **Скорость:** "Rotoscoping entire film takes hours, not weeks"
- **Категории:** people, objects, depth maps
- [batch.film](https://www.batch.film/)

### 2.3 Slapshot

- **Тип:** Cloud SaaS, AI VFX toolkit
- **Roto:** Click-based selection → cloud GPU processing → mattes
- **Autopilot:** Полностью автоматический режим, выход = organized Cryptomatte
- **Модели:** НЕ раскрыты (вероятно SAM2-based)
- **Выходные форматы:** EXR 16-bit, ProRes 4444, JPG/PNG sequences
- **Разрешение:** до 8K
- **Цены:** от $9/мес (250 frames, 2K) до Enterprise
- **Скорость:** ~8 минут на демо-шот
- **Ограничения SAM2 (по тестам Electric Sheep):** мелкие детали (пальцы), gaps между overlapping mattes
- [slapshot.ai](https://slapshot.ai/)

### 2.4 Weta FX

- **AI в pipeline:** Automated rotoscoping, AI match-moving, neural rendering
- **ML segmentation + depth:** Используют для weather/character insertion (lidar-based remapping)
- **Neural face rendering:** Модель одновременно рендерит лицо И сегментирует — композеры не ротоскопят вручную
- **AWS партнёрство (Nov 2025):** Разработка AI tools для VFX artists
- **Результаты:** 25-35% сокращение artist hours на технических задачах
- [Weta FX + AWS announcement](https://www.awn.com/news/w-t-fx-and-aws-develop-ai-tools-vfx-artists)

### 2.5 Framestore

- **Boris FX Silhouette** — основной roto tool
- **ML R&D:** Партнёрство с Weightshift (ускорение animation)
- **ASWF:** Участие в Dailies Notes Assistant (ML Working Group) с ILM, Sony Imageworks, DreamWorks, Autodesk
- **Публичный автоматический roto pipeline:** НЕ найден
- [Framestore ML article](https://www.framestore.com/news/machine-learning-vfx-voice)

### 2.6 SmartROTO (Foundry + DNEG + University of Bath)

- **Период:** Начат в 2019
- **Подход:** Artist-assisted ML (НЕ fully automatic)
- **Workflow:** Артист создаёт shapes + несколько keyframes → ML предсказывает intermediate keyframes
- **Цель:** Сэкономить ~25% времени ротоскопинга
- **Архитектура:** PyTorch-based tracking + shape consistency model
- **Роли:**
  - Foundry: Leadership, PyTorch integration
  - DNEG: Dataset (650,000+ shapes, 125 million keyframes), artists, UX feedback
  - University of Bath: CV/ML research, privacy-preserving data sharing
- **Статус:** ИССЛЕДОВАТЕЛЬСКИЙ ПРОЕКТ, НЕ продуктизирован
  - Ben Kent (DNEG): "Results weren't robust enough for an artist to rely on"
  - "Productization is still a long way off"
- **Ключевой вывод:** "Rotoscoping is extremely hard — people are going to be involved for the foreseeable future"
- [Foundry article](https://www.foundry.com/insights/machine-learning/smartroto-enabling-rotoscoping)

### 2.7 RotoShop (OTOY) — SIGGRAPH Asia 2025

- **Автор:** Sirak Ghebremusse (OTOY)
- **Суть:** Vectorization of segmentation masks → splines для Nuke
- **Вход:** Последовательность масок от CV-модели (любой, включая SAM2)
- **Выход:** Temporally consistent splines (Nuke-compatible)
- **Скорость:** 1000 frames за 20 минут
- **Storage:** 30 MB для 100K frames (vs 15 GB в PNG) — **500x compression**
- **Feedback от artists:** "Excellent starting point, only fine-tuning needed"
- [ACM paper](https://doi.org/10.1145/3757376.3771382)

### 2.8 Disney Soft Segmentation + CopyCat (Nuke)

- **Основа:** Disney Research Zurich 2017 — "Unmixing-Based Soft Color Segmentation"
- **Проблема оригинала:** ~1 час на 4K frame (слишком медленно для production)
- **Решение Rafael Silva:** CopyCat node в Nuke
  - Solve несколько frames оригинальным алгоритмом
  - Тренировать CopyCat на результатах
  - Inference = несколько секунд per frame
- **Результат:** Soft color segments (аналог layers с alpha) для live action
- **Подход:** Decomposition image → soft color layers (НЕ instance segmentation)
- [fxguide article](https://www.fxguide.com/fxfeatured/live-action-avos-disney-soft-segmentation-via-copycat-in-nuke/)

---

## 3. SAM2 AutomaticMaskGenerator — ПОЛНЫЕ ПАРАМЕТРЫ

Источник: [sam2/automatic_mask_generator.py](https://github.com/facebookresearch/sam2/blob/main/sam2/automatic_mask_generator.py) (GitHub API, дата: 2026-04-03)

### 3.1 Все параметры `SAM2AutomaticMaskGenerator.__init__()`

| Параметр | Тип | Default | Описание |
|----------|-----|---------|----------|
| `points_per_side` | int/None | **32** | Сетка точек NxN по изображению. Total = N². None если задан point_grids |
| `points_per_batch` | int | **64** | Точек за одну GPU-итерацию. Больше = быстрее, больше VRAM |
| `pred_iou_thresh` | float | **0.8** | Фильтр по predicted mask quality [0,1] |
| `stability_score_thresh` | float | **0.95** | Фильтр по стабильности маски при изменении binarization cutoff |
| `stability_score_offset` | float | **1.0** | Сдвиг cutoff для stability score |
| `mask_threshold` | float | **0.0** | Порог бинаризации mask logits |
| `box_nms_thresh` | float | **0.7** | IoU cutoff для NMS дедупликации |
| `crop_n_layers` | int | **0** | Доп. проходы на кропах. Layer i = 2^i кропов. 0 = без кропов |
| `crop_nms_thresh` | float | **0.7** | NMS между кропами |
| `crop_overlap_ratio` | float | **512/1500 ≈ 0.341** | Перекрытие кропов |
| `crop_n_points_downscale_factor` | int | **1** | Уменьшение points_per_side в кропах: N / factor^layer |
| `min_mask_region_area` | int | **0** | Минимальная площадь маски (OpenCV post-processing) |
| `output_mode` | str | **"binary_mask"** | `binary_mask` / `uncompressed_rle` / `coco_rle` |
| `use_m2m` | bool | **False** | Mask-to-mask refinement (доп. проход) |
| `multimask_output` | bool | **True** | Несколько масок на каждую точку сетки |

### 3.2 Рекомендации для VFX quality

**Для максимального качества краёв:**
```python
SAM2AutomaticMaskGenerator(
    model=sam2_model,
    points_per_side=64,        # 64x64 = 4096 точек (default 32x32=1024)
    pred_iou_thresh=0.88,      # выше default 0.8 → меньше мусора
    stability_score_thresh=0.95, # default, хорошо работает
    crop_n_layers=1,           # +1 уровень кропов для мелких деталей
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,  # убрать микро-фрагменты
    use_m2m=True,              # refinement pass
    multimask_output=True,
)
```

**Для баланса speed/quality:**
```python
SAM2AutomaticMaskGenerator(
    model=sam2_model,
    points_per_side=32,         # default
    pred_iou_thresh=0.88,
    stability_score_thresh=0.95,
    min_mask_region_area=50,
    use_m2m=False,
)
```

**Проблемные зоны SAM2 для VFX:**
- Пальцы, волосы → gaps и потеря деталей
- Overlapping mattes → пробелы между масками
- Motion blur → нестабильные края
- Мелкие/далёкие объекты → miss detection

### 3.3 Выходной формат каждой маски

```python
{
    "segmentation": np.ndarray,  # binary mask HxW
    "area": int,                  # площадь маски в пикселях
    "bbox": [x, y, w, h],        # bounding box (XYWH)
    "predicted_iou": float,       # predicted quality score
    "point_coords": [[x, y]],    # prompt point координаты
    "stability_score": float,     # stability metric
    "crop_box": [x, y, w, h],    # crop region
}
```

---

## 4. Grounded-SAM-2 — CONTINUOUS ID TRACKING (полный код)

Источник: [grounded_sam2_tracking_demo_with_continuous_id_plus.py](https://github.com/IDEA-Research/Grounded-SAM-2/blob/main/grounded_sam2_tracking_demo_with_continuous_id_plus.py)

### 4.1 Модели

```python
# SAM2
checkpoint = "checkpoints/sam2.1_hiera_large.pt"
config = "configs/sam2.1/sam2.1_hiera_l.yaml"
# dtype: bfloat16 + TF32 для Ampere GPUs

# GroundingDINO  
model_id = "IDEA-Research/grounding-dino-tiny"
# Загружается через HuggingFace transformers
```

### 4.2 Параметры детекции

```python
# GroundingDINO thresholds
box_threshold = 0.25
text_threshold = 0.25

# Text prompt format: lowercase + trailing period
text_prompt = "car."  # или "person. car. dog."
```

### 4.3 Алгоритм Continuous ID Tracking

**Forward pass:**
1. Семплируем keyframes каждые **step=20** кадров
2. На каждом keyframe: GroundingDINO detection → SAM2 image predictor → masks
3. Регистрируем masks в SAM2 video predictor через `add_new_mask()`
4. Propagate predictions между keyframes через `propagate_in_video()`
5. Object continuity через `MaskDictionaryModel` с **IoU matching (threshold=0.8)**

**Reverse pass:**
1. Итерация назад от keyframes
2. Reset predictor state между сегментами
3. Re-register masks + propagate с `reverse=True`
4. Merge reverse predictions с forward masks на pixel level

### 4.4 ID Management

```python
objects_count = {}           # total unique IDs per keyframe
frame_object_count = {}      # cumulative object counts
object_info_dict = {}        # metadata for reverse propagation
iou_threshold = 0.8          # matching threshold for ID continuity
```

### 4.5 Выходные файлы

```
outputs/
├── mask_data/     # uint16 numpy arrays (pixel value = object ID)
├── json_data/     # metadata: labels, boxes per frame
└── result/        # annotated visualization frames + video (15 fps)
```

---

## 5. Det-SAM2 — PERIODIC RE-DETECTION

Источник: [arxiv.org/html/2411.18977v1](https://arxiv.org/html/2411.18977v1)

### 5.1 Архитектура

- Detector: **YOLOv8** (не GroundingDINO!)
- Bounding boxes → SAM2 prompt encoder (автоматически, без manual interaction)
- **Interval-based detection:** НЕ на каждом кадре, а через configurable интервалы
- Кадры копятся в buffer (size K), затем batch processing

### 5.2 Memory management (ключевая инновация)

| Параметр | Назначение |
|----------|------------|
| `max_frame_num_to_track` | Макс. длина backward correction |
| `max_inference_state_frames` | Кадров в памяти (≥ propagation length) |
| Detection interval | Частота запуска YOLOv8 |
| Buffer size K | Кадров перед batch processing |

**Результат:** Inference бесконечно длинных видео с КОНСТАНТНЫМ VRAM/RAM.

### 5.3 Performance

- Baseline: ~200 frames per 24GB VRAM (6-7 objects/frame)
- `offload_video_to_cpu`: экономит ~0.025 GB/frame
- `offload_state_to_cpu`: -22% inference speed, но значительно меньше VRAM
- FP16: экономит ~0.007 GB/frame, минимальная потеря качества
- Computational reduction: O(N²) → O(N²/K) при накоплении K frames

### 5.4 Критическое ограничение

> "Only be used in scenarios where each category can only appear once in each frame"

Несколько объектов одного класса → путаница ID. Grounded-SAM-2 это решает лучше.

---

## 6. CVAT + SAM2 — Annotation Workflow

- **SAM2 Tracker** интегрирован в CVAT Online и Enterprise
- Click на объект в одном frame → automatic tracking across frames
- **Batch mode:** Track multiple objects simultaneously (Menu → Run Actions)
- **AI Agent:** Можно запустить SAM2 как worker process на своём hardware
- [CVAT SAM2 docs](https://docs.cvat.ai/docs/annotation/auto-annotation/segment-anything-2-tracker/)

---

## 7. MASKS → CRYPTOMATTE CONVERSION

### 7.1 Существующие Python tools

| Tool | Направление | Описание |
|------|-------------|----------|
| [decryptomatte](https://github.com/tappi287/decryptomatte) | Cryptomatte → masks | Extraction + manipulation. Требует OpenImageIO |
| [cryptomatte-masks](https://github.com/Synthesis-AI-Dev/cryptomatte-masks) | Cryptomatte → PNG | Export combined mask + JSON mapping |
| [kriptomatte](https://pypi.org/project/kriptomatte/) | Cryptomatte → PNG | Colored PNGs with alpha |

### 7.2 Cryptomatte формат (для записи)

- IDs хранятся в R и B каналах как FLOAT (интерпретируются как UINT)
- **НЕЛЬЗЯ** color manage или модифицировать ID-каналы
- Требуется 32-bit precision EXR
- Manifest (name → hash mapping) в metadata EXR

### 7.3 OpenEXR Deep IDs Specification

- [openexr.com/DeepIDsSpecification](https://openexr.com/en/latest/DeepIDsSpecification.html)
- Более современный подход чем классический Cryptomatte

### 7.4 Потенциальный pipeline: SAM2 masks → Cryptomatte EXR

```
SAM2 instance masks (uint16 per pixel)
    ↓
Assign stable hash per object ID (mmh3 / CityHash)
    ↓
Pack into Cryptomatte channel pairs (R=id_float, G=coverage)
    ↓
Write manifest metadata
    ↓
Save as multi-channel EXR (OpenEXR Python)
```

Готового open-source скрипта для SAM2 → Cryptomatte пока **НЕ НАЙДЕНО**.
Image Engine использует ObjectId (не Cryptomatte).

---

## 8. FMX 2026 (upcoming)

- **Даты:** 5-9 May 2026, Stuttgart
- **Тема:** "THE ROAD AHEAD"
- Подтверждённые сессии: Gaussian splat scanning, **automated rotoscoping**, marker removal
- [fmx.de / digiproconf.org](https://digiproconf.org/)

---

## 9. СВОДНАЯ ТАБЛИЦА: КТО ЧТО ИСПОЛЬЗУЕТ

| Студия/Продукт | Detector | Segmenter | Tracker | Output | Public? |
|----------------|----------|-----------|---------|--------|---------|
| **Image Engine** | GroundingDINO | SAM2 | SAM2 Video | ObjectId EXR | Paper (DigiPro 2025) |
| **Grounded-SAM-2** | GroundingDINO / Florence-2 / DINO-X | SAM2 | SAM2 Video | uint16 numpy + JSON | Open source (3.4k stars) |
| **Det-SAM2** | YOLOv8 | SAM2 | SAM2 Video | masks | Open source |
| **MARZ** | Proprietary | Proprietary | Proprietary | Production mattes | Closed |
| **Batch** | Proprietary (patented) | Proprietary | Proprietary | Multi-ch EXR / ProRes | Closed |
| **Slapshot** | Unknown (likely SAM2-based) | SAM2 (confirmed) | SAM2-based | EXR/ProRes/Cryptomatte | SaaS |
| **Weta FX** | Proprietary | Proprietary | Proprietary | Internal | Closed |
| **SmartROTO** | N/A | ML shape predictor | ML tracking | Nuke splines | Research (not shipped) |
| **RotoShop (OTOY)** | N/A (takes masks as input) | N/A | Differentiable splining | Nuke splines | Paper (SIGGRAPH Asia 2025) |
| **CopyCat Disney** | N/A | Soft color segmentation | CopyCat inference | Nuke soft layers | Open (Nuke built-in) |

---

## 10. КЛЮЧЕВЫЕ ВЫВОДЫ ДЛЯ НАШЕГО PIPELINE

1. **Image Engine — единственная студия с публичным paper.** Все остальные — proprietary или research.

2. **Grounded-SAM-2 repo — лучшая open-source основа:**
   - `grounded_sam2_tracking_demo_with_continuous_id_plus.py` — production-ready скрипт
   - SAM2.1 Hiera Large + GroundingDINO Tiny
   - step=20, IoU=0.8, thresholds=0.25

3. **ObjectId EXR (Image Engine) vs Cryptomatte:** Готового конвертера SAM2→Cryptomatte нет. Проще писать свой ObjectId-like формат или multi-channel EXR с масками.

4. **Масштаб проблемы SAM2:**
   - Max 1024px longest edge (Video Predictor)
   - Мелкие детали (пальцы, волосы) — gaps
   - 10 frames/min на A4000 — это batch, не real-time

5. **RotoShop (OTOY) — потенциально полезен** как post-processing: SAM2 masks → Nuke splines (1000 frames за 20 мин, 500x compression).

6. **Det-SAM2 лучше для длинных видео** (constant VRAM), но ограничен 1 объект per class.
