# Auto-Segmentation Pipeline: Detect + Segment + Track (без ручных промптов)

> Deep research, 2026-04-03
> Задача: папка кадров → instance masks (person_1, person_2, car_1...) с НУЛЕВЫМ участием человека

---

## Архитектура pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    AUTOMATIC PIPELINE                           │
│                                                                 │
│  КАДРЫ (PNG/EXR/DPX)                                          │
│     │                                                           │
│     ▼                                                           │
│  ┌──────────────────┐   text prompt: "person. car. dog."       │
│  │  OPEN-VOCABULARY  │   (задаётся 1 раз на весь батч)         │
│  │  DETECTOR         │                                          │
│  │  GroundingDINO /  │                                          │
│  │  Florence-2 /     │──→ bounding boxes + class labels        │
│  │  DINO-X /         │    + confidence scores                   │
│  │  YOLO-World       │                                          │
│  └──────────────────┘                                          │
│     │                                                           │
│     ▼  boxes как prompts                                       │
│  ┌──────────────────┐                                          │
│  │  SAM 2 / SAM 2.1 │                                          │
│  │  Image Predictor  │──→ pixel-perfect masks per instance     │
│  └──────────────────┘                                          │
│     │                                                           │
│     ▼  masks + object IDs                                      │
│  ┌──────────────────┐                                          │
│  │  SAM 2 Video      │                                          │
│  │  Predictor        │──→ propagation масок по всем кадрам     │
│  │  (tracking)       │    forward + reverse                     │
│  └──────────────────┘                                          │
│     │                                                           │
│     ▼                                                           │
│  ┌──────────────────┐                                          │
│  │  POST-PROCESSING  │                                          │
│  │  • IoU matching   │──→ person_1.exr, person_2.exr, car_1.exr│
│  │  • ID continuity  │    (отдельные маски per instance)        │
│  │  • Export masks   │                                          │
│  └──────────────────┘                                          │
└─────────────────────────────────────────────────────────────────┘
```

### Как это работает без промптов пользователя

**"Zero interaction" = один текстовый промпт на весь батч.**
Пользователь НЕ кликает, НЕ рисует bbox, НЕ выбирает точки. Он пишет:
```
"person. car. dog. chair."
```
И pipeline автоматически:
1. Детектит ВСЕ объекты этих классов на каждом N-м кадре (step = 10-30)
2. SAM2 генерит pixel-perfect маски по bbox от детектора
3. SAM2 Video Predictor пропагирует маски на ВСЕ промежуточные кадры
4. Reverse tracking заполняет пробелы в начале видео
5. IoU matching связывает один и тот же объект через кадры → continuous ID

**DINO-X пошёл ещё дальше**: поддерживает "prompt-free" режим — детектит ВСЁ без текста вообще. Но это обычно генерит слишком много объектов для VFX.

---

## Сравнение детекторов

| Детектор | Точность (COCO zero-shot) | Скорость | VRAM (inference) | Open-source | Mac/MPS |
|----------|---------------------------|----------|------------------|-------------|---------|
| **GroundingDINO 1.0** | 52.5 AP | ~5 FPS (A100) | ~8 GB | Да (ECCV 2024) | CPU only (~5 sec/frame M1) |
| **GroundingDINO 1.5 Pro** | 55.7 AP | ~3 FPS | ~10 GB | Нет (cloud API) | Через API |
| **GroundingDINO 1.6** | ~56 AP | ~3 FPS | ~10 GB | Нет (cloud API) | Через API |
| **DINO-X Pro** | **56.0 AP** (COCO), **59.8 AP** (LVIS) | ~3 FPS | ~12 GB | Нет (cloud API) | Через API |
| **DINO-X Edge** | Ниже Pro | **20 FPS** (640x640) | ~4 GB | Нет (cloud API) | Через API |
| **Florence-2-large** | ~45 AP (оценка) | ~8 FPS | ~4 GB | Да (HuggingFace) | CPU/MPS |
| **YOLO-World-L** | ~45 AP | **30+ FPS** | ~2-4 GB | Да (Ultralytics) | CPU/MPS |

### Рекомендация для VFX pipeline

- **Максимальная точность (offline rendering)**: GroundingDINO 1.0 (open-source) или DINO-X Pro (cloud API)
- **Баланс скорость/качество**: Florence-2-large — open-source, легче, HuggingFace
- **Real-time / previsualization**: YOLO-World — 30+ FPS, но хуже на мелких/нестандартных объектах
- **Mac M2 Max (твоя машина)**: Florence-2 или YOLO-World на CPU/MPS; GroundingDINO на CPU (~5 сек/кадр)

---

## Обработка множественных instance (person_1, person_2...)

### Проблема
Детектор выдаёт: `[person, person, person]`. Как отличить person_1 от person_2 через 500 кадров?

### Решение в Grounded-SAM-2: Continuous ID

Файл: `grounded_sam2_tracking_demo_with_continuous_id_plus.py`

```python
# Ключевые параметры:
step = 20                    # детекция каждые 20 кадров
detection_threshold = 0.25   # порог уверенности GroundingDINO
iou_threshold = 0.8          # порог для матчинга instance между кадрами

# Алгоритм:
# 1. Детекция на keyframe → N bbox
# 2. SAM2 → N масок
# 3. IoU matching с существующими масками предыдущего keyframe:
#    - IoU > 0.8 → тот же объект, сохраняем ID
#    - IoU < 0.8 → новый объект, objects_count += 1
# 4. Forward propagation: SAM2 Video Predictor пропагирует маски на step кадров вперёд
# 5. Reverse propagation: заполняет кадры ДО первого keyframe
```

### Формат выходных данных
```
output/
├── mask_000001.npy     # uint16, каждый пиксель = object_id (0=фон, 1=person_1, 2=person_2...)
├── mask_000002.npy
├── ...
├── result_video.mp4    # визуализация с цветными масками
└── annotations/
    ├── frame_000001.json  # {class_name, bbox, confidence, instance_id}
    └── ...
```

---

## Ключевые GitHub-репозитории

### 1. Grounded-SAM-2 (основной)
- **URL**: https://github.com/IDEA-Research/Grounded-SAM-2
- **Stars**: ~3.4k
- **Что даёт**: Полный pipeline detect→segment→track. Поддержка GroundingDINO, Florence-2, DINO-X
- **Демо-скрипты для batch/video**:
  - `grounded_sam2_tracking_demo.py` — базовый трекинг
  - `grounded_sam2_tracking_demo_with_continuous_id.py` — с сохранением ID
  - `grounded_sam2_tracking_demo_with_continuous_id_plus.py` — с reverse tracking
  - `grounded_sam2_florence2_autolabel_pipeline.py` — автолейблинг папки изображений
  - `grounded_sam2_tracking_demo_custom_video_input_gd1.5.py` — кастомное видео
- **Требования**: Python 3.10, PyTorch ≥ 2.3.1, CUDA 12.1
- **Docker**: есть Dockerfile

### 2. Det-SAM2 (автономный, без текстового промпта)
- **URL**: https://github.com/motern88/Det-SAM2
- **Stars**: ~57
- **Что даёт**: YOLOv8 как детектор → SAM2 как сегментатор. Полностью автоматический, constant VRAM
- **Фишка**: `detect_interval=30` — детекция каждые 30 кадров, между ними чистый трекинг SAM2
- **Infinite video**: постоянное потребление VRAM/RAM при бесконечном видео
- **Docker**: есть docker-compose.yaml

### 3. autodistill-grounded-sam-2 (auto-labeling)
- **URL**: https://github.com/autodistill/autodistill-grounded-sam-2
- **Stars**: ~134
- **Что даёт**: Florence-2 + SAM2 для автоматической разметки целых папок
- **Использование**:
  ```python
  from autodistill_grounded_sam_2 import GroundedSAM2
  from autodistill.detection import CaptionOntology

  base_model = GroundedSAM2(
      ontology=CaptionOntology({
          "person": "person",
          "car": "car",
      })
  )
  # Лейблит ВСЮ папку одной командой:
  base_model.label("./frames", extension=".jpg")
  ```

### 4. SAM 2 (Meta, базовый)
- **URL**: https://github.com/facebookresearch/sam2
- **Stars**: ~15k+
- **Automatic Mask Generator**: `sam2/notebooks/automatic_mask_generator_example.ipynb` — генерирует ВСЕ маски без промптов, но без class labels

### 5. Grounded-Segment-Anything (v1, legacy)
- **URL**: https://github.com/IDEA-Research/Grounded-Segment-Anything
- **Stars**: ~16k+
- **Статус**: legacy, но стабильный. Для изображений, не видео

---

## VRAM и производительность

### SAM 2 (только сегментация/трекинг, без детектора)

| Модель | Одно изображение 1024x1024 | Видео (steady-state, 1024x1024, 1 объект) |
|--------|---------------------------|------------------------------------------|
| **Tiny** | 458 MB | 1.2 GB |
| **Base+** | 551 MB | 1.3 GB |
| **Large** | 855 MB | 1.7 GB |

- Каждый дополнительный объект: +3-4 MB
- 2048x2048: ~1.2-1.6 GB на изображение
- **ВАЖНО**: по умолчанию SAM2 кеширует бесконечно → VRAM растёт. Отключение кеша фиксит

### Суммарный VRAM (детектор + SAM2)

| Комбинация | Оценка VRAM | Примечание |
|------------|-------------|------------|
| GroundingDINO 1.0 + SAM2-Large | ~10-12 GB | Для 1080p кадров |
| Florence-2-large + SAM2-Base+ | ~6-8 GB | Легче, подходит для 8 GB GPU |
| YOLO-World-L + SAM2-Tiny | ~4-6 GB | Минимальная конфигурация |
| GroundingDINO 1.5/DINO-X (API) + SAM2 | ~2-4 GB локально | Детекция в облаке, SAM2 локально |

### Скорость (на A100 80GB)

| Этап | Время на кадр 1080p |
|------|---------------------|
| GroundingDINO detection | ~200 ms |
| SAM2 segmentation (per box) | ~50 ms |
| SAM2 Video propagation (per frame) | ~30 ms |
| **Total (5 объектов, keyframe)** | **~450 ms** |
| **Total (propagation frame)** | **~30 ms** |

С TensorRT оптимизацией: до 123 ms на keyframe (3x speedup).

---

## Mac / Apple Silicon совместимость

### SAM 2 на Mac
- **MPS (Metal)**: Официально поддерживается (`device="mps"`), но есть баги
  - `RuntimeError: Placeholder storage has not been allocated on MPS device!`
  - **Workaround 1**: ARM64 conda environment (`CONDA_SUBDIR=osx-arm64`)
  - **Workaround 2**: Явно передавать `device` в `build_sam2()`
  - **Workaround 3**: CPU fallback (стабильнее, но медленнее)
- **Реальность**: MPS работает, но нестабильно. CPU — надёжнее на данный момент

### GroundingDINO на Mac
- **MPS**: НЕ работает — Swin Transformer использует `torch.roll`, который не поддерживается MPS
- **CPU**: Работает. ~5 секунд на кадр на M1. На M2 Max ~3-4 секунды (оценка)
- **Итого**: Только CPU inference. Медленно, но работает

### Florence-2 на Mac
- **MPS**: Работает лучше, чем GroundingDINO (нет Swin Transformer)
- **CPU**: ~2-3 секунды на кадр (оценка)
- **HuggingFace Transformers**: полная поддержка

### YOLO-World на Mac
- **MPS**: Работает через Ultralytics (PyTorch MPS backend)
- **CoreML export**: Возможен для максимальной скорости
- **Скорость**: ~10-20 FPS на M2 Max (оценка для 1080p)

### Рекомендация для M2 Max 96GB

| Задача | Лучший вариант |
|--------|---------------|
| Детекция | Florence-2 (MPS) или YOLO-World (MPS) |
| Сегментация | SAM2 Base+ (MPS с workaround, или CPU) |
| Трекинг | SAM2 Video Predictor (CPU стабильнее) |
| Тяжёлая работа | Отправлять на Linux RTX 3090 |

**Время обработки на Mac (оценка, 1080p, 5 объектов):**
- Keyframe (детекция + сегментация): ~5-8 секунд
- Propagation frame: ~0.5-1 секунда
- 1000 кадров (step=20): ~50 keyframes × 8s + 950 propagation × 1s = ~1350 секунд ≈ **22 минуты**

---

## VFX-интеграция: Nuke и OFX плагины

### ONYX Ai Matte (OFX)
- **URL**: https://onyxofx.com
- **Модель**: Meta SAM3 + VitMatte (ONNX, FP16)
- **Хосты**: Nuke 13+, DaVinci Resolve 18+, Fusion Studio
- **Текстовый промпт**: Да — "person", "car", "dog" → SAM3 находит все instance
- **Multi-instance**: Да — "find all people, select specific actors"
- **GPU**: NVIDIA RTX 2060+ (6 GB min, 8 GB+ рекомендовано), CUDA 12.6 + TensorRT 10.x
- **Mac**: НЕТ (Windows only)
- **Пробный период**: 7 дней бесплатно
- **Вердикт**: Ближе всего к тому, что тебе нужно. Но Windows/NVIDIA only

### Rotobot (Kognat, OFX)
- **URL**: https://kognat.com
- **Тип**: Semantic segmentation (81 класс), CNN-based
- **Хосты**: Nuke, OFX-совместимые
- **Платформы**: Linux, Mac, Windows
- **Цена**: $130/год (node-locked), $380/год (floating)
- **Проблема**: Нет temporal stability (мерцание между кадрами), нет editable splines
- **Вердикт**: Устаревший подход. Нет трекинга, нет SAM-level качества

### Boris FX Continuum 2025.5 / 2026
- **Object Brush ML**: Клик внутри объекта → автоматическая маска
- **Хосты**: Nuke, Resolve, After Effects, Avid
- **AI Masking**: Встроенный, не требует отдельного GPU setup
- **Вердикт**: Удобно, но не batch/automatic — требует ручного клика

### Foundry Nuke 17 (встроенные AI tools)
- **CopyCat / BigCat**: Обучение кастомных ML моделей внутри Nuke
- **Не автоматическая сегментация**: Нужно обучать модель на примерах
- **Вердикт**: Мощно, но не zero-interaction pipeline

### Свой Python pipeline → Nuke (лучший вариант для тебя)
Написать Python-скрипт, который:
1. Берёт кадры из папки
2. Прогоняет Florence-2 + SAM2 (или Grounded-SAM-2)
3. Экспортирует маски как EXR (по одному каналу на instance)
4. Импортирует в Nuke через Read node

Это то, что стоит в CLAUDE.md как "Альтернатива: свой Python Nuke node через MLX backend на Mac".

---

## Практический план для Игоря

### Быстрый старт (на Mac, сегодня)
```bash
# 1. Клонируем Grounded-SAM-2
cd /Users/igor/YD/aiRND/03_Claude_Projects/
git clone https://github.com/IDEA-Research/Grounded-SAM-2.git

# 2. Создаём venv
cd Grounded-SAM-2
python3.12 -m venv venv
source venv/bin/activate

# 3. Устанавливаем зависимости
pip install torch torchvision
pip install -e .  # SAM2
pip install transformers  # Florence-2 или GroundingDINO HF

# 4. Тест на папке кадров
python grounded_sam2_florence2_autolabel_pipeline.py \
    --text_prompt "person. car." \
    --img_path ./test_frames/ \
    --output_dir ./output_masks/
```

### Production pipeline (Linux RTX 3090, максимум качества)
```bash
# На Linux-сервере (ssh kostya@100.74.113.63)
# GroundingDINO 1.0 + SAM2-Large + continuous ID tracking
python grounded_sam2_tracking_demo_with_continuous_id_plus.py \
    --text "person. car." \
    --video_dir /path/to/frames/ \
    --step 15 \
    --sam2_checkpoint sam2.1_hiera_large.pt
```

---

---

## Расширенный каталог: Video Segmentation + Tracking репозитории

> Обновлено: 2026-04-03 (deep research round 2)
> Критерии: (1) truly automatic, (2) video-capable с трекингом, (3) качество масок, (4) actively maintained

---

### TIER 1 — Production-ready, automatic, video-capable

#### 1. SAM 3 / SAM 3.1 (Meta, 2025-2026) — НОВЕЙШИЙ
| Параметр | Значение |
|----------|----------|
| **URL** | https://github.com/facebookresearch/sam3 |
| **Stars** | 8,707 |
| **Last push** | 2026-03-31 |
| **License** | SAM License (custom, not Apache/MIT) |
| **Python** | 3.12+ |
| **GPU** | CUDA 12.6 required, NVIDIA only |
| **Mac/MPS** | НЕТ — hard dependency на Triton (CUDA-only) |
| **VRAM** | Не документировано, но тяжелее SAM2 |
| **Speed** | Не документировано |

**Что делает:** Unified model — detect + segment + track по text/image prompt. В отличие от SAM2, SAM3 сам находит ВСЕ instances указанного concept ("person" -> все люди в кадре). SAM 3.1 (март 2026) добавил Object Multiplex — shared-memory multi-object tracking, значительно быстрее.

**Truly automatic?** Почти. Нужен text prompt ("person", "car"), но НЕ нужны bbox/point prompts. AMG (automatic mask generator) через HuggingFace `pipeline("mask-generation", model="facebook/sam3")` — без промптов вообще, но нет video AMG (только images).

**Output:** `masks, boxes, scores`. Принимает JPEG folder или MP4. Видео: masklets per object с temporal tracking.

**Ограничения:** CUDA-only (нет MPS/CPU fallback для production), SAM License (не MIT/Apache), video AMG отсутствует.

---

#### 2. Grounded-SAM-2 (IDEA-Research)
| Параметр | Значение |
|----------|----------|
| **URL** | https://github.com/IDEA-Research/Grounded-SAM-2 |
| **Stars** | 3,395 |
| **Last push** | 2025-11-11 |
| **License** | Apache-2.0 |
| **Python** | 3.10+ |
| **GPU** | CUDA 12.1+, NVIDIA |
| **Mac/MPS** | CPU only для GroundingDINO (~3-4 sec/frame M2), SAM2 MPS нестабильно |
| **VRAM** | ~10-12 GB (GroundingDINO + SAM2-Large) |
| **Speed** | ~450 ms/keyframe, ~30 ms/propagation (A100) |

**Что делает:** Pipeline: GroundingDINO/Florence-2/DINO-X detect → SAM2 segment → SAM2 Video track. Один text prompt на весь батч.

**Truly automatic?** Да, с минимальным промптом ("person. car."). Zero interaction после запуска.

**Output:** NPY masks (uint16, pixel=object_id), JSON annotations, MP4 visualization. Per-frame, не per-object files.

**Лучший выбор для:** VFX pipeline на Linux с NVIDIA GPU. Самый зрелый и документированный.

---

#### 3. DEVA — Tracking Anything with Decoupled Video Segmentation (hkchengrex)
| Параметр | Значение |
|----------|----------|
| **URL** | https://github.com/hkchengrex/Tracking-Anything-with-DEVA |
| **Stars** | 1,490 |
| **Last push** | 2025-04-26 |
| **License** | Custom (see LICENSE.md) |
| **Python** | 3.9+ |
| **GPU** | NVIDIA recommended, --amp для mixed precision |
| **Mac/MPS** | Не тестировалось ("Tested on Ubuntu only") |
| **VRAM** | Не документировано |
| **Speed** | Не документировано, --size 480 для ускорения |

**Что делает:** Decoupled: (1) image-level segmentation (SAM automatic grid points ИЛИ Grounded-SAM text prompts), (2) temporal propagation. Два режима: automatic (SAM grid, zero prompts) и text-prompted (Grounded-SAM).

**Truly automatic?** ДА — automatic mode с points-in-grid, вообще без текстовых промптов. Единственный инструмент тут с TRUE zero-prompt video segmentation.

**Output:** PNG masks per frame (Annotations/), pred.json, Visualizations/. Каждый пиксель = instance ID.

**Ограничения:** Не обновлялся с апреля 2025. Ubuntu only. Dependencies: Grounded-SAM v1, SAM v1 (не SAM2).

---

#### 4. Segment-and-Track-Anything / SAM-Track (z-x-yang)
| Параметр | Значение |
|----------|----------|
| **URL** | https://github.com/z-x-yang/Segment-and-Track-Anything |
| **Stars** | 3,120 |
| **Last push** | 2026-03-13 |
| **License** | AGPL-3.0 |
| **Python** | 3.9+ |
| **GPU** | NVIDIA, ~16 GB для 255 объектов |
| **Mac/MPS** | Не упоминается |
| **VRAM** | <10 GB с amp для 255 objects, ~16 GB max |
| **Speed** | Зависит от sam_gap |

**Что делает:** SAM detects + DeAOT tracks. Автоматически обнаруживает новые объекты каждые `sam_gap` кадров.

**Truly automatic?** ДА — `sam_gap` контролирует как часто SAM сканирует новые объекты. Полностью automatic.

**Output:** PNG masks per frame + GIF visualization.

**Ключевые параметры:** `sam_gap` (частота детекции новых), `max_obj_num` (лимит объектов, default 255).

**Ограничения:** AGPL-3.0 (copyleft!). Использует SAM v1 + DeAOT (не SAM2). Нет text labels (просто instance IDs без class names).

---

#### 5. AutoSeg-SAM2 (zrporz)
| Параметр | Значение |
|----------|----------|
| **URL** | https://github.com/zrporz/AutoSeg-SAM2 |
| **Stars** | 225 |
| **Last push** | 2025-07-09 |
| **License** | MIT |
| **Python** | 3.10+ |
| **GPU** | NVIDIA (torch >= 2.3.1) |
| **Mac/MPS** | Не упоминается |
| **VRAM** | Не документировано |
| **Speed** | Не документировано |

**Что делает:** SAM1 для static segmentation → SAM2 для tracking. Автоматическая full segmentation видео.

**Truly automatic?** ДА — zero prompts. SAM1 AMG на keyframes → SAM2 propagation.

**Output:** Не документировано подробно, но есть `auto-mask-batch.py` и `visualization.py`.

**Ограничения:** Маленькое комьюнити (225 stars). Слабая документация. Но MIT license.

---

### TIER 2 — Semi-automatic / tracking-focused (нужен initial prompt)

#### 6. SAM 2 (Meta) — Automatic Mask Generator
| Параметр | Значение |
|----------|----------|
| **URL** | https://github.com/facebookresearch/sam2 |
| **Stars** | 18,843 |
| **Last push** | 2026-03-20 |
| **License** | Apache-2.0 |
| **Python** | 3.10+ |
| **GPU** | CUDA recommended, MPS частично |
| **Mac/MPS** | Частично — `device="mps"`, но баги (Placeholder storage errors) |
| **VRAM** | 458 MB (Tiny) — 855 MB (Large) per image; 1.2-1.7 GB video |
| **Speed** | ~44 FPS (SAM2 inference), ~30 ms/frame propagation |

**Что делает:** AMG для images (grid-based, zero prompts). Video: нужен initial prompt (point/box/mask) per object, потом automatic tracking.

**Truly automatic?** Images: ДА (AMG). Video: НЕТ — нужен initial prompt per object. AMG НЕ работает на видео из коробки.

**Output:** masks, scores, bounding boxes. PNG export через cv2.

---

#### 7. Cutie (hkchengrex)
| Параметр | Значение |
|----------|----------|
| **URL** | https://github.com/hkchengrex/Cutie |
| **Stars** | 1,032 |
| **Last push** | 2024-11-08 |
| **License** | MIT |
| **Python** | 3.9+ |
| **GPU** | NVIDIA |
| **Mac/MPS** | Не упоминается |
| **VRAM** | 1.35 GB (small, FIFO) — очень мало! |
| **Speed** | 45.5 FPS (small), 36.4 FPS (base) |

**Что делает:** VOS tracking с object-level memory. Нужна маска на первом кадре, потом automatic tracking.

**Truly automatic?** НЕТ — нужна initial mask. Но интегрируется с DEVA для automatic detection.

**Output:** Per-frame masks. Динамическое добавление/удаление объектов.

**Преимущества:** Очень быстрый (45 FPS), очень мало памяти (1.35 GB). MIT license.

---

#### 8. XMem (hkchengrex)
| Параметр | Значение |
|----------|----------|
| **URL** | https://github.com/hkchengrex/XMem |
| **Stars** | 1,965 |
| **Last push** | 2024-11-15 |
| **License** | MIT |
| **Python** | 3.9+ |
| **GPU** | NVIDIA (tested RTX 3090) |
| **Mac/MPS** | Не упоминается |
| **VRAM** | 3.03 GB (больше чем Cutie) |
| **Speed** | 30 FPS на 480p |

**Что делает:** Long-term VOS с тройной памятью (sensory + working + long-term). Основа для Track-Anything.

**Truly automatic?** НЕТ — нужна маска первого кадра. Чистый tracker.

**Output:** Per-frame masks.

**Статус:** Superseded Cutie (того же автора). Используется в Track-Anything.

---

#### 9. Track-Anything (gaomingqi)
| Параметр | Значение |
|----------|----------|
| **URL** | https://github.com/gaomingqi/Track-Anything |
| **Stars** | 6,951 |
| **Last push** | 2025-12-13 |
| **License** | MIT |
| **Python** | 3.9+ |
| **GPU** | NVIDIA |
| **Mac/MPS** | Не упоминается |
| **VRAM** | Не документировано |
| **Speed** | Не документировано |

**Что делает:** SAM + XMem + E2FGVI (video inpainting). GUI: кликаешь на объект → трекинг.

**Truly automatic?** НЕТ — interactive (нужен клик на первом кадре). Нет batch automatic mode.

**Output:** Masks per frame, overlay video.

---

### TIER 3 — Image-only / research / legacy

#### 10. Mask2Former (Facebook Research)
| Параметр | Значение |
|----------|----------|
| **URL** | https://github.com/facebookresearch/Mask2Former |
| **Stars** | 3,303 |
| **Last push** | 2024-07-29 |
| **License** | MIT |
| **Archived** | ДА — archived repo |

**Что делает:** Universal segmentation (semantic/instance/panoptic). Video extension через DVIS/DVIS++.

**Truly automatic?** ДА для images (предсказывает все masks + labels). Video — через DVIS pipeline.

**Ограничения:** ARCHIVED. Detectron2-based. Тяжёлый setup. Superseded SAM2/SAM3 для масок, но остаётся лучшим для panoptic labels.

---

#### 11. OneFormer (SHI-Labs)
| Параметр | Значение |
|----------|----------|
| **URL** | https://github.com/SHI-Labs/OneFormer |
| **Stars** | 1,701 |
| **Last push** | 2024-10-03 |
| **License** | MIT |
| **GPU** | 8x A6000 (training), inference ~15+ GB |

**Что делает:** One model для semantic + instance + panoptic. Image-only.

**Truly automatic?** ДА для images. НЕТ видео.

**Ограничения:** Image-only. Тяжёлый (219M params, ~15 GB inference).

---

#### 12. DVIS / DVIS++ (zhang-tao-whu / KwaiVGI)
| Параметр | Значение |
|----------|----------|
| **URL** | https://github.com/zhang-tao-whu/DVIS |
| **Stars** | 159 |
| **Last push** | 2024-04-02 |
| **License** | MIT |

**Что делает:** Decoupled Video Instance Segmentation. Поддержка VIS + VPS + VSS. Online и offline modes.

**Truly automatic?** ДА — автоматически сегментирует все instances с class labels.

**Ограничения:** Академический код. Detectron2-based. Сложный setup. Малое комьюнити.

---

#### 13. Ultralytics (YOLO + SAM2 auto_annotate)
| Параметр | Значение |
|----------|----------|
| **URL** | https://github.com/ultralytics/ultralytics |
| **Stars** | 55,381 |
| **Last push** | 2026-04-03 (сегодня!) |
| **License** | AGPL-3.0 |
| **Python** | 3.8+ |
| **GPU** | CUDA, MPS, CPU |
| **Mac/MPS** | ДА — полная поддержка |

**Что делает:** `auto_annotate(data="path/", det_model="yolo11n.pt", sam_model="sam2.1_b.pt")` — YOLO detect + SAM2 segment на папке images.

**Truly automatic?** ДА — zero interaction. Но image-only (нет video tracking).

**Output:** TXT файлы в YOLO format (class_id + normalized polygon points).

**Ограничения:** AGPL-3.0 (copyleft!). Нет video temporal tracking (per-frame only, нет ID continuity).

---

### ИТОГОВЫЙ РЕЙТИНГ

| # | Repo | Auto? | Video? | Tracking? | Labels? | Mac? | License | Stars | Active? |
|---|------|-------|--------|-----------|---------|------|---------|-------|---------|
| 1 | **Grounded-SAM-2** | text prompt | Да | Да (SAM2) | Да | CPU only | Apache-2.0 | 3.4k | 2025-11 |
| 2 | **DEVA** | ZERO prompts | Да | Да | Нет | Ubuntu only | Custom | 1.5k | 2025-04 |
| 3 | **SAM-Track** | ZERO prompts | Да | Да (DeAOT) | Нет | ? | AGPL-3.0 | 3.1k | 2026-03 |
| 4 | **SAM 3 / 3.1** | text prompt | Да | Да | Да | НЕТ | SAM License | 8.7k | 2026-03 |
| 5 | **AutoSeg-SAM2** | ZERO prompts | Да | Да (SAM2) | Нет | ? | MIT | 225 | 2025-07 |
| 6 | **Ultralytics** | ZERO prompts | Нет* | Нет | Да | Да (MPS) | AGPL-3.0 | 55k | 2026-04 |
| 7 | **SAM 2** (AMG) | ZERO prompts | Images only | Video needs prompt | Нет | MPS (баги) | Apache-2.0 | 18.8k | 2026-03 |
| 8 | **Cutie** | needs mask | Да | Да | Нет | ? | MIT | 1k | 2024-11 |
| 9 | **DVIS++** | auto | Да | Да | Да | ? | MIT | 159 | 2024-04 |

\* Ultralytics video tracking (ByteTrack/BoT-SORT) exists but separate from auto_annotate SAM2 pipeline.

### Рекомендация для Игоря (VFX pipeline)

**Лучший вариант сейчас: Grounded-SAM-2** (Apache-2.0, video tracking, class labels, зрелый код).

**Если нужен ZERO text prompts: DEVA** (automatic grid-based SAM) или **SAM-Track** (auto-detect + DeAOT tracking).

**Если нужны class labels + zero prompts: Grounded-SAM-2 с generic prompt** ("object. person. vehicle. animal.") — фактически zero-interaction.

**На Mac M2 Max: Ultralytics auto_annotate** для images (MPS supported), но для video tracking нужен Linux RTX 3090.

**Будущее: SAM 3.1** — лучшее качество, но CUDA-only и кастомная лицензия. Следить за MPS support.

---

## Источники

- [Grounded-SAM-2 (IDEA-Research)](https://github.com/IDEA-Research/Grounded-SAM-2) — 3.4k stars, основной репозиторий
- [Det-SAM2](https://github.com/motern88/Det-SAM2) — 57 stars, YOLOv8 + SAM2 автономный pipeline
- [autodistill-grounded-sam-2](https://github.com/autodistill/autodistill-grounded-sam-2) — 134 stars, auto-labeling
- [SAM 2 (Meta)](https://github.com/facebookresearch/sam2) — 18.8k stars, базовая модель
- [SAM 3 (Meta)](https://github.com/facebookresearch/sam3) — 8.7k stars, новейшая модель (2025-2026)
- [Grounded-Segment-Anything v1](https://github.com/IDEA-Research/Grounded-Segment-Anything) — 16k+ stars, legacy
- [DEVA](https://github.com/hkchengrex/Tracking-Anything-with-DEVA) — 1.5k stars, decoupled video segmentation
- [SAM-Track](https://github.com/z-x-yang/Segment-and-Track-Anything) — 3.1k stars, auto detect + track
- [AutoSeg-SAM2](https://github.com/zrporz/AutoSeg-SAM2) — 225 stars, SAM1+SAM2 auto segmentation
- [Cutie](https://github.com/hkchengrex/Cutie) — 1k stars, fast VOS tracker (45 FPS)
- [XMem](https://github.com/hkchengrex/XMem) — 2k stars, long-term VOS
- [Track-Anything](https://github.com/gaomingqi/Track-Anything) — 7k stars, interactive tracker
- [Mask2Former](https://github.com/facebookresearch/Mask2Former) — 3.3k stars, archived, universal segmentation
- [OneFormer](https://github.com/SHI-Labs/OneFormer) — 1.7k stars, one model all tasks
- [DVIS](https://github.com/zhang-tao-whu/DVIS) — 159 stars, decoupled video instance segmentation
- [Ultralytics](https://github.com/ultralytics/ultralytics) — 55k stars, YOLO + SAM2 auto_annotate
- [DINO-X API](https://github.com/IDEA-Research/DINO-X-API) — API для лучшего open-world детектора
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) — оригинальный детектор
- [Florence-2 (HuggingFace)](https://huggingface.co/microsoft/Florence-2-large) — Microsoft, open-source
- [ONYX OFX](https://onyxofx.com) — SAM3 + VitMatte, Nuke/Resolve plugin (Windows only)
- [SAM2 VRAM benchmarks](https://github.com/facebookresearch/sam2/issues/118) — данные по VRAM
- [SAM2 на Apple Silicon](https://github.com/facebookresearch/sam2/issues/687) — MPS workarounds
- [SAM3 MPS issue](https://github.com/facebookresearch/sam3/issues/164) — запрос на CPU/MPS support
- [SAM3 AMG issue](https://github.com/facebookresearch/sam3/issues/242) — automatic mask generator через HuggingFace
- [SAM 3.1 blog](https://ai.meta.com/blog/segment-anything-model-3/) — Object Multiplex, shared-memory tracking
- [Grounded SAM 2 tutorial (PyImageSearch)](https://pyimagesearch.com/2026/01/19/grounded-sam-2-from-open-set-detection-to-segmentation-and-tracking/)
- [SAM2 TensorRT optimization](https://medium.com/tier-iv-tech-blog/high-performance-sam2-inference-framework-with-tensorrt-9b01dbab4bf7) — 123ms/frame
