# Автоматическая ротоскопия и сегментация в VFX-пайплайнах студий (2024-2026)

> Deep research: как студии автоматизируют генерацию масок для каждого человека/объекта в кадре
> Дата исследования: 2026-04-03

---

## TL;DR — Ответ на главный вопрос

Студии используют комбинацию **детектора** + **сегментатора** + **трекера**, развёрнутую в контейнерах на GPU-серверах, которая автоматически запускается при инджесте плейтов. Конкретный стек: **GroundingDINO** (детекция по текстовому промпту "person") → **SAM2** (per-frame сегментация) → **SAM2 Video Tracking** (temporal consistency). На выходе — ObjectId/Cryptomatte EXR с отдельной маской на каждого человека. Это доказано в продакшне: **Image Engine** (Vancouver) опубликовала peer-reviewed paper на **DigiPro 2025** (SIGGRAPH) — 12 шоу, 1,241 шот обработан автоматически.

---

## 1. Эталонный пайплайн: Image Engine (DigiPro 2025)

### Источник
**"Automated Video Segmentation Machine Learning Pipeline"**
- Авторы: Johannes Merz, Lucien Fostier
- Студия: **Image Engine Design Inc.**, Vancouver, Canada
- Конференция: DigiPro '25 (The Digital Production Symposium), SIGGRAPH, 9 августа 2025
- [arXiv](https://arxiv.org/abs/2507.07242) | [ACM DL](https://dl.acm.org/doi/10.1145/3744199.3744635)

### Архитектура пайплайна (3 стадии)

```
PLATE INGEST (авто-триггер)
        │
        ▼
┌──────────────────────────────────────────┐
│  STAGE 1: OBJECT DETECTION               │
│  Модель: GroundingDINO                    │
│  Вход: кадр + текстовый промпт ("person") │
│  Выход: bounding boxes на каждого человека │
│  Режим: open-set (любой класс по тексту)  │
└─────────────────┬────────────────────────┘
                  ▼
┌──────────────────────────────────────────┐
│  STAGE 2: IMAGE SEGMENTATION              │
│  Модель: SAM2 (Segment Anything Model 2)  │
│  Вход: кадр + bounding boxes              │
│  Выход: per-frame binary masks             │
│  Post-processing:                          │
│   - Asymmetric overlap: Sim(A,B) = A∩B/A  │
│   - False positive removal (≥2 overlaps    │
│     при threshold 0.1)                     │
│   - Merge masks (overlap ≥0.6)             │
│   - Directional analysis (вложенность)     │
└─────────────────┬────────────────────────┘
                  ▼
┌──────────────────────────────────────────┐
│  STAGE 3: VIDEO TRACKING                  │
│  Модель: SAM2 (tracking mode)             │
│  Вход: маски Stage 2 → memory bank        │
│  Выход: temporally consistent masks        │
│  Каждый человек = отдельный ID             │
└─────────────────┬────────────────────────┘
                  ▼
┌──────────────────────────────────────────┐
│  OUTPUT: ObjectId EXR                     │
│  Формат: custom ObjectId (Cinesite, 2012) │
│  - Multiple samples per pixel              │
│  - Coverage/alpha values + numeric IDs     │
│  - Manifest: ID → human-readable name      │
│  → Nuke: ObjectId custom filtering nodes   │
└──────────────────────────────────────────┘
```

### Ключевые детали реализации

| Параметр | Значение |
|----------|----------|
| GPU | NVIDIA A4000 16GB (dedicated servers) |
| Контейнеризация | Singularity/Apptainer (не Docker — не нужен root, лучше GPU) |
| Скорость | ~10 frames/min (все 3 стадии), т.е. ~6 sec/frame average |
| 100-frame shot | ~10 минут |
| 173-frame shot | 11:40 (1:13 detect + 5:58 segment + 4:29 track) |
| 283-frame shot | 23:26 (1:44 detect + 11:58 segment + 9:44 track) |
| Промпт по умолчанию | "person" (auto-detect всех людей) |
| Кастомные промпты | "person, car", "eye, hair, necklace, clothing, mouth" |
| Развёрнуто на | 12 шоу, 1,241 шот (2024-2025) |
| Интеграция | Nuke (ObjectId nodes), планы на Gaffer (GafferML) |
| Ручная доработка | 10-15% шотов требуют минимальных правок |

### Подробности из paper (arxiv 2507.07242)

#### Asymmetric Overlap Metric — формула
```
Sim(A, B) = |A ∩ B| / |A|   ∀(A, B) ∈ Masks
```
Метрика **асимметричная**: Sim(A,B) ≠ Sim(B,A). Она измеряет "какая доля маски A перекрыта маской B". Используется для двух целей:

1. **False positive removal**: Если маска A перекрывается с **≥2** другими масками при Sim(A,B) ≥ 0.1 → маска A удаляется как false positive
2. **Merge**: Если после удаления false positives маска A имеет overlap с **ровно одной** другой маской B при Sim(A,B) ≥ 0.6 → merge (маска A вложена в маску B)

#### Stage 3: Tracking — параметры
| Параметр | Значение | Описание |
|----------|----------|----------|
| sP (step size) | 5 | SAM2 пропагирует маски порциями по 5 кадров |
| sT (tracking interval) | 20 | Каждые 20 кадров — re-detection (Stage 1+2) |
| IoU direct match | 0.9 | Если IoU(tracked, cached) ≥ 0.9 → тот же объект |
| Cache update margin (ε) | 0.1 | Порог для обновления cache при улучшении маски |
| Направление | Forward + Reverse | Сначала forward pass, потом reverse для поздно появившихся объектов |

#### Ограничения и failure cases (из paper)
- **Люди на дальнем плане:** Мелкие фигуры могут не сегментироваться — их маски пересекались бы с другими и деградировали бы tracking
- **Крупные bbox на фоне:** Большой bbox на background-персоне может захватить foreground-объекты → сложно для SAM2
- **Мелкие детали:** Тонкие элементы (волосы, пальцы) теряются
- **Приоритет:** Pipeline жертвует completeness ради quality и stability
- **Память GPU:** 16GB A4000 ограничивает resolution → max dimension масштабируется до 1024px

#### ObjectId EXR — формат
- Разработан в 2012 году на **World War Z** в **Cinesite**
- Integrated в Image Engine в 2021 (IE и Cinesite — партнёры с 2015)
- Хранит **произвольное количество samples per pixel** (не как Cryptomatte где 1 float ID)
- Каждый sample: **alpha value** + **numeric ID**
- **Manifest** в file metadata: маппинг numeric ID → human-readable name ("person_1", "person_2")
- Поддерживает overlapping objects и semi-transparent regions
- В Nuke: кастомные ObjectId filtering nodes (не стандартный Cryptomatte)

### Как работает разделение нескольких людей

1. **GroundingDINO** находит bounding box каждого человека в каждом кадре
2. **SAM2** генерирует per-instance mask внутри каждого bbox
3. Post-processing решает конфликты:
   - Если маска A полностью внутри маски B (Sim ≥ 0.6) → merge
   - Если маска перекрывается с 2+ другими (Sim ≥ 0.1) → false positive → удалить
   - Directional analysis определяет вложенность
4. **SAM2 tracking** присваивает persistent ID каждому человеку через весь шот

### Запуск без участия человека

Пайплайн **запускается автоматически при инджесте плейтов**. Никаких кликов, промптов, выделений. Артист получает готовые маски в Nuke.

---

## 2. Коммерческие сервисы авто-рото

### 2.1 Slapshot Autopilot

| Параметр | Значение |
|----------|----------|
| Тип | Cloud SaaS + Enterprise REST API |
| Сайт | [slapshot.ai](https://slapshot.ai/) |
| Основатели | Hotspring (креативное агентство) |
| Технология | Не раскрыта (proprietary) |
| Что делает | Автоматически рото ВСЕ объекты в шоте, без промптов |
| Выход | **Cryptomatte EXR** (каждый объект = отдельный слой) |
| Форматы | EXR 16-bit, MOV ProRes 4444, JPG/PNG, hard + motion blur mattes |
| Разрешение | До 8K |
| Скорость | "100 frames за ~1 час", "400 shots за 24 часа" |
| Цена | $0.25/frame сверх подписки |
| API | REST API для enterprise pipeline integration |
| Статус | В продакшне, early access с крупными студиями |

**Ключевое:** Autopilot — это **zero-prompt** система. Кидаешь шот → получаешь cryptomatte со ВСЕМИ объектами, каждый на своём слое. Без кликов, без промптов.

### 2.2 Batch

| Параметр | Значение |
|----------|----------|
| Тип | Managed service (не SaaS, а сервис) |
| Сайт | [batch.film](https://www.batch.film/) |
| Технология | Патентованная ML-система |
| Что делает | Обрабатывает целые фильмы/шоу, генерирует маски для людей, объектов, depth |
| Выход | Multi-channel EXR, ProRes |
| Качество | Два тира: "Color" (для грейдинга) и "VFX" (для композитинга) |
| Скорость | "Целый фильм за часы, не за недели" |
| Безопасность | TPN-certified facility |
| Цена | Закрытая (beta, ищут инвесторов) |
| Turnaround | Несколько дней |

**Отличие от Slapshot:** Batch — это **managed service** (отдаёшь им footage, получаешь маски), не self-service SaaS.

### 2.3 MARZ (Monsters Aliens Robots Zombies)

| Параметр | Значение |
|----------|----------|
| Тип | VFX-студия + AI products |
| Локация | Toronto |
| Продукт | Vanity AI (de-aging, digital makeup) + авто-рото pipeline |
| Клиенты | WandaVision, Watchmen, Umbrella Academy (27+ продакшнов) |
| Экономия | 30-40% ниже стоимости традиционных студий |
| Цифры | $8M экономии клиентам, ~100 недель сэкономлено |
| Скорость | "300x быстрее традиционного VFX pipeline" |
| Research | Chief Scientist Danny Cohen-Or, Director Ali Mahdavi-Amiri |

---

## 3. DIY-стек: как собрать свой auto-roto pipeline

### 3.1 Grounded-SAM-2 (open-source)

Это готовый open-source фреймворк, который делает то же, что Image Engine's pipeline.

| Параметр | Значение |
|----------|----------|
| GitHub | [IDEA-Research/Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) |
| Модели | GroundingDINO (1.0/1.5/1.6) + SAM2 + Florence-2 + DINO-X |
| Лицензия | Apache 2.0 (SAM2), GroundingDINO (Apache 2.0) |
| Что делает | Text prompt → detect → segment → track across video |

**Пайплайн:**
```python
# 1. Detect all people in frame 0
grounding_model = GroundingDINO("person.")
boxes = grounding_model.detect(frame_0)

# 2. Segment each detection with SAM2
sam2_predictor.set_image(frame_0)
masks = sam2_predictor.predict(boxes)

# 3. Track through entire video
video_predictor = SAM2VideoPredictor()
video_predictor.add_new_mask(frame_0, masks)
for frame in video:
    tracked_masks = video_predictor.propagate(frame)
```

**Ключевые скрипты:**
- `grounded_sam2_tracking_demo_with_continuous_id.py` — отслеживание с persistent ID
- Промпт `"person."` — auto-detect всех людей
- Составные промпты: `"person. car. dog."` — несколько классов одновременно

### 3.2 Segment Anything for Nuke (Rafael Perez)

| Параметр | Значение |
|----------|----------|
| GitHub | [rafaelperez/Segment-Anything-for-Nuke](https://github.com/rafaelperez/Segment-Anything-for-Nuke) |
| Nuke | 13.2+ |
| GPU | 8GB (HQ) / 4GB (LQ) |
| Платформы | Linux, Windows |
| Принцип | Pre-encode frame → interactive point select → instant mask |

**Ограничение:** SAM (v1, не v2), per-frame (нет video tracking), интерактивный (нужны клики). Но полезен как отправная точка.

### 3.3 Cattery: SAM в Nuke (Foundry Community)

SAM доступен как `.cat` модель через Cattery marketplace в Nuke, запускается через inference node.

---

## 4. Nuke / Foundry экосистема

### 4.1 CopyCat — тренировка show-specific рото

**Workflow для авто-рото через CopyCat:**
1. Артист вручную ротоскопит 5-10 "обучающих" кадров из шота
2. CopyCat обучает нейросеть на парах (input frame → target matte)
3. Обученная модель инференсит маски для оставшихся 100+ кадров
4. Артист правит только проблемные кадры

**Скорость инференса:** ~секунды на кадр (после обучения)

**Disney soft segmentation через CopyCat:**
Disney Research Zurich (2017) — Unmixing-Based Soft Color Segmentation. Оригинал: ~1 час на 4K frame. Через CopyCat: решаешь несколько кадров → обучаешь → инференс за секунды на кадр для всего клипа.

### 4.2 BigCat (Nuke 17) — масштаб продакшна

Расширение CopyCat для тренировки на **десятках/сотнях шотов** одновременно. Общая модель для всего проекта, а не per-shot.

### 4.3 SmartROTO (Research: Foundry + DNEG + University of Bath)

| Параметр | Значение |
|----------|----------|
| Начат | 2019 |
| Участники | Foundry (lead), DNEG (данные + artists), Uni of Bath (CV/ML) |
| Задача | ML-ускорение ручного рото: artist keyframes → ML interpolation |
| Статус | Research (далеко от продакшна, Foundry описывает как "AI-assisted Roto — coming soon") |

**Подход:** артист создаёт spl spline-shapes и 2-3 keyframes → ML предсказывает intermediate keyframes. Это НЕ автоматическая сегментация, а ускорение ручного рото.

### 4.4 Foundry + Griptape (приобретён февраль 2026)

Griptape — AI-оркестрация, MCP-совместимая. Стратегия Foundry: встроить AI как оркестрационный слой в pipeline. Потенциально позволит запускать GroundingDINO + SAM2 + любые модели из Nuke через node-based интерфейс.

---

## 5. Как студии разделяют людей на отдельные маски

### Общий алгоритм (доказан Image Engine paper)

```
КАДР → GroundingDINO ("person")
         │
         ├── bbox Person_1 → SAM2 → mask_1
         ├── bbox Person_2 → SAM2 → mask_2
         ├── bbox Person_3 → SAM2 → mask_3
         └── ... (сколько людей, столько bbox)
                    │
                    ▼
         POST-PROCESSING
         ├── Удалить false positives (overlap check)
         ├── Merge вложенные маски
         └── Directional analysis
                    │
                    ▼
         SAM2 VIDEO TRACKING
         ├── Person_1 → tracked through all frames → ID_001
         ├── Person_2 → tracked through all frames → ID_002
         └── Person_3 → tracked through all frames → ID_003
                    │
                    ▼
         ObjectId EXR / Cryptomatte
         (каждый человек = отдельный ID/layer)
```

### Что используется для детекции

| Детектор | Тип | Промпт | Precision | Используется |
|----------|-----|--------|-----------|-------------|
| **GroundingDINO** | Open-set, text-prompted | "person", "car", любой текст | Высокая | Image Engine (DigiPro paper) |
| **YOLO v8/v11** | Closed-set, 80 классов COCO | Фиксированные классы | Очень высокая для COCO | Некоторые пайплайны (быстрее) |
| **Florence-2** | Multi-task, text-prompted | Любой текст | Высокая | Grounded-SAM-2 (альтернатива GDINO) |
| **DINO-X** | Advanced open-set | Текст + контекст | Наивысшая | Grounded-SAM-2 (новейшая) |

**Почему GroundingDINO, а не YOLO:** open-set detection — можно промптить ЛЮБОЙ класс текстом, не ограничиваясь 80 классами COCO. VFX нуждается в гибкости: "helmet", "sword", "alien" — YOLO этого не знает.

---

## 6. Формат вывода: ObjectId vs Cryptomatte

| Формат | Кто использует | Суть |
|--------|---------------|------|
| **ObjectId** | Image Engine (custom, с 2012 Cinesite) | Multiple samples/pixel, coverage+alpha, numeric IDs + manifest |
| **Cryptomatte** | Slapshot, индустрия | Стандарт Psyop/ASWF. Float ID per pixel, Nuke-нативный (Cryptomatte node) |
| **Multi-channel EXR** | Batch, custom pipelines | person_1.alpha, person_2.alpha — каждый человек в отдельном канале |

Для собственного пайплайна **Cryptomatte** — лучший выбор: Nuke-нативный, стандарт индустрии, gizmos готовы.

---

## 7. Pipeline-архитектура: от инджеста до Nuke

### Вариант A: Auto-ingest (Image Engine стиль)

```
PLATE PUBLISHED TO SHOTGRID/FTRACK
        │
        ▼ (webhook / scheduled job)
GPU SERVER (Singularity container)
        │
        ├── GroundingDINO → detect persons
        ├── SAM2 → segment each instance
        ├── SAM2 tracking → temporal consistency
        └── Export → ObjectId/Cryptomatte EXR
                    │
                    ▼
PUBLISH BACK TO SHOTGRID/FTRACK
(as "auto_roto" version, linked to plate)
        │
        ▼
NUKE COMP TEMPLATE
Auto-loads: Read(plate) + Read(auto_roto) + Cryptomatte picker
Compositor gets pre-separated masks immediately
```

### Вариант B: Cloud API (Slapshot стиль)

```
PLATE SEQUENCE
        │
        ▼ (REST API call)
SLAPSHOT CLOUD (AWS)
        │
        ├── Auto-detect all subjects (zero-prompt)
        ├── Track + separate
        └── Export Cryptomatte EXR
                    │
                    ▼ (download / webhook)
NUKE COMP
```

### Вариант C: CopyCat/BigCat (Foundry стиль)

```
ARTIST MANUALLY ROTOS 5-10 FRAMES
        │
        ▼
COPYCAT TRAINING (GPU, minutes)
        │
        ▼
COPYCAT INFERENCE → masks for ALL frames
        │
        ▼
ARTIST QC + FIX edge cases
```

---

## 8. Скорость и масштаб (benchmarks)

| Система | Скорость | Hardware | Примечание |
|---------|----------|----------|-----------|
| Image Engine pipeline | 10 fps (all 3 stages) | NVIDIA A4000 16GB | 100-frame shot = 10 min |
| Slapshot | 100 frames/hour | Cloud (AWS) | Preview за минуты |
| Slapshot Autopilot | 400 shots/24h | Cloud | Production scale |
| Batch | "фильм за часы" | On-prem (TPN) | Managed service |
| MARZ | "300x быстрее" | Custom | Vs traditional roto |
| CopyCat inference | ~seconds/frame | GPU | After training |

---

## 9. Кто что делает: сводка по студиям

| Студия | Подход | Детали |
|--------|--------|--------|
| **Image Engine** | In-house: GroundingDINO + SAM2 в контейнерах | DigiPro paper, 12 шоу, 1241 шот |
| **DNEG** | SmartROTO research + Metaphysic AI (digital humans) | Research с Foundry + Uni of Bath |
| **Weta FX** | Automated roto integrated into core pipeline | Не раскрыто детально, подтверждено как "in production" |
| **ILM** | Neural rendering + automated tech tasks | Мало публичных деталей |
| **MARZ** | Vanity AI + авто-рото pipeline | 27+ продакшнов, WandaVision, Umbrella Academy |
| **Framestore** | AI pipeline exploration (FMX 2025 talks) | Детали закрыты |

---

## 10. Open-source repos и инструменты для сборки своего pipeline

| Repo | Что делает | Лицензия | Ссылка |
|------|-----------|----------|--------|
| **Grounded-SAM-2** | GroundingDINO + SAM2 detect+segment+track | Apache 2.0 | [GitHub](https://github.com/IDEA-Research/Grounded-SAM-2) |
| **SAM2** | Foundation segmentation + video tracking | Apache 2.0 | [GitHub](https://github.com/facebookresearch/sam2) |
| **GroundingDINO** | Open-set object detection by text prompt | Apache 2.0 | [GitHub](https://github.com/IDEA-Research/GroundingDINO) |
| **Segment-Anything-for-Nuke** | SAM v1 plugin для Nuke | Open | [GitHub](https://github.com/rafaelperez/Segment-Anything-for-Nuke) |
| **Ultralytics YOLO** | Fast instance segmentation (closed-set) | AGPL-3.0 | [GitHub](https://github.com/ultralytics/ultralytics) |
| **Florence-2** | Multi-task vision model (alternative to GDINO) | MIT | [HuggingFace](https://huggingface.co/microsoft/Florence-2-large) |

---

## 11. Практические рекомендации

### Для студии с GPU-серверами (как у Image Engine)

1. Развернуть Grounded-SAM-2 в Singularity/Apptainer контейнере
2. Настроить auto-trigger при публикации плейта в ShotGrid/ftrack
3. Промпт по умолчанию: "person" (auto-detect все люди)
4. Экспорт в Cryptomatte EXR
5. Auto-publish обратно в трекер, привязать к плейту
6. Nuke template auto-loads маски

### Для маленькой студии / freelancer

1. **Slapshot** — self-service cloud, $0.25/frame, zero-prompt Autopilot
2. **Batch** — managed service для больших объёмов
3. **CopyCat** — если уже есть NukeX, train on 5-10 frames

### Для Mac (M2 Max)

Ограниченные варианты GPU-инференса, но:
1. SAM2 работает на MPS (медленнее CUDA, но работает)
2. GroundingDINO можно запустить на CPU/MPS
3. Slapshot/Batch — cloud, не зависит от железа
4. Для production-scale — отправлять на Linux-сервер с RTX 3090

---

## Sources

- [Image Engine DigiPro 2025 Paper (arXiv)](https://arxiv.org/abs/2507.07242)
- [Image Engine DigiPro 2025 Paper (ACM DL)](https://dl.acm.org/doi/10.1145/3744199.3744635)
- [Grounded-SAM-2 GitHub](https://github.com/IDEA-Research/Grounded-SAM-2)
- [Slapshot AI](https://slapshot.ai/)
- [Slapshot Autopilot](https://slapshot.ai/slapshot-autopilot/)
- [Slapshot Autopilot Launch — Televisual](https://www.televisual.com/news/slapshot-launches-auto-ai-rotoscoping-system/)
- [Batch Automated Rotoscoping](https://www.batch.film/)
- [Batch: Automatic roto for the people — RedShark News](https://www.redsharknews.com/batch-automatic-roto-for-the-people)
- [MARZ VFX](https://monstersaliensrobotszombies.com/)
- [MARZ x Weights & Biases](https://wandb.ai/site/customers/marz/)
- [SmartROTO — Foundry](https://www.foundry.com/insights/machine-learning/smartroto-enabling-rotoscoping)
- [Foundry AI Solutions](https://www.foundry.com/ai-solutions)
- [CopyCat + ML in Nuke — fxguide](https://www.fxguide.com/fxfeatured/copycat-inference-machine-learning-in-nuke/)
- [Disney soft segmentation via CopyCat — fxguide](https://www.fxguide.com/fxfeatured/live-action-avos-disney-soft-segmentation-via-copycat-in-nuke/)
- [Segment-Anything-for-Nuke](https://github.com/rafaelperez/Segment-Anything-for-Nuke)
- [Kognat Rotobot — fxguide](https://www.fxguide.com/quicktakes/rotobot-bringing-machine-learning-to-roto/)
- [Mocha Pro 2026 — Digital Production](https://digitalproduction.com/2025/12/11/mocha-pro-2026-refined-re-solved-re-edged/)
- [Image Engine SIGGRAPH 2025](https://image-engine.com/presentations/siggraph-2025/)
- [Nuke 17.0 Release — CG Channel](https://www.cgchannel.com/2026/02/foundry-releases-nuke-17-0-nukex-17-0-nuke-studio-17-0/)
- [ActionVFX Top 10 AI Tools 2026](https://www.actionvfx.com/blog/top-10-ai-tools-for-vfx-workflows)
- [AI Rewriting VFX Pipelines — AWN](https://www.awn.com/vfxworld/how-ai-rewriting-vfx-pipelines)

---

*Формат: сущность-исследование. Обновлять при появлении новых данных.*
*Создано: 2026-04-03*
