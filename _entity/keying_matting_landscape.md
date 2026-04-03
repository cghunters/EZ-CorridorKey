# Ландшафт технологий Keying & Matting в VFX (2024–2026)

> Исследование: сравнение подходов к сегментации, матированию и кеингу в VFX-индустрии

---

## Иерархия технологий

```
СЛОЙ 3 — ПРОДУКТЫ (GUI, pipeline-интеграция)
  ├── EZ-CorridorKey       — desktop-обёртка над CorridorKey + alpha-генераторы
  ├── Silhouette FX        — Boris FX, ручная рото + paint + keying
  ├── Mocha Pro            — Boris FX, planar tracking + masking
  ├── After Effects Rotobrush 3.0 — Adobe
  ├── DaVinci Resolve Magic Mask — Blackmagic Design
  ├── Runway               — облачный SaaS
  └── Wonder Studio        — Wonder Dynamics / Autodesk

СЛОЙ 2 — СПЕЦИАЛИЗИРОВАННЫЕ ML-МОДЕЛИ (обучены под конкретные VFX-задачи)
  ├── CorridorKey model    — Hiera backbone + CNN refiner, "физическое размешивание" FG/BG
  ├── GVM                  — Generative Video Matting, diffusion-based alpha
  ├── VideoMaMa            — Video Mask-to-Matte, Adobe Research + KAIST, CVPR 2026
  ├── MatAnyone2           — универсальный маттинг по промптам
  ├── BiRefNet             — bilateral reference network, salient object
  ├── RVM                  — Robust Video Matting, real-time
  └── ViTMatte / DiffMatte — transformer/diffusion matting

СЛОЙ 1 — ФУНДАМЕНТАЛЬНЫЕ МОДЕЛИ (general-purpose segmentation)
  ├── SAM2 / SAM 2.1       — Meta, Segment Anything Model 2
  ├── SAM                  — Meta, оригинал (только изображения)
  └── DINOv2               — Meta, self-supervised vision features
```

**Ключевой инсайт:** SAM2 — это «GPS-чип»: говорит ГДЕ объект. Модели Слоя 2 говорят КАК его извлечь с production-quality краями (волосы, motion blur, прозрачность). Продукты Слоя 3 оборачивают это в рабочий процесс.

---

## Сравнительная таблица

| Технология | Тип | Нужен GS? | Использует SAM2? | Качество alpha | Скорость | Цена | Лучше всего для |
|------------|-----|-----------|-------------------|----------------|----------|------|-----------------|
| **CorridorKey** | Open-source модель | Да | Как alpha-hint | Исключительное (волосы, MBlur) | ~1.5с/кадр (4K CUDA) | Free (NC) | AI-keying зелёнки |
| **GVM** | Open-source модель | Заточена под GS | Нет | Очень хорошие soft mattes | Средняя | Free (NC) | Авто-alpha для GS |
| **VideoMaMa** | Open-source модель | Нет (с промптами) | Через SAM2 tracking | Отличное | Медленная | Free (NC) | High-quality маттинг |
| **MatAnyone2** | Open-source модель | Нет | Через SAM2 tracking | Очень хорошее | Средняя | Free (Apache 2.0) | Универсальный маттинг |
| **BiRefNet** | Open-source модель | Нет | Нет | Хорошее (single frame) | Быстрая | Free (MIT) | Быстрая вырезка |
| **SAM2** | Фундаментальная модель | Нет | **ЭТО И ЕСТЬ SAM2** | Только binary mask | Быстрая | Free (Apache 2.0) | Трекинг/сегментация |
| **Silhouette FX** | Коммерческий | Нет | Исследует | Ручная = идеальная | Зависит от артиста | $1,695 / $595/год | Рото для кино |
| **Mocha Pro** | Коммерческий | Нет | Исследует | Ручная = идеальная | Зависит от артиста | $995 / $295/год | Planar tracking + рото |
| **Rotobrush 3.0** | Фича AE | Нет | Нет (Sensei) | Хорошее | Почти real-time | CC подписка | Motion graphics |
| **Resolve Magic Mask** | Фича Resolve | Нет | Нет | ОК | Быстрая | Free / $295 | Изоляция для грейдинга |
| **Runway** | Cloud SaaS | Нет | Не раскрыто | Базовое | Быстрая | $12–76/мес | Контент-мейкеры |
| **Wonder Studio** | Cloud SaaS | Нет | Не раскрыто | Хорошее | Средняя | Autodesk pricing | CG character replacement |

---

## Детальный разбор каждой технологии

### 1. CorridorKey

| Поле | Детали |
|------|--------|
| Разработчик | Niko Pueringer (Corridor Digital) |
| Лицензия | CC BY-NC-SA 4.0 (non-commercial) |
| Статус | Выпущен, активно поддерживается через EZ-CorridorKey GUI |
| Архитектура | Hiera backbone (hierarchical ViT) + CNN Refiner |

**Принципиальное отличие от традиционных кееров:**

Традиционные кееры (Keylight, Primatte, IBK) **вычитают** зелёный канал и выводят альфу из разницы. CorridorKey **физически размешивает** (unmixes) foreground RGB и alpha как **независимые** выходы нейросети:

1. Вход: RGB-кадр + грубая маска-подсказка (alpha hint)
2. Hiera backbone извлекает иерархические фичи
3. CNN Refiner уточняет края на полном разрешении 2K
4. Выход: **чистый FG RGB** (без spill) + **уточнённая alpha** — раздельно

Поэтому CorridorKey сохраняет **волосы, motion blur, прозрачность** — то, что убивает классические кееры. Он не вычитает зелёный из пикселей, а _учится_, как выглядит foreground без зелёного экрана.

**Требование alpha hint:** CorridorKey **не генерирует** свою альфу. Ему нужна грубая маска, которую он затем уточняет. Генераторы масок:

| Генератор alpha | Что делает | Размер модели | Скорость |
|----------------|------------|---------------|----------|
| GVM | Auto-matte из GS-футажа, diffusion | ~6 GB | Средняя |
| VideoMaMa | Лучшее качество, нужны paint prompts (SAM2) | ~37 GB | Медленная |
| MatAnyone2 | Универсальный по промптам | Средний | Средняя |
| BiRefNet | Single-image, salient object | Маленький | Быстрая |
| SAM2 Track Mask | Рисуешь FG/BG → SAM2 распространяет по кадрам | ~324 MB | Быстрая |
| Import Alpha | Маски из Silhouette, Nuke, AE и т.д. | N/A | N/A |

---

### 2. SAM2 (Segment Anything Model 2) — Meta

| Поле | Детали |
|------|--------|
| Разработчик | Meta FAIR |
| Лицензия | Apache 2.0 |
| Текущая версия | SAM 2.1 |

**Архитектура:**

```
INPUT: Видеокадры + промпты (точки, боксы, маски)
        │
        ▼
┌──────────────────────┐
│  Hiera Image Encoder │  ← Hierarchical ViT (та же семья, что у CorridorKey)
│  (каждый кадр        │
│   независимо)        │
└─────────┬────────────┘
          ▼
┌──────────────────────┐
│  Memory Attention    │  ← Cross-attention к банку памяти
│  Module              │     (temporal consistency)
└─────────┬────────────┘
          ▼
┌──────────────────────┐
│  Mask Decoder        │  ← Лёгкий декодер + Prompt Encoder
└─────────┬────────────┘
          ▼
┌──────────────────────┐
│  Memory Bank         │  ← Хранит прошлые предсказания + фичи
│  (6–8 последних      │     Обеспечивает "трекинг" через видео
│   кадров + prompted) │
└──────────────────────┘
          ▼
OUTPUT: Бинарные маски сегментации (на кадр)
```

**Что SAM2 МОЖЕТ и чего НЕ МОЖЕТ:**

| ✅ Может | ❌ Не может |
|----------|------------|
| Находить ГДЕ объекты и трекать их | Делать soft alpha (волосы, прозрачность) |
| Интерактивно: клик → маска по всему видео | Работать с зелёным экраном или color science |
| Быть фундаментом для downstream-тулов | Удалять фон или генерировать clean plate |
| Обрабатывать видео произвольной длины | Заменить compositor или matting model |

**Размеры моделей SAM 2.1:**

| Вариант | Параметры | Скорость | Качество |
|---------|-----------|----------|----------|
| Tiny | 38.9M | Самая быстрая | Хорошее |
| Small | 46M | Быстрая | Лучше |
| Base+ | 80.8M | Средняя | Отличное |
| Large | 224.4M | Медленнее | Лучшее |

---

### 3. VideoMaMa (Video Mask-to-Matte) — Adobe Research + KAIST

| Поле | Детали |
|------|--------|
| Тип | Open-source модель (CVPR 2026) |
| Разработчик | Adobe Research + KAIST + Korea University (3 из 6 авторов — Adobe) |
| Архитектура | Stable Video Diffusion как generative prior, single forward pass |
| Лицензия | CC BY-NC 4.0 (код), Stability AI Community License (веса) |
| GitHub | https://github.com/cvlab-kaist/VideoMaMa |
| Демо | https://huggingface.co/spaces/SammyLim/VideoMaMa |

**Что делает:** превращает грубые маски сегментации в **pixel-accurate alpha mattes** для видео. Обрабатывает волосы, дым, motion blur, полупрозрачные объекты. Работает на **любом** футаже, не только зелёнка.

**Ключевое:** использует предобученный Stable Video Diffusion как generative prior — за один forward pass (не итеративная диффузия). Авторы собрали датасет **MA-V** (50,541 реальных видео с matting-аннотациями) — в 50 раз больше всех предыдущих video matting датасетов.

Также выпустили **SAM2-Matte** — SAM2, дообученный на этих данных.

В EZ-CorridorKey интегрирован с 13 марта 2026 как один из генераторов alpha-hint. Даёт **лучшее качество** среди всех генераторов, но самая тяжёлая модель (~37 GB) и медленная.

**Это то самое «VideoMama» — Adobe разрабатывает с KAIST.**

---

### 4. MatAnyone2

| Поле | Детали |
|------|--------|
| Тип | Open-source модель |
| Разработчик | S-Lab (NTU, Сингапур) + SenseTime Research |
| Лицензия | Apache 2.0 ✅ (коммерческое использование разрешено) |
| GitHub | https://github.com/pq-yang/MatAnyone2 |
| Демо | https://huggingface.co/spaces/PeiqingYang/MatAnyone |

"Mat Anyone from Any Video" — универсальный маттинг по интерактивным промптам. Работает на **любом** футаже (не только зелёнка). Временная согласованность через memory-based video-aware архитектуру. Pixel-level alpha.

**Это ближайший аналог к идее "вырезать ЛЮБОЙ объект из фона без GS".**

---

### 5. Silhouette FX (Boris FX)

| Поле | Детали |
|------|--------|
| Цена | $1,695 perpetual / $595/год |
| Платформы | Win, macOS, Linux. Standalone + OFX plugin (Nuke, Resolve, Flame) |

Золотой стандарт ручной ротоскопии с 2005 года. ML-фичи (Silhouette 2024–2025):
- **Smart Roto / ML Segmentation:** one-click сегментация людей → начальные рото-шейпы, которые артист дорабатывает Bezier-инструментами
- Используют **собственную** ML-модель (не SAM2 напрямую, но архитектурно похожую)
- ML — **стартовая точка**, не замена ручному рото

Boris FX публично заявляли, что **исследуют SAM2-интеграцию**, но пока не выпустили.

---

### 6. Mocha Pro (Boris FX)

| Поле | Детали |
|------|--------|
| Цена | $995 perpetual / $295/год |
| Ключевая технология | Planar tracking (трекинг плоскостей, а не точек) |

Лучший planar tracker в индустрии. ML-интеграция (2024–2025):
- **PowerMesh:** трекинг деформируемых поверхностей (лица, тело)
- **ML Object Segmentation:** авто-генерация начальных сплайн-масок
- ML-маски **грубые** по сравнению с SAM2 — дают rough shapes, не pixel-perfect

Boris FX осторожничают с SAM2 в Mocha — их planar tracking уже best-in-class, ML-маски как дополнение.

---

### 7. Другие заметные игроки

| Продукт | Разработчик | Тип | Суть | Цена |
|---------|------------|-----|------|------|
| **Boris FX Matte Refine ML** | Boris FX | Фича Silhouette/Mocha/Continuum | AI-уточнение масок, специализация на волосах/мехе. NAB 2025 | $595–1,695/год |
| **ONYX AI Matte** | Evgeniy Shatskiy | OFX plugin (Nuke, Resolve) | SAM3 + VitMatte, автомаски из видео, волосы/мех. ⚠️ **Windows + NVIDIA only** | €80 perpetual |
| **Beeble SwitchLight 3.0** | Beeble (Корея) | Cloud SaaS | Full VFX passes из обычного видео, вкл. alpha | Подписка |
| **Kognat Rotobot** | Kognat | OFX plugin (Nuke) | DL-сегментация, semantic/instance | Коммерческий |
| **Rotobrush 3.0** | Adobe | Фича AE | ML-рото через Adobe Sensei (своя модель, не SAM2) | CC подписка |
| **DaVinci Resolve Magic Mask** | Blackmagic | Фича Resolve | ML-сегментация людей/объектов, для цветокоррекции | Free / $295 |
| **Runway** | Runway AI | Cloud SaaS | One-click background removal, таргет — контент-мейкеры | $12–76/мес |
| **Wonder Studio** | Autodesk | Cloud SaaS | Автоматическая замена актёра на CG-персонажа | Autodesk pricing |
| **Nuke CopyCat** | Foundry | Фича Nuke | Тренировка кастомных ML-моделей внутри Nuke | ~$5,000/год |
| **Foundry + Griptape** | Foundry | Coming (2026) | AI-интеграция в Nuke pipeline, Foundry купила Griptape (Feb 2026) | TBD |

---

## Пайплайн: как всё работает вместе

```
ЗЕЛЁНКА:
  Футаж (GS) → GVM (авто-alpha) → CorridorKey → FG RGB + Alpha (EXR)

ПРОИЗВОЛЬНЫЙ ФУТАЖ:
  Футаж → SAM2 (рисуем промпты → бинарная маска) → VideoMaMa/MatAnyone2 (soft matte) → CorridorKey* → FG RGB + Alpha

РУЧНАЯ РАБОТА:
  Футаж → Silhouette/Mocha/Rotobrush (ML-assist + ручная доработка) → Alpha → CorridorKey* → FG RGB + Alpha

* CorridorKey опционален, если маттинг-модель уже даёт достаточное качество
```

---

## Выводы для VFX-супервайзера

### 1. CorridorKey — реально новый подход
"Физическое размешивание" (separate FG RGB + alpha) архитектурно отличается от всего на рынке. Традиционные кееры вычитают; CorridorKey выводит. Результаты на волосах, motion blur и прозрачности объективно лучше.

### 2. SAM2 — инфраструктура, не продукт
Это «движок», на котором строят. Даёт бинарные маски (ГДЕ объект?), а не production alpha (НАСКОЛЬКО прозрачен этот пиксель?). Пайплайн: SAM2 → matting model → CorridorKey.

### 3. Open-source стек уже конкурирует с коммерческими инструментами
Комбинация SAM2 + VideoMaMa/MatAnyone2 + CorridorKey в EZ-CorridorKey даёт результаты, сравнимые или превосходящие Keylight/Primatte для многих шотов, особенно сложных (волосы, прозрачность).

### 4. Boris FX (Silhouette/Mocha) в переходном периоде
ML-фичи пока ассистивные, на простых проприетарных моделях. SAM2-уровень интеграции на подходе, но ещё не выпущен. Их ценность — ручной контроль артиста, ML ускоряет старт.

### 5. Лицензионный барьер
CC BY-NC-SA 4.0 на CorridorKey и GVM — **блокер для студийного продакшна**. MatAnyone2 (Apache 2.0) и SAM2 (Apache 2.0) — без ограничений. Для коммерческой работы нужно либо решать лицензию, либо использовать только permissive-компоненты.

### 6. macOS — второй сорт для ML
Все GPU-тяжёлые ML-модели (кроме самого CorridorKey с MLX) работают на MPS, что значительно медленнее CUDA. Для Mac-пайплайна рекомендуется импорт готовых альфа-масок из другого инструмента.

---

*Дата исследования: 2026-04-03*
*Источники: анализ кодовой базы EZ-CorridorKey, публичные репозитории, документация Boris FX, Meta AI Research, анонсы NAB/SIGGRAPH 2024–2025*
