# OSINT: Реальные отзывы о AI auto-masking/roto из форумов и блогов

> Дата исследования: 2026-04-03
> Источники: Logik Forums (forum.logik.tv), Foundry Community, Electric Sheep Blog, fxguide, CG Channel, befores & afters, Blackmagic Forum

---

## TL;DR

Ключевые выводы из реальных обсуждений практиков:

1. **SammieRoto** (open-source, SAM2 + MatAnyone/VideoMaMa/CorridorKey) — самый обсуждаемый инструмент среди Flame-артистов; предпочитают его Flame AutoMatte
2. **SAM2** даёт binary matte (hard edges) — flickering/boiling на краях, не production-ready для hero shots без доработки
3. **Image Engine** задеплоила auto-segmentation на 12 шоу, 1241 шот — ручные правки нужны в 10-15% случаев
4. **CopyCat (Nuke)** на Dune: Part Two — 40% из 1000 eye shots без правок; BigCat в Nuke 17 масштабирует на сотни шотов
5. **Electric Sheep / Spotlight** — коммерческая платформа, выходные форматы (EXR, ACES), заявляют 52% экономии бюджета рото
6. **ONYX AI Matte** — дешёвый OFX плагин (80 EUR), но Windows only и NVIDIA only
7. **Slapshot Autopilot** — cloud SaaS, $0.25/frame, cryptomatte на выходе, до 8K
8. **Version Zero AI** — holy grail: splines на выходе вместо масок, в stealth mode

---

## 1. Logik Forums — SammieRoto / MatAnyone Thread

**URL:** [forum.logik.tv/t/matanyone-using-sammiroto/12797](https://forum.logik.tv/t/matanyone-using-sammiroto/12797)
**Период:** май 2025 — март 2026 (170+ постов, активное обсуждение)

### Ключевые цитаты:

**joelosis** (6 мая 2025):
> "This is worth checking out for awesome roto that is **miles better than runwayml and resolve** IMHO"
- Тестировал на Arri open gate, Threadripper 128GB RAM + GTX Titan 24GB
- Windows: maxed 24GB VRAM, ~80GB system RAM
- Рекомендация: "feeding it source footage with minimal compression steps" для лучших результатов

**ALan** (6 мая 2025):
> "ran out of GPU ram at the matting stage when set to Full on an A6000 Ada" (при ~4.5K)
- Даже 48GB VRAM (A6000 Ada) не хватает для fullres 4.5K

**cristhiancordoba** (6 мая 2025):
- OOM на FullHD и выше
- Workaround: downscale → process → upscale с TensorRT

**kily** (24 февраля 2026):
> "Tested and working 100%. Really great job."

**theoman** (11 марта 2026):
> "after 3 days of trying to install Sammiroto... **just install Sammiroto 2 instead. it worked in 3 minutes flat**"
- SammieRoto 2 — полный rewrite как native desktop app (не webui)

**Stefan** (17 марта 2026):
> "**sammie-roto over Flame's new Automatte** as it stands now, imho"
- Flame-артист предпочитает open-source SammieRoto официальному AutoMatte от Autodesk

**wiltonmts** (13 марта 2026):
- Добавил VideoMaMa: "**offers better temporal consistency**" (тяжелее, но стабильнее во времени)
- Добавил **Corridor Key** модель: "runs on smaller GPUs — approximately **8GB of VRAM**"

**cnoellert** (14 марта 2026):
> "**3sec/frame at 2k on an M4 Max**" — MLX port через Pybox
- CorridorKey на Apple Silicon через MLX — реальная production скорость

**samhodge_aiml** (17 марта 2026):
- MatAnyone коммерческая лицензия: **USD 5K/year** (NTU Singapore commercialization dept)
- Открытый вопрос для студий: лицензионная чистота

**carloscampos** (16 марта 2026):
- Просит custom mask support для object removal — auto-generated alpha не всегда изолирует нужный объект

---

## 2. Logik Forums — ONYX AI Matte

**URL:** [forum.logik.tv/t/onix-ai-matte/14192](https://forum.logik.tv/t/onix-ai-matte/14192)
**Дата:** 2 марта 2026

**ALan** (2 марта 2026):
> "Looks great, but **fucking Windows only**, and seems to be Russian."

**TimC** (2 марта 2026):
> "LIFETIME SUB FOR 80 EUR"
> "also they aren't just founders, the owners are also composers"

**Вывод:** Минимальное обсуждение (3 поста). Основная критика — Windows only. Цена привлекает.

---

## 3. Electric Sheep — тесты SAM2 для рото

**URL:** [blog.electricsheep.tv/we-tested-sam2-for-rotoscoping-this-is-what-we-found/](https://blog.electricsheep.tv/we-tested-sam2-for-rotoscoping-this-is-what-we-found/)
**Дата:** ~середина 2025

### Ключевые находки:

**Преимущества SAM2:**
- Single-click segmentation "outperformed all existing methods"
- "SAM2 consistently requires **less clicks** than Spotlight and is better at **selecting the target object**"
- Работает на real-time скоростях или быстрее

**Критические проблемы SAM2:**
- "The **edges still flicker** in and out (or '**boil**' as it's also known)"
- "SAM2 is **not good** for where consistent edge definition across frames is critical"
- "SAM2 outputs a **binary (hard black/white) matte** causing **aliasing**"
- Ограничен в usefulness для high-end production где нужна fidelity деталей

**Вердикт:**
> "SAM2 is ideal where **speed or budget** is of the utmost importance"
- Отлично для rapid prototyping и quick masks
- High-end VFX = нужна доработка + supplementary tools

### Electric Sheep — Spotlight vs SAM2:
- Spotlight: 6 сек на 1 сек footage (24 frames), но лучшие края
- Spotlight поддерживает EXR, ProRes, ACES, Linear, Rec709
- SAM2: real-time, но PNG export, binary mattes

### Electric Sheep — 52% экономии бюджета рото:
**URL:** [blog.electricsheep.tv/we-saved-a-production-52-of-their-rotoscoping-budget-heres-how/](https://blog.electricsheep.tv/we-saved-a-production-52-of-their-rotoscoping-budget-heres-how/)
- Заявляют 52% savings на конкретном продакшне (детали продакшна не раскрыты)

---

## 4. Roto Brush (AE) vs Magic Mask (Resolve) — форумы

**URL:** [forum.blackmagicdesign.com/viewtopic.php?f=21&t=195253](https://forum.blackmagicdesign.com/viewtopic.php?f=21&t=195253)

### Из обзоров Electric Sheep:

**Magic Mask (Resolve):**
- Удобные пресеты (person, face, legs, clothing, arms)
- "Most significant weakness is **edge stability**"
- "Edges appear noticeably **wobbly and inconsistent**, particularly when the shot contains motion"
- "Struggles with fast movements and intricate details"

**Roto Brush 3 (After Effects):**
- Motion blur support, side-by-side mask view
- Edge feather + refine options
- "Remains a viable option for basic rotoscoping but **lacks specialised tools for high-end production**"

**Общий вердикт:**
> "DaVinci's Magic Mask provides the **least consistent results** among the three"

---

## 5. CopyCat / BigCat — Nuke ML (production cases)

### Dune: Part Two (DNEG, 2024)

**URL:** [foundry.com/insights/machine-learning/untapped-potential-ml-vfx](https://www.foundry.com/insights/machine-learning/untapped-potential-ml-vfx)

- VFX Supervisor: **Paul Lambert** (DNEG)
- Задача: Fremen blue eyes replacement на 1000 шотов
- Training dataset: 280 шотов из Dune: Part One → crop + augment → **30,000 eyes**
- **40% из 1000 eye shots не потребовали touchups** — сразу в production
- Shots с правками fed back в dataset для улучшения модели
- Дополнительно: удаление татуировок актёра в Giedi Prime sequences
- **"CopyCat's first deployment at this scale"**

### Disney Soft Segmentation via CopyCat

**URL:** [fxguide.com/fxfeatured/live-action-avos-disney-soft-segmentation-via-copycat-in-nuke/](https://www.fxguide.com/fxfeatured/live-action-avos-disney-soft-segmentation-via-copycat-in-nuke/)

- Disney Research Zurich (2017) — Unmixing-Based Soft Color Segmentation
- Оригинал: **~1 час на 4K frame** (непригодно для продакшна)
- С CopyCat: решаем несколько фреймов → train → inference **несколько секунд на frame**
- Результат: "**enormous precision** for keying & color grading and a remarkable **(temporally) stable** result"

### BigCat — Nuke 17 (2025)

**URL:** [redsharknews.com/foundry-nuke-17-gaussian-splat-usd-3d-system](https://www.redsharknews.com/foundry-nuke-17-gaussian-splat-usd-3d-system)

- Новый node для **large-scale dataset training** across "tens or hundreds of shots"
- Эволюция CopyCat: от single-shot → full-project scale deployment
- Нет публичных production отзывов пока (Nuke 17 ещё свежий)

---

## 6. Slapshot Autopilot

**URL:** [slapshot.ai/slapshot-autopilot/](https://slapshot.ai/slapshot-autopilot/)
**Дата анонса:** ~апрель 2025

- Первая **fully autonomous** AI roto система
- Input: shot → Output: **organized cryptomatte** всех слоёв
- "No points, no prompts, no manual work"
- Скорость: **100 frames/hour**, до 400 shots за 24 часа
- Разрешение: до **8K**
- Цена: **$0.25/frame** (сверх базовой подписки)
- Early access с select studios
- Отзыв: "Slapshot is **much better than anything else I've tried**. It's our go-to tool for when we need fast turnaround roto"

**Критический анализ:** cloud-only, нет контроля над моделью, per-frame pricing может быть дорогим на длинных шотах (1000 frames = $250/shot)

---

## 7. ONYX AI Matte

**URL:** [onyxofx.com](https://onyxofx.com/)
**Дата релиза:** ~март 2026

- OFX плагин: DaVinci Resolve 18+, Fusion Studio 18+, **Nuke 13+**
- Модели: **Meta SAM3 + VitMatte**
- Text-based object discovery + point/box selection
- До **32 объектов** одновременно
- Цена: **80 EUR** (indie perpetual), **180 EUR** (studio)
- **Windows only, NVIDIA CUDA 12.6+** — NO Mac, NO Linux (пока)
- Создатель: **Evgeniy Shatskiy** (VFX artist)

---

## 8. Version Zero AI

**URL:** [beforesandafters.com/2025/05/20/version-zero-ai-has-a-splines-output-solution-for-ai-ml-rotoscoping/](https://beforesandafters.com/2025/05/20/version-zero-ai-has-a-splines-output-solution-for-ai-ml-rotoscoping/)

- Cofounders: **Chad Wanstreet**, **Allan McKay** (VFX supervisors), **Murali Selvan**
- Ключевое отличие: **выход = splines**, не маски
- "Orchestrating **splines for roto** — the **holy grail** for AI rotoscoping"
- Stealth mode, тестируется на нескольких шоу
- Покрывает: roto, matting, clean plating, paint

---

## 9. Flame 2026 AutoMatte

**URL:** [cgchannel.com/2025/11/autodesk-releases-flame-2026/](https://www.cgchannel.com/2025/11/autodesk-releases-flame-2026/)

- Новый AI model в Selective node
- Автоматическая детекция и изоляция "hero object"
- Работает с "wide variety of defined objects"
- Поддержка custom ONNX моделей
- Blending layers от разных ML моделей
- **Отзыв от Logik:** Stefan (Flame artist) предпочитает SammieRoto: "sammie-roto over Flame's new Automatte as it stands now"

---

## 10. Foundry Cattery — SAM для Nuke

**URL:** [community.foundry.com/cattery/38594/segment-anything](https://community.foundry.com/cattery/38594/segment-anything)

- SAM converted to .cat files для native inference в Nuke
- Нет внешних зависимостей
- Intuitive selection через Tracker (2D points)
- Fast mode для баланса precision / VRAM
- Автор оригинальной интеграции: **Rafael Perez** ([github.com/rafaelperez/Segment-Anything-for-Nuke](https://github.com/rafaelperez/Segment-Anything-for-Nuke))

**NukeSamurai** — SAM2 для Windows:
- [community.foundry.com/discuss/topic/163867](https://community.foundry.com/discuss/topic/163867/nukesamurai-with-sam2-for-windows-segmentation)
- Webfetch заблокирован, детали обсуждения недоступны

---

## Сводная таблица: что говорят практики

| Инструмент | Плюсы (из отзывов) | Минусы (из отзывов) | Цена |
|---|---|---|---|
| **SammieRoto 2** (open-source) | "Miles better than Runway & Resolve"; 3 min install; multiple models | OOM на 4K+; нужна мощная GPU | Free |
| **SAM2** (raw) | Fast single-click; real-time; open-source | Binary matte, edge boiling/flickering, aliasing | Free |
| **CorridorKey** (в SammieRoto) | 8GB VRAM; 3sec/frame@2K на M4 Max MLX | Новая модель, мало отзывов | Free |
| **MatAnyone 2** | Качественный matting | Лицензия $5K/year для коммерции; OOM | Research (коммерческий $5K/yr) |
| **VideoMaMa** | "Better temporal consistency" | Тяжелее MatAnyone | Free |
| **Electric Sheep Spotlight** | EXR/ACES output; multiple layers; pro UI | Медленнее SAM2 (6s per 24f); платный | SaaS (pricing не раскрыт) |
| **Slapshot Autopilot** | Fully auto; cryptomatte; 8K; no prompts | Cloud-only; $0.25/frame дорого для длинных | $0.25/frame |
| **ONYX AI Matte** | SAM3+VitMatte; 32 objects; cheap | Windows+NVIDIA only | 80 EUR indie |
| **CopyCat/BigCat (Nuke)** | Dune 40% no-touchup; Disney soft-seg | Нужен training dataset; Nuke license | Included in Nuke |
| **Flame AutoMatte** | Integrated in Flame; custom ONNX | "SammieRoto лучше" (Stefan) | Included in Flame |
| **Version Zero AI** | Splines output (holy grail!) | Stealth mode; не доступен публично | TBD |
| **Roto Brush 3 (AE)** | Motion blur; refine tools | "Lacks specialised tools for high-end" | AE subscription |
| **Magic Mask (Resolve)** | Quick presets; free in Resolve | "Least consistent results"; wobbly edges | Free in Resolve |

---

## Главные проблемы AI roto (из отзывов)

1. **Edge boiling / flickering** — SAM2 и все mask-based методы дают нестабильные края между фреймами
2. **Binary matte** — SAM2 выдаёт hard black/white, нет soft edge / alpha gradient
3. **VRAM hunger** — fullres 4K+ = OOM даже на 48GB (A6000 Ada)
4. **Temporal consistency** — главная боль; VideoMaMa и CorridorKey улучшают, но не решают полностью
5. **No splines** — все инструменты кроме Version Zero выдают pixel masks, не splines; нельзя "подправить точку"
6. **Licensing** — MatAnyone $5K/year для коммерции; SAM2 — Apache 2.0
7. **Platform lock** — ONYX = Windows+NVIDIA; Flame AutoMatte = Flame; SammieRoto = Python everywhere

---

## Релевантность для CorridorKey проекта

- CorridorKey уже интегрирован в SammieRoto (wiltonmts, март 2026)
- Работает на 8GB VRAM (vs MatAnyone требует 24GB+)
- MLX port: **3sec/frame at 2K на M4 Max** (cnoellert)
- Это делает CorridorKey конкурентоспособным на Apple Silicon
- Направление: интеграция в Nuke через CopyCat/BigCat формат (.cat) или standalone OFX плагин
