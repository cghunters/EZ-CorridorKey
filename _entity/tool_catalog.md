# Каталог AI-инструментов: Keying / Matting / Roto / Relighting

> Самодополняемая сущность. Агент обращается сюда для контекста.
> Последнее обновление: 2026-04-03

---

## Иерархия технологий

```
СЛОЙ 4 — ОРКЕСТРАЦИЯ / ИНФРАСТРУКТУРА
  ├── Griptape (теперь Foundry)  — AI-оркестрация, MCP, node-based, model-agnostic
  ├── Nuke CopyCat / BigCat      — тренировка ML-моделей внутри Nuke (1 шот / 100+ шотов)
  └── Cattery                    — каталог open-source нейросетей для Nuke (.cat формат)

СЛОЙ 3 — ПРОДУКТЫ (GUI, pipeline-интеграция)
  ├── EZ-CorridorKey             — desktop GUI над CorridorKey + все alpha-генераторы
  ├── Beeble Studio              — PBR passes + relighting из видео, Nuke plugin
  ├── Silhouette FX              — Boris FX, ручная рото + paint + Matte Refine ML
  ├── Mocha Pro 2026             — Boris FX, planar tracking + AI roto refine
  ├── ONYX AI Matte              — OFX plugin, SAM3 + VitMatte (Win+NVIDIA)
  ├── Kognat Rotobot             — OFX plugin, DL-сегментация в Nuke
  ├── After Effects Rotobrush 3  — Adobe Sensei
  ├── DaVinci Resolve Magic Mask — Blackmagic, ML-сегментация
  ├── Runway                     — облачный SaaS
  ├── Wonder Studio / Flow       — Autodesk, CG character replacement
  └── Nuke Stage                 — Foundry, virtual production (NAB 2025 Best of Show)

СЛОЙ 2 — СПЕЦИАЛИЗИРОВАННЫЕ ML-МОДЕЛИ
  ├── CorridorKey model          — Hiera + CNN refiner, "физическое размешивание" FG/BG
  ├── VideoMaMa                  — Adobe Research + KAIST, mask→matte через Stable Video Diffusion
  ├── GVM                        — Generative Video Matting, diffusion-based alpha
  ├── MatAnyone2                 — S-Lab (NTU) + SenseTime, универсальный маттинг
  ├── SwitchLight 3.0            — Beeble, video-to-PBR (7 passes)
  ├── BiRefNet                   — bilateral reference network, salient object
  ├── ViTMatte                   — trimap-based alpha matting (есть .cat для Nuke)
  ├── RVM                        — Robust Video Matting, real-time
  └── DiffMatte                  — diffusion matting

СЛОЙ 1 — ФУНДАМЕНТАЛЬНЫЕ МОДЕЛИ
  ├── SAM2 / SAM 2.1 / SAM3     — Meta, Segment Anything Model
  ├── DINOv2                     — Meta, self-supervised vision features
  └── Stable Video Diffusion     — Stability AI, видео-генерация (используется VideoMaMa)
```

---

## Полный каталог инструментов

### Open-source модели

| # | Инструмент | Разработчик | Задача | Архитектура | Лицензия | Качество волос | Нужен GS? | GPU | Демо |
|---|-----------|------------|--------|-------------|----------|---------------|-----------|-----|------|
| 1 | **CorridorKey** | Niko Pueringer (Corridor Digital) | AI keying зелёнки, раздельный FG RGB + alpha | Hiera backbone + CNN Refiner | CC BY-NC-SA 4.0 | ★★★★★ | Да | CUDA / MLX / MPS | [corridorkey.app](https://www.corridorkey.app/) |
| 2 | **VideoMaMa** | Adobe Research + KAIST (CVPR 2026) | Mask → pixel-accurate alpha matte | Stable Video Diffusion, single forward pass | CC BY-NC 4.0 | ★★★★★ | Нет | CUDA | [HF демо](https://huggingface.co/spaces/SammyLim/VideoMaMa) |
| 3 | **GVM** | AIM Lab, Uni of Adelaide (SIGGRAPH 2025) | Генеративный video matting | Diffusion UNet, spatio-temporal | CC BY-NC-SA 4.0 | ★★★★☆ | Заточен под GS | CUDA | [Project page](https://yongtaoge.github.io/project/gvm) |
| 4 | **MatAnyone2** | S-Lab (NTU) + SenseTime (CVPR 2026) | Универсальный video matting по промптам | Memory-based, pixel-level alpha | **Apache 2.0** ✅ | ★★★★☆ | Нет | CUDA | [HF демо](https://huggingface.co/spaces/PeiqingYang/MatAnyone) |
| 5 | **SAM2 / SAM 2.1** | Meta FAIR | Сегментация + трекинг объектов | Hiera encoder + memory attention | **Apache 2.0** ✅ | N/A (binary) | Нет | CUDA / MPS | [MetaDemoLab](https://sam2.metademolab.com/demo) |
| 6 | **BiRefNet** | Zheng Peng (CAAI AIR 2024) | Salient object segmentation | Bilateral reference network | **MIT** ✅ | ★★★☆☆ | Нет | CUDA | [HF демо](https://huggingface.co/spaces/ZhengPeng7/BiRefNet_demo) |
| 7 | **ViTMatte** | (academic) | Trimap → alpha matte | Vision Transformer matting | Open | ★★★★☆ | Нет | CUDA | Cattery (.cat для Nuke) |
| 8 | **RVM** | ByteDance | Real-time video matting | Lightweight recurrent | Open | ★★★☆☆ | Нет | CUDA | — |

### Коммерческие продукты

| # | Инструмент | Разработчик | Задача | ML внутри | OS | GPU | Цена | Демо / Ссылка |
|---|-----------|------------|--------|-----------|-----|-----|------|---------------|
| 9 | **Beeble Studio** | Beeble Inc. (Корея) | **7 PBR passes из видео + relighting** | SwitchLight 3.0 (custom UNet) | Win, Linux | NVIDIA RTX 30+ (12GB+) | $504/год (indie), $3000/год (studio) | [beeble.ai](https://beeble.ai/), [VFX Passes](https://app.beeble.ai/vfx-passes) |
| 10 | **Beeble Cloud** | Beeble Inc. | То же, облачно (2K, 8-bit) | SwitchLight 3.0 | Браузер | Облако | Free 90 credits, $19–75/мес | [app.beeble.ai](https://app.beeble.ai/vfx-passes) |
| 11 | **Silhouette 2025.5** | Boris FX | Ручная рото + paint + **Matte Refine ML** | Proprietary + Matte Refine ML | Win, Mac, Linux | NVIDIA / CPU | $1,695 / $595/год | [borisfx.com](https://borisfx.com/products/silhouette/) |
| 12 | **Mocha Pro 2026** | Boris FX | Planar tracking + **AI roto refine** + 3D solve | Matte Refine ML | Win, Mac, Linux | NVIDIA / CPU | $995 / $295/год | [borisfx.com](https://borisfx.com/products/mocha-pro/) |
| 13 | **ONYX AI Matte** | Evgeniy Shatskiy | Auto-matte из видео, 5 режимов, text prompts | **SAM3 + VitMatte** (ONNX) | **Windows only** | **NVIDIA RTX 2060+** | **€80** perpetual | [onyxofx.com](https://onyxofx.com/) |
| 14 | **Kognat Rotobot** | Kognat | DL-сегментация в Nuke | Semantic/instance segmentation | Win, Linux | NVIDIA | Коммерческий | [kognat.com](https://kognat.com/) |
| 15 | **Rotobrush 3.0** | Adobe | ML-рото в After Effects | Adobe Sensei | Win, Mac | CUDA / Metal | CC подписка | — |
| 16 | **DaVinci Resolve Magic Mask** | Blackmagic | ML-сегментация для грейдинга | Proprietary | Win, Mac, Linux | CUDA / Metal | Free / $295 | — |
| 17 | **Runway** | Runway AI | Background removal, генерация | Proprietary | Браузер | Облако | $12–76/мес | [runway.ml](https://runway.ml/) |
| 18 | **Wonder Studio / Flow** | Autodesk | CG character replacement | Proprietary | Браузер | Облако | Autodesk pricing | [wonderdynamics.com](https://wonderdynamics.com/) |

### Foundry / Nuke экосистема

| # | Инструмент | Что делает | Статус | Версия |
|---|-----------|-----------|--------|--------|
| 19 | **CopyCat** | Тренировка ML на паре before/after → применение на секвенцию | Выпущен | NukeX 13+ |
| 20 | **BigCat** | CopyCat на масштабе десятков/сотен шотов, custom loss (LPIPS) | **Выпущен** | **NukeX 17** (фев 2026) |
| 21 | **Cattery** | Каталог .cat нейросетей (ViTMatte, DepthPro, RIFE, MiDas, LPIPS) | Выпущен, пополняется | NukeX 16+ |
| 22 | **AI-assisted Roto** | Автоматическое движение/деформация spline roto по секвенции | **Coming soon** ⏳ | Дата не объявлена |
| 23 | **Griptape** | AI-оркестрация, MCP, подключение любых моделей к pipeline | **Приобретён Foundry** (фев 2026) | Интеграция идёт |
| 24 | **Nuke Stage** | Virtual production, ICVFX, LED walls | Выпущен | NAB 2025 Best of Show |
| 25 | **SmartROTO** | Research: AI roto совместно с DNEG + Uni of Bath | Исследование | Далеко от продакшна |
| 26 | **CorridorKey-for-Nuke** | Нативный Nuke plugin для CorridorKey | Выпущен | [GitHub](https://github.com/petermercell/CorridorKey-for-Nuke) |

---

## Beeble — подробный разбор

### Что генерирует VFX Pass Generator (7 passes):

| Pass | Описание |
|------|----------|
| **Alpha** | Foreground matte (ротоскопия) |
| **Depth** | Normalized depth map |
| **Normal** | Surface normals (object space, OpenGL) |
| **Base Color** | Diffuse/albedo без освещения |
| **Roughness** | Шероховатость поверхности |
| **Specular** | Блики и отражения |
| **Metallic** | Металлические свойства |

### SwitchLight 3.0 — технология:
- **CVPR 2024 paper**: physics-driven архитектура с Cook-Torrance specular reflectance
- Модульная UNet из 5 сетей: Normal → Illum → Diffuse → Specular → Render
- **Собственный LightStage** для training data (287 субъектов, 137 LED, 7 камер)
- v3.0 — true end-to-end video model (не per-frame), temporal consistency встроена

### Nuke plugin Beeble:
- Nuke 14.1v2+
- Ноды: PBRPacker, PBRController, DirectionalLight, PointLight, EnvironmentLight
- 16-bit multi-channel EXR (из Studio)
- Workflow: Load PBR → PBRPacker → PBRController → Add Lights → Merge → Output

### Production proof:
- **Superman & Lois Season 4** (HBO Max) — сотни шотов за 3 недели, Boxel Studio
- Relighting глаз Superman (laser vision), subsurface glow

### Beeble Camera (iOS):
- Бесплатное приложение
- Camera tracking через LiDAR + ARKit
- Загрузка в Beeble Cloud

### Ограничения:
- Cloud: max 2K, 8-bit PNG, 60 сек — не production grade
- Studio: нужен NVIDIA RTX 30+ с 12GB+ VRAM
- macOS НЕ поддерживается для Studio
- Сильные тени, отражающие поверхности (очки), face paint — проблемные

---

## Foundry Griptape — почему это важно

- **Jason Schleifer** (бывший Head of Character Animation в DreamWorks) + **Kyle Roche** (бывший GM в AWS)
- **Model-agnostic**: подключает Google, OpenAI, Black Forest Labs, ElevenLabs, Topaz и др.
- **MCP поддержка** — тот же протокол, что в нашем Nuke MCP
- **USD MCP Server** — работа с OpenUSD через AI-агентов
- **Node-based интерфейс** — художники строят AI-пайплайны без кода
- **On-premises** — для студий с жёсткими требованиями безопасности
- Стратегия Foundry: "AI-first pipeline" — не заменять художников, а дать им оркестрацию множества AI-моделей

---

## Лицензии — сводка для продакшна

| Модель | Лицензия | Коммерция | Примечание |
|--------|----------|-----------|-----------|
| CorridorKey | CC BY-NC-SA 4.0 | ❌ | Блокер для студий |
| VideoMaMa | CC BY-NC 4.0 + Stability AI | ❌ | Adobe Research, но NC |
| GVM | CC BY-NC-SA 4.0 | ❌ | |
| MatAnyone2 | Apache 2.0 | ✅ | Единственный open-source matting с коммерцией |
| SAM2/SAM3 | Apache 2.0 | ✅ | Foundation model |
| BiRefNet | MIT | ✅ | |
| ViTMatte | Open | ✅ | |
| Griptape Framework | Apache 2.0 | ✅ | Open-source часть Foundry |

---

## Ссылки — полный реестр

→ См. [demo_links.md](demo_links.md) — отдельная сущность со всеми URL

---

*Формат: самодополняемая сущность. При обнаружении нового инструмента — добавлять в соответствующую секцию.*
*Обновлено: 2026-04-03*
