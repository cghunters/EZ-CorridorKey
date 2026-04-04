# МЕГА-РЕСЕРЧ: Auto-Masking Pipeline для VFX — Сводный отчёт

> Дата: 2026-04-04
> 15 агентов, 3 волны исследований, 200+ источников
> Задача: как VFX-студии автоматизируют batch-сегментацию без ручного промптинга

---

## EXECUTIVE SUMMARY

**Ответ на главный вопрос:** Студии используют **детектор** (GroundingDINO / Florence-2) + **сегментатор** (SAM2) + **трекер** (SAM2 Video). Один текстовый промпт `"person."` на весь батч — дальше pipeline работает полностью автоматически. Image Engine (Vancouver) задеплоила это в продакшн: 12 шоу, 1,241 шот, ручные правки только в 10-15% случаев.

**Лучший open-source repo:** [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) — 3.4k stars, Apache 2.0.

**Ключевые открытия ресерча:**
1. **EfficientTAM** — Apache 2.0, MPS support, 20x легче SAM2 → tracker для Mac
2. **YOLOE** — 306 FPS, zero-prompt detection+segmentation → замена GroundingDINO
3. **OmnimatteZero** — извлекает объект ВМЕСТЕ С ТЕНЯМИ И ОТРАЖЕНИЯМИ
4. **SAM 3.1 Object Multiplex** — 7x speedup при 128 объектах, 32 FPS на H100
5. **Version Zero AI** — "Holy Grail": AI roto с выходом в splines (stealth mode)
6. **TensorRT на RTX 3090 быстрее чем vanilla PyTorch на RTX 5090** — software optimization > hardware upgrade

---

## 1. АРХИТЕКТУРА PIPELINE

```
КАДРЫ (PNG/EXR/DPX)
    │
    ▼
┌──────────────────┐   text prompt: "person. car." (1 раз)
│  DETECTOR         │
│  GroundingDINO /  │──→ bounding boxes + labels
│  YOLOE / Florence │
└──────────────────┘
    │
    ▼  boxes как prompts
┌──────────────────┐
│  SAM 2 / SAM 2.1 │──→ pixel-perfect masks per instance
│  Image Predictor  │
└──────────────────┘
    │
    ▼  masks + object IDs
┌──────────────────┐
│  SAM 2 Video      │──→ propagation через все кадры
│  Predictor        │    forward + reverse
└──────────────────┘
    │
    ▼
┌──────────────────┐
│  POST-PROCESSING  │──→ person_1.exr, person_2.exr, car_1.exr
│  IoU matching     │    ObjectId / Cryptomatte EXR
│  ID continuity    │
└──────────────────┘
```

---

## 2. PRODUCTION CASE: IMAGE ENGINE (DigiPro 2025)

| Параметр | Значение |
|----------|----------|
| Paper | [arXiv:2507.07242](https://arxiv.org/abs/2507.07242) \| [ACM DL](https://dl.acm.org/doi/10.1145/3744199.3744635) |
| Стек | GroundingDINO → SAM2 → SAM2 Video Tracking |
| GPU | NVIDIA A4000 16GB, Singularity containers |
| Скорость | ~10 frames/min (~6 sec/frame) |
| Масштаб | 12 шоу, 1,241 шот |
| Ручные правки | 10-15% шотов |
| Выход | ObjectId EXR → Nuke |
| Авто-триггер | При инджесте плейтов |

### Ключевые параметры из paper
- Asymmetric overlap: `Sim(A,B) = |A∩B| / |A|`
- False positive removal: ≥2 overlaps при threshold 0.1
- Merge: overlap ≥0.6
- Tracking step: 5 кадров propagation, re-detection каждые 20 кадров
- IoU direct match: 0.9

---

## 3. ВЫБОР ДЕТЕКТОРА

| Детектор | Точность | Скорость | VRAM | Mac MPS | Zero-prompt? | License |
|----------|----------|----------|------|---------|:---:|---------|
| **GroundingDINO 1.0** | 52.5 AP | ~5 FPS (A100) | ~8 GB | CPU only | text prompt | Apache 2.0 |
| **DINO-X Pro** | 56.0 AP | ~3 FPS | ~12 GB | API | text / prompt-free | Cloud API |
| **Florence-2-large** | ~45 AP | ~8 FPS | ~4 GB | CPU/MPS | text prompt | MIT |
| **YOLOE** | 26 AP (prompt-free) | **306 FPS** | 2-4 GB | CoreML | **ДА** | AGPL-3.0 |
| **YOLO-World-L** | ~45 AP | 30+ FPS | 2-4 GB | MPS | text prompt | GPL-3.0 |
| **YOLOv8x** | высокая (COCO) | 30+ FPS | 2-4 GB | **MPS** | closed-set | AGPL-3.0 |

**Для Linux RTX 3090:** GroundingDINO 1.0 (лучшая точность open-source)
**Для Mac M2 Max:** YOLOv8x (полная MPS) или Florence-2 (MPS с фиксом)

---

## 4. SAM2 ALTERNATIVES — НОВЫЕ МОДЕЛИ

### SAM 3.1 Object Multiplex (март 2026)
- 848M параметров, unified detector+tracker
- Text prompt → все instances + tracking **в одной модели**
- 4M+ concepts, 7x speedup при 128 объектах
- VRAM: sub-4 GB (!), меньше чем GroundingDINO + SAM2
- **НО:** CUDA 12.6 + Triton only. Mac = невозможно (workaround: SAM3_CPU fork)

### EfficientTAM (ICCV 2025)
- **Apache 2.0 + MPS support** — единственный production tracker для Mac
- 20x легче SAM, 1.6x быстрее, ~10 FPS на iPhone 15
- Semi-supervised (needs first-frame mask)
- [GitHub](https://github.com/yformer/EfficientTAM)

### YOLOE (ICCV 2025)
- **306 FPS** prompt-free detection + segmentation
- Built-in vocabulary, не нужен language model
- [GitHub](https://github.com/THU-MIG/yoloe) — AGPL-3.0

### OmnimatteZero (SIGGRAPH Asia 2025)
- Извлекает объект **ВМЕСТЕ С ТЕНЯМИ И ОТРАЖЕНИЯМИ** — training-free
- 25 FPS на A100, но **32 GB+ VRAM**
- [GitHub](https://github.com/dvirsamuel/OmnimatteZero)

---

## 5. GPU BENCHMARKS — РЕАЛЬНЫЕ ЦИФРЫ

### SAM2 Video Tracking

| GPU | Model | FPS | Source |
|-----|-------|-----|--------|
| **RTX 3090** | Hiera-Tiny | ~25-26 | [GitHub #448](https://github.com/facebookresearch/sam2/issues/448) |
| **A100** | Hiera-B+ | 43.8 | [GitHub #159](https://github.com/facebookresearch/sam2/issues/159) |
| **A100** | Hiera-Large | 30.2 | [GitHub #159](https://github.com/facebookresearch/sam2/issues/159) |
| **A100** (без torch.compile) | video | ~21 | [GitHub #159](https://github.com/facebookresearch/sam2/issues/159) |

### VRAM (SAM2 Video, 1024x1024, 1 объект)

| Model | VRAM |
|-------|------|
| Tiny | 1.2 GB |
| Base+ | 1.3 GB |
| Large | 1.7 GB |

### GPU Upgrade Decision

| GPU | Inference ratio | VRAM | Memory BW | MSRP |
|-----|:---:|:---:|:---:|:---:|
| **RTX 3090** | 1.0x | 24 GB | 936 GB/s | ~$800 б/у |
| **RTX 4090** | ~1.5x | 24 GB | 1,008 GB/s | ~$1,600 |
| **RTX 5090** | ~2.2x | **32 GB** | **1,792 GB/s** | ~$2,000 |

### Оптимизации (от самых простых к сложным)

| Техника | Speedup | Качество | Сложность |
|---------|---------|----------|-----------|
| **torch.compile** | **2x** | 100% | 1 строка кода |
| FP16 / BF16 | 1.3-1.5x | ~99.9% | Низкая |
| Flash Attention 2 | 2-4x (attention) | 100% | Средняя |
| **TensorRT** | **3-4x** | ~99% | Высокая (C++) |
| image_size 512 | **4x** | ~90-95% | 1 параметр |

**Вывод: TensorRT на RTX 3090 ≈ vanilla PyTorch на RTX 5090.** Software optimization first.

---

## 6. COMMERCIAL SOLUTIONS — TOP-10

| # | Продукт | Auto level | Nuke | Mac | Цена | Что делает |
|---|---------|-----------|:---:|:---:|------|-----------|
| 1 | **Slapshot Autopilot** | Полный автомат | Export Cryptomatte | Web | $0.25/frame | Zero-input, cloud, до 8K |
| 2 | **ONYX AI Matte** | Text → mask | OFX | ❌ Win | €80 | SAM3+VitMatte, OFX plugin |
| 3 | **Silhouette 2025.5** | Text → mask (Mask ML) | OFX | ✅ | $1,195+ | Boris FX, industry standard |
| 4 | **Mocha Pro 2026** | Click → splines | OFX | ✅ | $295/yr | Planar tracking + AI roto |
| 5 | **Beeble** | Upload → mask+PBR | Plugin | Cloud | $19/mo | 7 PBR passes + relighting |
| 6 | **Continuum 2026** | Auto face masks | OFX | ✅ | $325/yr | Face ML + Matte Refine |
| 7 | **Sapphire 2025.5** | One-click + track | OFX | ✅ | $545/yr | Object Brush ML |
| 8 | **DaVinci Magic Mask** | Click → track | ❌ | ✅ | $295 once | Grading isolation |
| 9 | **Flow Studio** | Auto (CG replace) | Export | Web | $10-95/mo | Autodesk, CG characters |
| 10 | **Kognat Rotobot** | Auto by class | OFX | ✅ | $130/yr | 81 categories, flickery |

---

## 7. OSINT — РЕАЛЬНЫЕ ОТЗЫВЫ

### Что работает
- **Image Engine**: 1,241 шот автоматически, 10-15% правок — [DigiPro 2025](https://arxiv.org/abs/2507.07242)
- **CopyCat (Dune Part Two)**: 40% из 1000 eye shots = zero touchups — [Foundry](https://www.foundry.com/insights/machine-learning/untapped-potential-ml-vfx)
- **SammieRoto**: open-source бьёт Flame AutoMatte — [Logik Forums](https://forum.logik.tv/t/matanyone-using-sammiroto/12797)
- **CorridorKey MLX**: 3 sec/frame 2K на M4 Max — Logik Forums

### Что НЕ работает
- **SAM2 края**: binary matte, flickering/boiling — [Electric Sheep](https://blog.electricsheep.tv/we-tested-sam2-for-rotoscoping-this-is-what-we-found/)
- **4K+ разрешение**: OOM даже на A6000 48GB при fullres 4.5K
- **Temporal consistency**: главная жалоба — маски "дышат" между кадрами

### Holy Grail
- **Version Zero AI**: AI roto с выходом в **splines** (не pixel masks) — [beforesandafters](https://beforesandafters.com/2025/05/20/version-zero-ai-has-a-splines-output-solution-for-ai-ml-rotoscoping/)
- Cofounders: Chad Wanstreet + Allan McKay. Stealth mode.

---

## 8. РЕКОМЕНДУЕМЫЕ PIPELINES

### Pipeline A: Linux RTX 3090 (production, сейчас)
```
GroundingDINO 1.0 ("person.") → SAM2.1 Large → Video Tracking
  + torch.compile + FP16 → ~50 FPS
  → ObjectId/Cryptomatte EXR → Nuke
```

### Pipeline B: Mac M2 Max (preview/dev)
```
YOLOE prompt-free OR YOLOv8x ("person" class)
  → EfficientTAM (MPS, Apache 2.0)
  → per-instance PNG/EXR masks
```

### Pipeline C: Future Linux (SAM3, когда CUDA 12.6)
```
SAM 3.1 ("person." text prompt)
  → Object Multiplex (detect + segment + track в одной модели)
  → sub-4 GB VRAM, 32 FPS
```

### Pipeline D: VFX Layer Decomposition (heavy, A100/5090)
```
SAM3 / Grounded-SAM-2 (object masks)
  → OmnimatteZero (FG + shadows + reflections)
  → Nuke composite
```

---

## 9. КЛЮЧЕВЫЕ РЕПОЗИТОРИИ

| Repo | Stars | License | Задача | URL |
|------|:---:|---------|--------|-----|
| **Grounded-SAM-2** | 3.4k | Apache 2.0 | Full pipeline | [GitHub](https://github.com/IDEA-Research/Grounded-SAM-2) |
| **SAM 3** | 8.7k | SAM License | Next-gen segmentation | [GitHub](https://github.com/facebookresearch/sam3) |
| **EfficientTAM** | ~1k | Apache 2.0 | Lightweight tracker (MPS!) | [GitHub](https://github.com/yformer/EfficientTAM) |
| **YOLOE** | ~2k | AGPL-3.0 | Zero-prompt detect+seg | [GitHub](https://github.com/THU-MIG/yoloe) |
| **Det-SAM2** | ~500 | MIT | Periodic re-detection | [GitHub](https://github.com/motern88/Det-SAM2) |
| **DEVA** | 1.5k | Custom | True zero-prompt video | [GitHub](https://github.com/hkchengrex/Tracking-Anything-with-DEVA) |
| **SAM-Track** | 3.1k | AGPL-3.0 | Auto detect + track | [GitHub](https://github.com/z-x-yang/Segment-and-Track-Anything) |
| **OmnimatteZero** | new | — | FG + shadows extraction | [GitHub](https://github.com/dvirsamuel/OmnimatteZero) |
| **sam2_trt_inference** | ~200 | — | TensorRT for SAM2 | [GitHub](https://github.com/tier4/sam2_trt_inference) |

---

## 10. ИСТОЧНИКИ

### Papers
- Image Engine DigiPro 2025: [arXiv:2507.07242](https://arxiv.org/abs/2507.07242)
- SAM2: [arXiv:2408.00714](https://arxiv.org/abs/2408.00714)
- SAM3: [arXiv:2511.16719](https://arxiv.org/abs/2511.16719)
- EfficientTAM: [arXiv:2411.18933](https://arxiv.org/abs/2411.18933)
- YOLOE: [arXiv:2503.07465](https://arxiv.org/abs/2503.07465)
- Efficient-SAM2 (ICLR 2026): [arXiv:2602.08224](https://arxiv.org/abs/2602.08224)
- OmnimatteZero: [GitHub](https://github.com/dvirsamuel/OmnimatteZero)

### Forums & Blogs
- Logik Forums SammieRoto: [forum.logik.tv](https://forum.logik.tv/t/matanyone-using-sammiroto/12797)
- Logik Forums ONYX: [forum.logik.tv](https://forum.logik.tv/t/onix-ai-matte/14192)
- Electric Sheep SAM2 test: [blog](https://blog.electricsheep.tv/we-tested-sam2-for-rotoscoping-this-is-what-we-found/)
- Foundry CopyCat Dune: [foundry.com](https://www.foundry.com/insights/machine-learning/untapped-potential-ml-vfx)
- Version Zero AI: [beforesandafters.com](https://beforesandafters.com/2025/05/20/version-zero-ai-has-a-splines-output-solution-for-ai-ml-rotoscoping/)

### GitHub Issues (benchmarks)
- SAM2 VRAM: [#118](https://github.com/facebookresearch/sam2/issues/118)
- SAM2 FPS: [#159](https://github.com/facebookresearch/sam2/issues/159), [#448](https://github.com/facebookresearch/sam2/issues/448)
- SAM2 optimization: [#543](https://github.com/facebookresearch/sam2/issues/543)
- TIER IV TensorRT: [blog](https://medium.com/tier-iv-tech-blog/high-performance-sam2-inference-framework-with-tensorrt-9b01dbab4bf7)
