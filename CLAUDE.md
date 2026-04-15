# CLAUDE.md — 20_CorridorKey

## Суть проекта
EZ-CorridorKey — open-source AI chroma keyer (нейросетевой кеер). Форк edenaion/EZ-CorridorKey.
Основа: модель GreenFormer (Hiera encoder + dual decoder для alpha + foreground).

## Git remotes
- `origin` → `cghunters/EZ-CorridorKey` (форк Игоря)
- `upstream` → `edenaion/EZ-CorridorKey` (оригинал)

## Установка на Mac (M2 Max) — ✅ ГОТОВО
- CorridorKey PyTorch (383 MB) + MLX (380 MB)
- SAM2 Base+, GVM (~6 GB), VideoMaMa (~10 GB fp16)
- FFmpeg 8.1, Desktop app (CorridorKey.app)
- 431 тест: 428 passed, 2 failed (MPS), 1 skipped

## CorridorKey-for-Nuke (плагин Mercell)
- GitHub: https://github.com/petermercell/CorridorKey-for-Nuke
- Тип: C++ NDK plugin + TensorRT inference
- **Требования:** Nuke 17, TensorRT 10.15, NVIDIA GPU 24GB+, ~128 GB RAM (ONNX export)
- **macOS:** ❌ невозможно (TensorRT = NVIDIA only)
- **Linux-машина (pop-os):** ⏸️ отложено — Nuke 16.0v4 (нужен 17), 64 GB RAM (нужно 128)

## Статус задач
- [x] Установка EZ-CorridorKey на Mac
- [x] Первый тест (DPX 2K, ARRI Alexa, зелёный хромакей)
- [x] DPX → EXR конвертер (OCIO, compositing_log → ACES 2065-1)
- [x] Исследование CorridorKey-for-Nuke плагина
- [ ] CorridorKey-for-Nuke: установить когда будет Nuke 17 + 128 GB RAM
- [x] CorridorKey-Runtime v0.7.3 OFX → DaVinci Resolve на Mac (CoreML, Apple Silicon)
- [ ] Альтернатива: свой Python Nuke node через MLX backend на Mac
- [x] Cryptomatte EXR генератор (Python, Nuke 16+ compatible)
- [x] Тест Cryptomatte EXR в Nuke 16.0v7 — РАБОТАЕТ (picker + matte)
- [x] Mega-research: auto-masking pipeline (15 агентов, 200+ источников)
- [x] AutoRoto pipeline v1-v4 на Linux RTX 3090 (YOLO/GroundingDINO + SAM2)
- [x] 9 шотов обработано (602+ кадров, DPX + EXR)
- [x] Temporal smoothing (edge flicker fix)
- [x] GroundingDINO scene description prompts
- [x] QC автоматический (drop detection, area jumps)
- [ ] Auto-describe: Claude описывает сцену → prompt для GroundingDINO
- [ ] ViTMatte post-processing (волосы, прозрачность)
- [ ] Batch mode (все шоты одной командой)
- [ ] Re-detection каждые N кадров (вход/выход объектов)
- [ ] Nuke Gizmo для импорта масок

## AutoRoto Pipeline
- Проект на Linux: `~/Desktop/AutoRoto/` (kostya@100.74.113.63)
- Зеркало на Mac: `~/Desktop/AutoRoto/`
- Scripts: process_shot.py (v6), process_shot_v2.py (v7), v3 (v8), v4 (v9)
- Исходники курса: `/mnt/Promise/WORK/PROJECTS/_СOMPOSE/Compositing Courses/IND2024/`

## Cryptomatte EXR Generator
- Скрипт: `scripts/generate_cryptomatte_exr.py`
- Спецификация: `_entity/cryptomatte_specification.md`
- Тестовый файл: `Output/test_cryptomatte.exr`
- Формат: Cryptomatte Spec v1.2.0 (Psyop), MurmurHash3_32, float32, ZIP compression
- Pure Python (без mmh3 C extension) — работает в любом окружении
