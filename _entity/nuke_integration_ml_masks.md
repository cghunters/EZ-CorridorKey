# Доставка ML масок в Nuke — полное руководство

> Deep research, 2026-04-04
> Pipeline: auto-segmentation на Linux RTX 3090 → Nuke 16/17 на Mac M2 Max

---

## 1. ФОРМАТЫ ВЫВОДА МАСОК

### 1.1 Multi-channel EXR (РЕКОМЕНДУЕМЫЙ ПУТЬ для начала)

Каждый объект = отдельный именованный канал. Nuke читает через Shuffle2.

**Формат каналов:** `mask_person.alpha`, `mask_car.alpha`, `mask_sky.alpha`

**Рабочий код — SAM2 logits → soft alpha → 16-bit EXR:**

```python
import numpy as np
import torch
import OpenEXR
import Imath

# === SAM2 logits → soft alpha ===
def logits_to_soft_alpha(logits_tensor, temperature=0.8):
    """
    temperature < 1.0 = жёсткие края
    temperature > 1.0 = мягкие переходы (для волос, дыма)
    """
    soft = torch.sigmoid(logits_tensor / temperature)
    if soft.dim() > 2:
        soft = soft.squeeze()
    return soft.cpu().numpy().astype(np.float32)

# === Запись multi-channel 16-bit EXR ===
def write_masks_exr(filepath, masks_dict, width, height):
    """
    masks_dict: {"person": np.array(H,W), "car": np.array(H,W)}
    """
    header = OpenEXR.Header(width, height)
    half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))

    channels_spec = {}
    channels_data = {}

    for name, mask in masks_dict.items():
        chan_key = f"mask_{name}.alpha"
        channels_spec[chan_key] = half_chan
        channels_data[chan_key] = mask.astype(np.float16).tobytes()

    header['channels'] = channels_spec
    header['compression'] = Imath.Compression(Imath.Compression.PIZ_COMPRESSION)

    exr = OpenEXR.OutputFile(filepath, header)
    exr.writePixels(channels_data)
    exr.close()
```

**В Nuke:**
```
Read (shot010_masks.####.exr) → Shuffle2 (mask_person.alpha → rgba.alpha) → Premult → Merge
```

### 1.2 Cryptomatte — ID-маски с anti-aliasing

Каждый пиксель: float32 ID (MurmurHash3) + coverage. Nuke Cryptomatte node — кликаешь на объект в Viewer, получаешь маску.

**Код генерации:**

```python
import numpy as np
import struct
import json
import mmh3  # pip install mmh3
import OpenEXR
import Imath

def mm3hash_float(name: str) -> float:
    """MurmurHash3 → float32 (стандарт Cryptomatte)"""
    hash_32 = mmh3.hash(name)
    exp = (hash_32 >> 23) & 255
    if exp == 0 or exp == 255:
        hash_32 ^= 1 << 23
    packed = struct.pack('<I', hash_32 & 0xFFFFFFFF)
    return struct.unpack('<f', packed)[0]

def create_cryptomatte_exr(filepath, masks_dict, width, height,
                           layer_name="CryptoObject"):
    """
    masks_dict: {"person": np.array(H,W, float32 0-1), ...}
    """
    header = OpenEXR.Header(width, height)
    float_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
    header['compression'] = Imath.Compression(Imath.Compression.ZIPS_COMPRESSION)

    manifest = {}
    object_ids = {}
    for name in masks_dict:
        fid = mm3hash_float(name)
        manifest[name] = format(struct.unpack('<I',
            struct.pack('<f', fid))[0], '08x')
        object_ids[name] = fid

    # Build channels (ID, coverage pairs)
    names = list(masks_dict.keys())
    all_masks = np.stack([masks_dict[n] for n in names], axis=0)
    all_ids = np.array([object_ids[n] for n in names], dtype=np.float32)
    sort_idx = np.argsort(-all_masks, axis=0)

    num_levels = min(6, (len(names) + 1) // 2)
    channels_spec = {}
    channels_data = {}

    for level in range(num_levels):
        level_name = f"{layer_name}{level:02d}"
        for sub, suffix in enumerate(["red", "green", "blue", "alpha"]):
            chan_name = f"{level_name}.{suffix}"
            channels_spec[chan_name] = float_chan
            pair_idx = level * 2 + sub // 2
            is_coverage = (sub % 2 == 1)

            if pair_idx < len(names):
                data = np.zeros((height, width), dtype=np.float32)
                for y in range(height):
                    for x in range(width):
                        obj_rank = sort_idx[pair_idx, y, x]
                        data[y, x] = all_masks[obj_rank, y, x] if is_coverage \
                            else all_ids[obj_rank]
                channels_data[chan_name] = data.tobytes()
            else:
                channels_data[chan_name] = np.zeros((height, width),
                    dtype=np.float32).tobytes()

    header['channels'] = channels_spec

    import hashlib
    layer_hash = hashlib.md5(layer_name.encode()).hexdigest()[:7]
    header[f'cryptomatte/{layer_hash}/name'] = layer_name
    header[f'cryptomatte/{layer_hash}/hash'] = 'MurmurHash3_32'
    header[f'cryptomatte/{layer_hash}/conversion'] = 'uint32_to_float32'
    header[f'cryptomatte/{layer_hash}/manifest'] = json.dumps(manifest)

    exr = OpenEXR.OutputFile(filepath, header)
    exr.writePixels(channels_data)
    exr.close()
```

**ВАЖНО:** Cryptomatte = только 32-bit float (вдвое больше места чем half EXR).

### 1.3 Сравнительная таблица форматов

| Формат | Anti-aliasing | Nuke support | Bit depth | Лучше для |
|--------|:---:|:---:|:---:|:---:|
| **Multi-ch EXR** | Зависит от маски | Shuffle2 | half/float | 1-10 объектов |
| **Cryptomatte** | Да (native) | Cryptomatte node | float32 only | 10-100+ объектов |
| **ObjectId / Deep EXR** | Да (per-sample) | Кастомный plugin | uint32/float | Production, 100+ |
| **PNG sequence** | Нет (8-bit) | Read → Shuffle | 8/16 bit int | Черновик |

---

## 2. NUKE NODES И ПЛАГИНЫ

### Foundry Cattery (встроенный ML inference)

| Модель | Задача | URL |
|--------|--------|-----|
| **ViTMatte** | Alpha matting из garbage matte | [Cattery](https://community.foundry.com/cattery) |
| **Segment Anything** | Сегментация по точкам | [rafaelperez](https://github.com/rafaelperez/Segment-Anything-for-Nuke) |
| **Depth Anything V2** | Depth maps | [Cattery](https://community.foundry.com/cattery) |

### Segment Anything for Nuke (Rafael Perez)
- [GitHub](https://github.com/rafaelperez/Segment-Anything-for-Nuke) — SAM1 в Cattery .cat формат
- Nuke 13.2+, NVIDIA 4GB+
- До 8 точек через Tracker
- **SAM1 only** (не SAM2!)

### ViTMatte for Nuke (рефайн краёв)
- [GitHub](https://github.com/rafaelperez/ViTMatte-for-Nuke)
- Принимает garbage matte → вытягивает волосы, мех, полупрозрачность
- <1 sec/frame 2K

### Nuke-ML-Server (client-server)
- [GitHub](https://github.com/TheFoundryVisionmongers/nuke-ML-server)
- Client в Nuke → TCP → Python сервер с PyTorch
- Можно воткнуть ЛЮБУЮ модель (SAM2, CorridorKey)
- Сервер на Linux GPU, Nuke на Mac

---

## 3. PIPELINE АВТОМАТИЗАЦИЯ

### Архитектура

```
LINUX RTX 3090              SHARED STORAGE           MAC M2 MAX
SAM2 + GroundingDINO  ───→  /mnt/ftp2/masks/  ───→  Nuke 16/17
Python pipeline              EXR sequences            Read → Comp
```

### Batch-процессинг скрипт (Linux)

```bash
ssh kostya@100.74.113.63
python3 mask_pipeline.py \
  --frames /mnt/ftp2/projects/VIM/shots/sh010/plates/ \
  --output /mnt/ftp2/projects/VIM/shots/sh010/masks/ \
  --prompts prompts_sh010.json \
  --model large
```

---

## 4. КАЧЕСТВО МАСОК ДЛЯ COMPOSITING

### Binary vs Soft Alpha

```python
# ПЛОХО (бинарная):
mask = (logits > 0.5).float()  # 0 или 1, ступеньки

# ХОРОШО (soft alpha):
soft = torch.sigmoid(logits / 0.8)  # мягкие переходы
```

### Улучшение краёв

| Метод | Качество | Скорость |
|-------|:---:|:---:|
| `sigmoid(logits/T)` | Хорошее | Мгновенно |
| **ViTMatte refinement** | Отличное | <1 сек 2K |
| Gaussian blur | Среднее | Быстро |
| EdgeBlur (Nuke) | Среднее | Быстро |

**Рекомендуемый pipeline:**
```
SAM2 (soft alpha, T=0.8) → EXR → Nuke → ViTMatte → Final matte
```

### Bit depth

| Формат | Подходит для |
|--------|:---:|
| 8-bit PNG | Превью |
| **16-bit half EXR** | **Оптимум для масок** |
| 32-bit float EXR | Cryptomatte (обязательно) |

---

## 5. РЕКОМЕНДАЦИИ

### Вариант A: Простой (начать с него)
```
Linux: SAM2 → sigmoid → multi-ch 16-bit EXR → /mnt/ftp2/
Mac:   Read → Shuffle2 → ViTMatte → Premult → Merge
```

### Вариант B: Production (Cryptomatte)
```
Linux: SAM2 + GDINO → MurmurHash3 → Cryptomatte 32-bit EXR
Mac:   Read → Cryptomatte node → клик = маска → ViTMatte → Comp
```

### Вариант C: Локальный (Mac only)
```
Mac: Read → Segment Anything (Cattery .cat) → ViTMatte → Comp
```
SAM1 only, нет video propagation, но всё локально.

---

## Sources

- [Psyop/Cryptomatte](https://github.com/Psyop/Cryptomatte)
- [rafaelperez/Segment-Anything-for-Nuke](https://github.com/rafaelperez/Segment-Anything-for-Nuke)
- [rafaelperez/ViTMatte-for-Nuke](https://github.com/rafaelperez/ViTMatte-for-Nuke)
- [TheFoundryVisionmongers/nuke-ML-server](https://github.com/TheFoundryVisionmongers/nuke-ML-server)
- [MercenariesEngineering/openexrid](https://github.com/MercenariesEngineering/openexrid)
- [Foundry Cattery](https://community.foundry.com/cattery)
- [OpenEXR Deep IDs Specification](https://openexr.com/en/latest/DeepIDsSpecification.html)
- [Image Engine DigiPro 2025](https://arxiv.org/abs/2507.07242)
