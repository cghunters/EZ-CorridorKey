# Cryptomatte Specification — Справочник для генерации EXR

> Источник: [Psyop/Cryptomatte](https://github.com/Psyop/Cryptomatte) specification v1.2.0
> Дата исследования: 2026-04-03

## TL;DR — Что нужно для работающего Cryptomatte EXR в Nuke 16

1. **Channels**: `CryptoObject00.R/G/B/A`, `CryptoObject01.R/G/B/A`, `CryptoObject02.R/G/B/A` — ВСЕ FLOAT32 (не HALF!)
2. **Metadata**: 4 обязательных ключа в header
3. **ID encoding**: MurmurHash3_32 → avoid denormals → reinterpret as float32
4. **Channel layout**: R=ID1, G=coverage1, B=ID2, A=coverage2 (пары)
5. **Manifest**: JSON dict `{"name": "hex_id"}`
6. **Compression**: ZIP/ZIPS/PIZ — любая ок

## Формат каналов (Channel Naming)

```
{LayerName}{NN}.{R|G|B|A}
```

- `LayerName` = имя слоя (`CryptoObject`, `CryptoMaterial`, `CryptoAsset`)
- `NN` = двузначный номер level (00, 01, 02, ...)
- Каждый level хранит 2 пары ID-coverage:
  - `.R` = float ID первого объекта (rank 2*N)
  - `.G` = coverage первого объекта
  - `.B` = float ID второго объекта (rank 2*N+1)
  - `.A` = coverage второго объекта

Типичная конфигурация — 3 levels (00-02) = 6 ranks. Покрывает до 6 перекрывающихся объектов на пиксель.

## Metadata Keys (обязательные)

В EXR header должны быть 4 ключа:

```
cryptomatte/{layer_hash}/name       = "CryptoObject"
cryptomatte/{layer_hash}/hash       = "MurmurHash3_32"
cryptomatte/{layer_hash}/conversion = "uint32_to_float32"
cryptomatte/{layer_hash}/manifest   = '{"obj1":"a1b2c3d4","obj2":"e5f6a7b8"}'
```

- `{layer_hash}` = 7-символьный hex ID слоя
- Вычисляется как: `id_to_hex(mm3hash_float(layer_name))[:-1]`
- Пример: `CryptoObject` → `3ae39a5`

### Nuke и metadata prefix

- В EXR файле: `cryptomatte/HASH/key` (без prefix)
- Nuke при чтении добавляет `exr/` → `exr/cryptomatte/HASH/key`
- Psyop-плагин ищет ОБА: `["exr/cryptomatte/", "cryptomatte/"]`
- Nuke 16 built-in node тоже работает с обоими

## Hash (ID) Encoding

### MurmurHash3_32 → float32

```python
def mm3hash_float(name):
    hash_32 = murmur3_32(name)          # MurmurHash3_x86_32
    exp = (hash_32 >> 23) & 255
    if exp == 0 or exp == 255:           # avoid denormals и NaN/Inf
        hash_32 ^= (1 << 23)
    packed = struct.pack('<I', hash_32)  # uint32 little-endian
    return struct.unpack('<f', packed)[0] # reinterpret as float32
```

### Float → Hex (для manifest)

```python
def id_to_hex(float_id):
    return "{0:08x}".format(struct.unpack('<I', struct.pack('<f', float_id))[0])
```

## Manifest Format

JSON dict: ключ = имя объекта, значение = 8-символьный hex ID.

```json
{"hero_character":"6b537c68","building":"e171d793","vehicle":"9d966ac0"}
```

Hex ID — это little-endian reinterpretation float32 → uint32 → hex.

## Типичные ошибки ("soft mask" в Nuke)

| Проблема | Результат | Решение |
|----------|-----------|---------|
| Каналы в HALF (float16) | Потеря precision, soft/wrong mask | Все Crypto каналы = FLOAT32 |
| Отсутствие metadata | Nuke не видит Cryptomatte | Все 4 ключа обязательны |
| Неправильный layer_hash | Nuke не связывает metadata с каналами | `id_to_hex(mm3hash_float(name))[:-1]` |
| Reformat/LensDistortion до Cryptomatte | Повреждение ID значений | Убрать фильтрацию до Cryptomatte |
| ID в integer каналах | Nuke не считывает | ID хранить как FLOAT (reinterpreted) |
| Manifest hash не совпадает | Picker не выделяет объект | Перегенерировать manifest |
| Нет exponent fix (denormals) | Некорректные float значения | Обязательно флипать bit 23 |
| Coverage не отсортирована по убыванию | Неправильная иерархия | Rank 0 = максимальное coverage |

## Рабочий генератор

Скрипт: `scripts/generate_cryptomatte_exr.py`

```bash
# Demo с тестовыми объектами
.venv/bin/python3 scripts/generate_cryptomatte_exr.py

# Custom
.venv/bin/python3 scripts/generate_cryptomatte_exr.py \
    --output /path/to/output.exr \
    --width 1920 --height 1080 \
    --layer CryptoObject --levels 3
```

## Ссылки

- [Psyop/Cryptomatte](https://github.com/Psyop/Cryptomatte) — спецификация + reference Nuke implementation
- [Foundry Learn: Cryptomatte](https://learn.foundry.com/nuke/content/comp_environment/cryptomatte/keying_with_cryptomatte.html) — Nuke docs
- [Cryptomatte Spec v1.2.0 PDF](https://raw.githubusercontent.com/Psyop/Cryptomatte/master/specification/cryptomatte_specification.pdf)
- [cryptomatte_utilities.py](https://github.com/Psyop/Cryptomatte/blob/master/nuke/cryptomatte_utilities.py) — reference Python code
