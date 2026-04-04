#!/usr/bin/env python3
"""
Cryptomatte EXR Generator — Nuke 16+ compatible.

Generates a valid Cryptomatte EXR file that works with Nuke's built-in Cryptomatte node.
Follows the Cryptomatte Specification v1.2.0 (Psyop/Cryptomatte).

Usage:
    python generate_cryptomatte_exr.py                    # demo with test objects
    python generate_cryptomatte_exr.py --output /path/out.exr --width 1920 --height 1080

Requirements (all in project .venv):
    - OpenEXR >= 3.x
    - Imath
    - numpy

Author: Igor Baydak / Claude Code
Date: 2026-04-03
"""

import struct
import json
import argparse
import numpy as np
from pathlib import Path

try:
    import OpenEXR
    import Imath
except ImportError:
    raise ImportError("OpenEXR and Imath are required. Install via: pip install OpenEXR")


# ============================================================================
# Cryptomatte Core Functions (from Psyop/Cryptomatte specification)
# ============================================================================

def murmur3_32(key, seed=0):
    """Pure Python MurmurHash3_x86_32 implementation.

    Matches the C reference implementation from Austin Appleby.
    Compatible with mmh3.hash(key, signed=False).
    """
    if isinstance(key, str):
        key = key.encode('utf-8')
    length = len(key)
    h = seed & 0xFFFFFFFF
    c1 = 0xcc9e2d51
    c2 = 0x1b873593

    # body
    nblocks = length // 4
    for i in range(nblocks):
        k = struct.unpack('<I', key[i * 4:(i + 1) * 4])[0]
        k = (k * c1) & 0xFFFFFFFF
        k = ((k << 15) | (k >> 17)) & 0xFFFFFFFF
        k = (k * c2) & 0xFFFFFFFF
        h ^= k
        h = ((h << 13) | (h >> 19)) & 0xFFFFFFFF
        h = (h * 5 + 0xe6546b64) & 0xFFFFFFFF

    # tail
    tail = key[nblocks * 4:]
    k = 0
    tlen = len(tail)
    if tlen >= 3:
        k ^= tail[2] << 16
    if tlen >= 2:
        k ^= tail[1] << 8
    if tlen >= 1:
        k ^= tail[0]
        k = (k * c1) & 0xFFFFFFFF
        k = ((k << 15) | (k >> 17)) & 0xFFFFFFFF
        k = (k * c2) & 0xFFFFFFFF
        h ^= k

    # finalize
    h ^= length
    h ^= (h >> 16)
    h = (h * 0x85ebca6b) & 0xFFFFFFFF
    h ^= (h >> 13)
    h = (h * 0xc2b2ae35) & 0xFFFFFFFF
    h ^= (h >> 16)
    return h


def mm3hash_float(name):
    """Convert object name to Cryptomatte float ID.

    Steps:
    1. MurmurHash3_32 of the name string
    2. Avoid denormals (exp==0) and NaN/Inf (exp==255) by flipping bit 23
    3. Reinterpret uint32 as IEEE 754 float32

    This is the EXACT algorithm from Psyop/Cryptomatte specification.
    """
    hash_32 = murmur3_32(name)
    # Avoid special float values (denormals, NaN, Inf)
    exp = (hash_32 >> 23) & 255
    if exp == 0 or exp == 255:
        hash_32 ^= (1 << 23)
    # Reinterpret as float32
    packed = struct.pack('<I', hash_32 & 0xFFFFFFFF)
    return struct.unpack('<f', packed)[0]


def id_to_hex(float_id):
    """Convert Cryptomatte float ID to 8-character hex string.

    Reinterprets IEEE 754 float32 bits as uint32, formats as lowercase hex.
    """
    return "{0:08x}".format(struct.unpack('<I', struct.pack('<f', float_id))[0])


def layer_hash(layer_name):
    """Compute 7-character hex identifier for Cryptomatte metadata keys.

    Used in metadata keys like: cryptomatte/{layer_hash}/name
    The layer_hash is the first 7 chars of id_to_hex(mm3hash_float(layer_name)).
    """
    return id_to_hex(mm3hash_float(layer_name))[:-1]


# ============================================================================
# Cryptomatte EXR Writer
# ============================================================================

class CryptomatteLayer:
    """Represents one Cryptomatte layer (e.g. CryptoObject, CryptoMaterial)."""

    def __init__(self, layer_name, num_levels=3):
        """
        Args:
            layer_name: Base name, e.g. "CryptoObject", "CryptoMaterial", "CryptoAsset"
            num_levels: Number of ID-coverage pair levels (3 = 6 ranks, typical default).
                       Each level stores 2 ID-coverage pairs in RGBA.
        """
        self.layer_name = layer_name
        self.num_levels = num_levels
        self.objects = {}  # name -> float_id

    def add_object(self, name):
        """Register an object name and compute its Cryptomatte float ID."""
        float_id = mm3hash_float(name)
        self.objects[name] = float_id
        return float_id

    def get_manifest(self):
        """Build JSON manifest: {"object_name": "hex_id", ...}"""
        manifest = {}
        for name, float_id in self.objects.items():
            manifest[name] = id_to_hex(float_id)
        return json.dumps(manifest, separators=(',', ':'))

    def get_layer_hash(self):
        """7-char hex identifier for metadata keys."""
        return layer_hash(self.layer_name)

    def get_metadata(self):
        """Build all metadata key-value pairs for this layer.

        Returns dict with keys like:
            cryptomatte/{hash}/name -> layer_name
            cryptomatte/{hash}/hash -> MurmurHash3_32
            cryptomatte/{hash}/conversion -> uint32_to_float32
            cryptomatte/{hash}/manifest -> JSON manifest

        All values are bytes (required by OpenEXR Python binding).
        """
        lh = self.get_layer_hash()
        prefix = f"cryptomatte/{lh}/"
        return {
            prefix + "name": self.layer_name.encode('utf-8'),
            prefix + "hash": b"MurmurHash3_32",
            prefix + "conversion": b"uint32_to_float32",
            prefix + "manifest": self.get_manifest().encode('utf-8'),
        }

    def get_channel_names(self):
        """Return list of EXR channel names for this layer.

        Format: {LayerName}{NN}.{R|G|B|A}
        where NN is zero-padded level index (00, 01, 02, ...)

        Each level has 4 channels:
            .R = ID of rank 0 (within this level)
            .G = coverage of rank 0
            .B = ID of rank 1
            .A = coverage of rank 1
        """
        channels = []
        for level in range(self.num_levels):
            prefix = f"{self.layer_name}{level:02d}"
            channels.extend([
                f"{prefix}.R",
                f"{prefix}.G",
                f"{prefix}.B",
                f"{prefix}.A",
            ])
        return channels


def generate_cryptomatte_exr(
    output_path,
    width,
    height,
    object_masks,
    layer_name="CryptoObject",
    num_levels=3,
    compression="ZIP",
    include_rgba=True,
):
    """Generate a Cryptomatte EXR file.

    Args:
        output_path: Path to output EXR file
        width: Image width in pixels
        height: Image height in pixels
        object_masks: Dict of {object_name: numpy_mask_array}
                     Each mask is float32 array (H, W) with values 0.0-1.0
        layer_name: Cryptomatte layer name (default "CryptoObject")
        num_levels: Number of coverage levels (default 3)
        compression: EXR compression ("ZIP", "ZIPS", "PIZ", "NONE")
        include_rgba: Whether to include RGBA beauty channels (constant color)

    Returns:
        Path to generated file
    """
    output_path = str(output_path)

    # --- Create Cryptomatte layer ---
    crypto_layer = CryptomatteLayer(layer_name, num_levels)

    # Register all objects and get their float IDs
    object_ids = {}
    for name in object_masks:
        object_ids[name] = crypto_layer.add_object(name)

    # --- Build per-pixel ranked ID-coverage lists ---
    # For each pixel, we need a ranked list of (id, coverage) pairs
    # sorted by coverage descending.

    num_ranks = num_levels * 2  # 2 pairs per level

    # Initialize channel data arrays
    # Each level: R=id0, G=cov0, B=id1, A=cov1
    channel_data = {}
    for ch_name in crypto_layer.get_channel_names():
        channel_data[ch_name] = np.zeros(width * height, dtype=np.float32)

    # For each pixel, collect all (id_float, coverage) and sort by coverage desc
    # Then distribute into ranked channels

    # Build coverage matrix: (num_objects, H*W)
    obj_names = list(object_masks.keys())
    num_objects = len(obj_names)

    if num_objects > 0:
        coverage_matrix = np.zeros((num_objects, height * width), dtype=np.float32)
        id_values = np.zeros(num_objects, dtype=np.float32)

        for i, name in enumerate(obj_names):
            mask = object_masks[name]
            if mask.shape != (height, width):
                raise ValueError(f"Mask for '{name}' has shape {mask.shape}, expected ({height}, {width})")
            coverage_matrix[i] = mask.astype(np.float32).ravel()
            id_values[i] = object_ids[name]

        # For each pixel, sort objects by coverage (descending)
        # Use argsort on negative coverage for descending order
        sorted_indices = np.argsort(-coverage_matrix, axis=0)  # (num_objects, H*W)

        # Fill ranked channels
        for rank in range(min(num_ranks, num_objects)):
            level = rank // 2
            sub_rank = rank % 2  # 0 or 1 within the level

            prefix = f"{layer_name}{level:02d}"
            if sub_rank == 0:
                id_channel = f"{prefix}.R"
                cov_channel = f"{prefix}.G"
            else:
                id_channel = f"{prefix}.B"
                cov_channel = f"{prefix}.A"

            # Get the object index for this rank at each pixel
            obj_idx = sorted_indices[rank]  # (H*W,) indices into obj_names

            # Get coverage for this rank
            cov = coverage_matrix[obj_idx, np.arange(height * width)]

            # Get float IDs for this rank
            ids = id_values[obj_idx]

            # Only write where coverage > 0
            mask_nonzero = cov > 0
            channel_data[id_channel][mask_nonzero] = ids[mask_nonzero]
            channel_data[cov_channel][mask_nonzero] = cov[mask_nonzero]

    # --- Build EXR header ---
    header = OpenEXR.Header(width, height)

    # Compression
    comp_map = {
        "NONE": OpenEXR.NO_COMPRESSION,
        "ZIP": OpenEXR.ZIP_COMPRESSION,
        "ZIPS": OpenEXR.ZIPS_COMPRESSION,
        "PIZ": OpenEXR.PIZ_COMPRESSION,
        "RLE": OpenEXR.RLE_COMPRESSION,
    }
    header['compression'] = comp_map.get(compression.upper(), OpenEXR.ZIP_COMPRESSION)

    # Channel definitions (ALL must be FLOAT, not HALF!)
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    channels = {}

    if include_rgba:
        for ch in ['R', 'G', 'B', 'A']:
            channels[ch] = Imath.Channel(FLOAT)

    for ch_name in crypto_layer.get_channel_names():
        channels[ch_name] = Imath.Channel(FLOAT)

    header['channels'] = channels

    # Cryptomatte metadata (MUST be bytes for OpenEXR Python binding)
    for key, value in crypto_layer.get_metadata().items():
        header[key] = value

    # --- Write pixel data ---
    pixel_data = {}

    if include_rgba:
        # Simple constant RGBA (black with full alpha)
        zeros = np.zeros(width * height, dtype=np.float32).tobytes()
        ones = np.ones(width * height, dtype=np.float32).tobytes()
        pixel_data['R'] = zeros
        pixel_data['G'] = zeros
        pixel_data['B'] = zeros
        pixel_data['A'] = ones

    for ch_name in crypto_layer.get_channel_names():
        pixel_data[ch_name] = channel_data[ch_name].tobytes()

    # Write file
    out = OpenEXR.OutputFile(output_path, header)
    out.writePixels(pixel_data)
    out.close()

    return output_path


# ============================================================================
# Verification: read back and validate
# ============================================================================

def verify_cryptomatte_exr(filepath):
    """Read back a Cryptomatte EXR and verify its structure.

    Returns dict with verification results.
    """
    filepath = str(filepath)
    inp = OpenEXR.InputFile(filepath)
    header = inp.header()

    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'info': {},
    }

    # Check channels
    channels = list(header['channels'].keys())
    results['info']['channels'] = channels

    crypto_channels = [ch for ch in channels if 'Crypto' in ch or 'crypto' in ch]
    results['info']['crypto_channels'] = crypto_channels

    if not crypto_channels:
        results['valid'] = False
        results['errors'].append("No Cryptomatte channels found")

    # Check channel data type (must be FLOAT, not HALF)
    for ch_name in crypto_channels:
        ch_info = header['channels'][ch_name]
        if str(ch_info).startswith('HALF'):
            results['valid'] = False
            results['errors'].append(f"Channel {ch_name} is HALF, must be FLOAT")

    # Check metadata
    crypto_meta = {}
    for key in header:
        if key.startswith('cryptomatte/'):
            crypto_meta[key] = header[key]

    results['info']['metadata'] = {k: v.decode('utf-8') if isinstance(v, bytes) else v
                                    for k, v in crypto_meta.items()}

    if not crypto_meta:
        results['valid'] = False
        results['errors'].append("No cryptomatte/ metadata found in header")

    # Extract layer info
    layers = {}
    for key, value in crypto_meta.items():
        parts = key.split('/')
        if len(parts) == 3:
            _, layer_id, attr = parts
            if layer_id not in layers:
                layers[layer_id] = {}
            layers[layer_id][attr] = value.decode('utf-8') if isinstance(value, bytes) else value

    results['info']['layers'] = layers

    # Validate each layer
    required_attrs = ['name', 'hash', 'conversion', 'manifest']
    for layer_id, attrs in layers.items():
        for req in required_attrs:
            if req not in attrs:
                results['valid'] = False
                results['errors'].append(f"Layer {layer_id} missing '{req}' metadata")

        # Verify hash type
        if attrs.get('hash') != 'MurmurHash3_32':
            results['warnings'].append(f"Layer {layer_id}: unexpected hash type '{attrs.get('hash')}'")

        # Verify conversion
        if attrs.get('conversion') != 'uint32_to_float32':
            results['warnings'].append(f"Layer {layer_id}: unexpected conversion '{attrs.get('conversion')}'")

        # Verify manifest is valid JSON
        manifest_str = attrs.get('manifest', '')
        try:
            manifest = json.loads(manifest_str)
            results['info']['manifest_objects'] = list(manifest.keys())

            # Verify each manifest entry hash matches
            layer_name = attrs.get('name', '')
            for obj_name, hex_id in manifest.items():
                expected_hex = id_to_hex(mm3hash_float(obj_name))
                if hex_id != expected_hex:
                    results['errors'].append(
                        f"Manifest hash mismatch for '{obj_name}': "
                        f"manifest={hex_id}, computed={expected_hex}"
                    )
                    results['valid'] = False
        except json.JSONDecodeError:
            results['valid'] = False
            results['errors'].append(f"Layer {layer_id}: manifest is not valid JSON")

        # Verify layer_hash matches
        name = attrs.get('name', '')
        expected_lh = layer_hash(name)
        if layer_id != expected_lh:
            results['warnings'].append(
                f"Layer ID '{layer_id}' doesn't match computed layer_hash "
                f"'{expected_lh}' for name '{name}'"
            )

        # Check that channels exist for this layer
        if name:
            found_levels = set()
            for ch in crypto_channels:
                if ch.startswith(name):
                    # Extract level number
                    rest = ch[len(name):]
                    level_str = rest.split('.')[0]
                    if level_str.isdigit():
                        found_levels.add(int(level_str))

            if not found_levels:
                results['errors'].append(f"No channels found for layer '{name}'")
                results['valid'] = False
            else:
                results['info']['levels'] = sorted(found_levels)

    # Read actual pixel data for first crypto channel to verify non-zero
    if crypto_channels:
        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
        first_ch = crypto_channels[0]
        data = inp.channel(first_ch, FLOAT)
        arr = np.frombuffer(data, dtype=np.float32)
        nonzero = np.count_nonzero(arr)
        results['info']['first_channel_nonzero_pixels'] = int(nonzero)
        results['info']['first_channel_total_pixels'] = len(arr)

    inp.close()
    return results


# ============================================================================
# Demo: generate test Cryptomatte with simple geometric masks
# ============================================================================

def create_demo_masks(width, height):
    """Create simple geometric masks for testing.

    Returns dict of {object_name: mask_array}.
    """
    masks = {}

    # Object 1: circle in upper-left
    y, x = np.mgrid[0:height, 0:width]
    cx1, cy1, r1 = width * 0.25, height * 0.25, min(width, height) * 0.15
    dist1 = np.sqrt((x - cx1) ** 2 + (y - cy1) ** 2)
    masks['hero_character'] = np.clip(1.0 - (dist1 - r1 + 2) / 4.0, 0, 1).astype(np.float32)

    # Object 2: rectangle in center
    mask2 = np.zeros((height, width), dtype=np.float32)
    x1, y1 = int(width * 0.35), int(height * 0.35)
    x2, y2 = int(width * 0.65), int(height * 0.65)
    mask2[y1:y2, x1:x2] = 1.0
    masks['building'] = mask2

    # Object 3: circle in lower-right
    cx3, cy3, r3 = width * 0.75, height * 0.75, min(width, height) * 0.2
    dist3 = np.sqrt((x - cx3) ** 2 + (y - cy3) ** 2)
    masks['vehicle'] = np.clip(1.0 - (dist3 - r3 + 2) / 4.0, 0, 1).astype(np.float32)

    # Object 4: gradient bar at bottom (tests soft edges)
    mask4 = np.zeros((height, width), dtype=np.float32)
    bar_top = int(height * 0.85)
    bar_bottom = height
    for row in range(bar_top, bar_bottom):
        t = (row - bar_top) / (bar_bottom - bar_top)
        mask4[row, :] = t
    masks['ground'] = mask4

    # Object 5: background (everything not covered by other objects)
    combined = np.zeros((height, width), dtype=np.float32)
    for m in masks.values():
        combined = np.maximum(combined, m)
    masks['background'] = np.clip(1.0 - combined, 0, 1).astype(np.float32)

    return masks


def main():
    parser = argparse.ArgumentParser(
        description="Generate Cryptomatte EXR files compatible with Nuke 16+",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                          # Demo: 1920x1080 test
  %(prog)s --output /tmp/crypto.exr --width 2048 --height 1152
  %(prog)s --layer CryptoMaterial --levels 6
        """,
    )
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output EXR path (default: scripts/test_cryptomatte.exr)')
    parser.add_argument('--width', '-W', type=int, default=1920)
    parser.add_argument('--height', '-H', type=int, default=1080)
    parser.add_argument('--layer', type=str, default='CryptoObject',
                       help='Cryptomatte layer name (default: CryptoObject)')
    parser.add_argument('--levels', type=int, default=3,
                       help='Number of levels/ranks (default: 3 = 6 ranks)')
    parser.add_argument('--compression', type=str, default='ZIP',
                       choices=['ZIP', 'ZIPS', 'PIZ', 'NONE', 'RLE'])
    parser.add_argument('--verify', action='store_true', default=True,
                       help='Verify output after generation (default: True)')
    parser.add_argument('--no-verify', dest='verify', action='store_false')

    args = parser.parse_args()

    if args.output is None:
        script_dir = Path(__file__).parent
        args.output = str(script_dir / 'test_cryptomatte.exr')

    print(f"=== Cryptomatte EXR Generator ===")
    print(f"Output:      {args.output}")
    print(f"Resolution:  {args.width}x{args.height}")
    print(f"Layer:       {args.layer}")
    print(f"Levels:      {args.levels} ({args.levels * 2} ranks)")
    print(f"Compression: {args.compression}")
    print()

    # Create demo masks
    print("Creating demo object masks...")
    masks = create_demo_masks(args.width, args.height)
    for name, mask in masks.items():
        coverage_pct = (mask > 0).sum() / (args.width * args.height) * 100
        float_id = mm3hash_float(name)
        hex_id = id_to_hex(float_id)
        print(f"  {name:20s}  ID={hex_id}  coverage={coverage_pct:.1f}%")

    print()
    print(f"Layer hash for '{args.layer}': {layer_hash(args.layer)}")
    print()

    # Generate EXR
    print("Writing Cryptomatte EXR...")
    result_path = generate_cryptomatte_exr(
        output_path=args.output,
        width=args.width,
        height=args.height,
        object_masks=masks,
        layer_name=args.layer,
        num_levels=args.levels,
        compression=args.compression,
    )

    file_size = Path(result_path).stat().st_size
    print(f"Written: {result_path} ({file_size / 1024 / 1024:.1f} MB)")
    print()

    # Verify
    if args.verify:
        print("=== Verification ===")
        results = verify_cryptomatte_exr(result_path)

        if results['valid']:
            print("STATUS: VALID Cryptomatte EXR")
        else:
            print("STATUS: INVALID — issues found!")

        print()
        print(f"Channels ({len(results['info']['channels'])}):")
        for ch in sorted(results['info']['channels']):
            print(f"  {ch}")

        print()
        print("Metadata:")
        for key, value in sorted(results['info']['metadata'].items()):
            display_val = value if len(value) < 80 else value[:77] + "..."
            print(f"  {key} = {display_val}")

        print()
        if 'manifest_objects' in results['info']:
            print(f"Manifest objects: {results['info']['manifest_objects']}")

        if 'levels' in results['info']:
            print(f"Channel levels: {results['info']['levels']}")

        if 'first_channel_nonzero_pixels' in results['info']:
            nz = results['info']['first_channel_nonzero_pixels']
            total = results['info']['first_channel_total_pixels']
            print(f"First crypto channel: {nz}/{total} non-zero pixels ({nz/total*100:.1f}%)")

        if results['errors']:
            print()
            print("ERRORS:")
            for err in results['errors']:
                print(f"  [!] {err}")

        if results['warnings']:
            print()
            print("WARNINGS:")
            for warn in results['warnings']:
                print(f"  [?] {warn}")

    print()
    print("Done! Open in Nuke: Read node -> Cryptomatte node -> pick objects")


if __name__ == '__main__':
    main()
