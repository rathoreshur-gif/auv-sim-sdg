"""
Minimal OpenEXR reader (numpy + zlib) for the files Movie Render Queue writes:
single-part scanline images with NONE / RLE / ZIPS / ZIP compression and HALF / FLOAT / UINT
channels, plus the header attributes (MRQ stores the camera pose and FOV there).

    img = exr.read(path)
    img.channels["R"]                  # (H, W) array, float16 / float32 / uint32
    img.layer("FinalImageSDGLabels")   # (H, W, C) stack of that layer's R, G, B[, A]
    img.attrs["unreal/camera/curPos/x"]
"""
from __future__ import annotations

import struct
import zlib

import numpy as np

MAGIC = 20000630
COMPRESSION_NAMES = {0: "none", 1: "rle", 2: "zips", 3: "zip", 4: "piz", 5: "pxr24", 6: "b44", 7: "b44a",
                     8: "dwaa", 9: "dwab"}
LINES_PER_BLOCK = {0: 1, 1: 1, 2: 1, 3: 16}
DTYPES = {0: np.dtype("<u4"), 1: np.dtype("<f2"), 2: np.dtype("<f4")}


class ExrError(Exception):
    pass


class ExrIncomplete(ExrError):
    """The file is still being written (or was cut short)."""


class ExrImage:
    def __init__(self, width, height, attrs, channels):
        self.width, self.height, self.attrs, self.channels = width, height, attrs, channels

    def layer_names(self):
        """Layer prefixes present ("" = the unnamed first layer)."""
        return sorted({n.rpartition(".")[0] for n in self.channels})

    def layer(self, name, components="RGBA"):
        pre = f"{name}." if name else ""
        planes = [self.channels[pre + c] for c in components if pre + c in self.channels]
        if not planes:
            raise KeyError(f"no layer {name!r}; layers: {self.layer_names()}")
        return np.stack(planes, axis=-1)


# ---------------------------------------------------------------------------- header
def _cstr(buf, pos):
    end = buf.index(b"\0", pos)
    return buf[pos:end].decode("latin-1"), end + 1


def _parse_attr(typ, data):
    if typ == "string":
        return data.decode("utf-8", "replace")
    if typ == "int":
        return struct.unpack("<i", data)[0]
    if typ == "float":
        return struct.unpack("<f", data)[0]
    if typ == "double":
        return struct.unpack("<d", data)[0]
    if typ in ("compression", "lineOrder", "envmap", "deepImageState"):
        return data[0]
    if typ == "box2i":
        return struct.unpack("<4i", data)
    if typ == "box2f":
        return struct.unpack("<4f", data)
    if typ == "v2i":
        return struct.unpack("<2i", data)
    if typ == "v2f":
        return struct.unpack("<2f", data)
    if typ == "v3f":
        return struct.unpack("<3f", data)
    if typ == "m44f":
        return struct.unpack("<16f", data)
    if typ == "chromaticities":
        return struct.unpack("<8f", data)
    if typ == "chlist":
        chans, pos = [], 0
        while data[pos] != 0:
            name, pos = _cstr(data, pos)
            ptype, _plinear, xs, ys = struct.unpack_from("<iB3xii", data, pos)
            pos += 16
            chans.append((name, ptype, xs, ys))
        return chans
    if typ == "stringvector":
        out, pos = [], 0
        while pos < len(data):
            n = struct.unpack_from("<i", data, pos)[0]
            out.append(data[pos + 4:pos + 4 + n].decode("utf-8", "replace"))
            pos += 4 + n
        return out
    return data


def _read_header(buf):
    if len(buf) < 8:
        raise ExrIncomplete("file too short")
    magic, version = struct.unpack_from("<ii", buf, 0)
    if magic != MAGIC:
        raise ExrError("not an EXR file")
    if version & 0x200:
        raise ExrError("tiled EXR is not supported")
    if version & 0x800:
        raise ExrError("deep EXR is not supported")
    if version & 0x1000:
        raise ExrError("multi-part EXR is not supported (turn off 'multipart' in the EXR output)")
    pos, attrs = 8, {}
    try:
        while True:
            name, pos = _cstr(buf, pos)
            if not name:
                break
            typ, pos = _cstr(buf, pos)
            size = struct.unpack_from("<i", buf, pos)[0]
            pos += 4
            if pos + size > len(buf):
                raise ExrIncomplete("header cut short")
            attrs[name] = _parse_attr(typ, bytes(buf[pos:pos + size]))
            pos += size
    except (ValueError, struct.error, IndexError):
        raise ExrIncomplete("header cut short") from None
    return attrs, pos


def read_header(path):
    with open(path, "rb") as f:
        buf = f.read(1 << 16)
    return _read_header(buf)[0]


# ---------------------------------------------------------------------------- pixel data
def _undo_predictor_and_interleave(raw):
    """ZIP/RLE pre-processing, reversed: delta-decode, then re-interleave the two byte halves."""
    d = np.frombuffer(raw, dtype=np.uint8).copy()
    if d.size > 1:
        d[1:] ^= 0x80                          # (x - 128) mod 256
    t = np.cumsum(d, dtype=np.uint8)            # running sum wraps mod 256
    out = np.empty_like(t)
    half = (t.size + 1) // 2
    out[0::2] = t[:half]
    out[1::2] = t[half:]
    return out


def _rle_decode(data, expected):
    src = np.frombuffer(data, dtype=np.int8)
    out = bytearray()
    i, n = 0, src.size
    while i < n:
        c = int(src[i])
        if c < 0:
            out += data[i + 1:i + 1 - c]
            i += 1 - c
        else:
            out += bytes([data[i + 1]]) * (c + 1)
            i += 2
    if len(out) != expected:
        raise ExrError("corrupt RLE block")
    return bytes(out)


def read(path, channels=None):
    """Read an EXR. `channels` (optional) limits which channels are converted to arrays."""
    with open(path, "rb") as f:
        buf = f.read()
    attrs, pos = _read_header(buf)
    try:
        xmin, ymin, xmax, ymax = attrs["dataWindow"]
        chlist = attrs["channels"]
        comp = attrs.get("compression", 0)
    except KeyError as e:
        raise ExrError(f"missing header attribute {e}") from None
    if comp not in LINES_PER_BLOCK:
        raise ExrError(f"{COMPRESSION_NAMES.get(comp, comp)} compression is not supported - "
                       f"set render.exr_compression = \"zip\" in the config")
    if any(xs != 1 or ys != 1 for _, _, xs, ys in chlist):
        raise ExrError("sub-sampled channels are not supported")

    width, height = xmax - xmin + 1, ymax - ymin + 1
    lines = LINES_PER_BLOCK[comp]
    nblocks = (height + lines - 1) // lines
    if pos + 8 * nblocks > len(buf):
        raise ExrIncomplete("offset table cut short")
    offsets = np.frombuffer(buf, dtype="<u8", count=nblocks, offset=pos)
    if offsets.size and (offsets.min() < pos + 8 * nblocks or offsets.max() >= len(buf)):
        raise ExrIncomplete("offset table not written yet")

    line_bytes = sum(width * DTYPES[t].itemsize for _, t, _, _ in chlist)
    image = np.empty((height, line_bytes), dtype=np.uint8)
    for off in offsets:
        y, size = struct.unpack_from("<ii", buf, int(off))
        start = int(off) + 8
        if start + size > len(buf):
            raise ExrIncomplete("pixel block cut short")
        row0 = y - ymin
        n = min(lines, height - row0)
        if row0 < 0 or n <= 0:
            raise ExrError(f"bad scanline block y={y}")
        expected = n * line_bytes
        data = buf[start:start + size]
        if comp == 0 or size == expected:       # stored raw (also when compression didn't help)
            raw = np.frombuffer(data, dtype=np.uint8)
        elif comp == 1:
            raw = _undo_predictor_and_interleave(_rle_decode(data, expected))
        else:
            raw = _undo_predictor_and_interleave(zlib.decompress(data))
        if raw.size != expected:
            raise ExrError(f"block y={y}: {raw.size} bytes, expected {expected}")
        image[row0:row0 + n] = raw.reshape(n, line_bytes)

    out, col = {}, 0
    for name, ptype, _, _ in chlist:        # chlist is sorted; within a scanline channels follow that order
        dt = DTYPES[ptype]
        nb = width * dt.itemsize
        if channels is None or name in channels:
            out[name] = np.ascontiguousarray(image[:, col:col + nb]).view(dt).reshape(height, width)
        col += nb
    return ExrImage(width, height, attrs, out)
