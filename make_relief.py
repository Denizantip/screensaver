#!/usr/bin/env python3
"""ETOPO 2022 60-arcsec GeoTIFF -> relief.bin (720x360 int16, little-endian)."""
import sys
import numpy as np
import rasterio   # pip install rasterio

SRC, DST = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "relief.bin"
OUT_W, OUT_H = 720, 360

with rasterio.open(SRC) as ds:
    h, w = ds.height, ds.width
    assert w % OUT_W == 0 and h % OUT_H == 0, f"unexpected size {w}x{h}"
    bw, bh = w // OUT_W, h // OUT_H
    out = np.empty((OUT_H, OUT_W), dtype=np.float32)
    for r in range(OUT_H):                       # one output row at a time
        strip = ds.read(1, window=((r * bh, (r + 1) * bh), (0, w)))
        out[r] = strip.reshape(bh, OUT_W, bw).mean(axis=(0, 2))

out.round().astype("<i2").tofile(DST)
print(f"wrote {DST}: {out.size * 2 / 1024:.0f} KB, "
      f"min {out.min():.0f} m, max {out.max():.0f} m")
