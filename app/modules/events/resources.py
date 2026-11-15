"""Fit the processing to the machine: full resolution where memory allows, NxN multilook only
where it does not.

Processing peaks at about 255 bytes per output pixel with HH+HV (measured: 483 MB for a 1.5 M px
scene, ~100 MB of it imports), so a 6 M px scene needs ~1.6 GB. On a capable computer the
pipeline keeps the native 10/20 m grid; on a small one (a WSL VM with ~1.8 GB, say) it averages
2x2 or 3x3 blocks of linear power first, the standard SAR multilook, and records it in the event.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np

# Peak memory of the pipeline (pipeline.run -> write_event) per output pixel, with ~25% headroom
# over the measured 255 B/px. HV-less scenes use less; one conservative figure keeps it simple.
BYTES_PER_PIXEL = 320
# The two full-resolution input dates (HH + HV, float32) held before any multilook.
INPUT_BYTES_PER_PIXEL = 16
# Interpreter, numpy/scipy/h5py and the HTTP stack.
BASE_BYTES = 150 * 2**20
# HDF5 raw-chunk cache while streaming a granule: large keeps the request count down on a capable
# machine; on a small one it alone can exhaust memory mid-read (seen: OOM reading a 7 M px window).
READ_CACHE_MAX = 256 * 2**20
READ_CACHE_MIN = 32 * 2**20
# Beyond 4x4 (40-80 m pixels) the change maps lose the detail they exist for: ask for a smaller box.
MAX_LOOKS = 4


def _meminfo_available() -> int | None:
    """MemAvailable from /proc/meminfo (Linux, WSL); None elsewhere."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def _cgroup_headroom() -> int | None:
    """Room left under this process's cgroup v2 memory cap (systemd-run -p MemoryMax=...,
    containers); None when there is no cap or no cgroup v2."""
    try:
        line = Path("/proc/self/cgroup").read_text().splitlines()[0]
        folder = Path("/sys/fs/cgroup") / line.split("::", 1)[1].lstrip("/")
        cap = (folder / "memory.max").read_text().strip()
        if cap == "max":
            return None
        return int(cap) - int((folder / "memory.current").read_text())
    except (OSError, IndexError, ValueError):
        return None


def available_memory() -> int | None:
    """Bytes this process can still use without swapping: the tighter of the machine's free
    memory and any cgroup cap. None when neither can be read (then assume a capable machine)."""
    known = [v for v in (_meminfo_available(), _cgroup_headroom()) if v is not None]
    return min(known) if known else None


def read_cache_bytes(available: int | None) -> int:
    """HDF5 chunk cache for streaming: full size unless memory is short, then a quarter of it."""
    if available is None:
        return READ_CACHE_MAX
    return max(READ_CACHE_MIN, min(READ_CACHE_MAX, available // 4))


def choose_looks(pixels: int, available: int | None, forced: int | None = None) -> int:
    """Smallest NxN multilook whose estimated peak fits in `available` bytes. 1 = native grid."""
    if forced:
        return forced
    if available is None:
        return 1
    held = BASE_BYTES + pixels * INPUT_BYTES_PER_PIXEL
    for looks in range(1, MAX_LOOKS + 1):
        if held + pixels / looks**2 * BYTES_PER_PIXEL <= available:
            return looks
    need = (held + pixels / MAX_LOOKS**2 * BYTES_PER_PIXEL) / 2**20
    raise SystemExit(
        f"Not enough memory: {pixels / 1e6:.1f} M px needs ~{need:.0f} MB even at "
        f"{MAX_LOOKS}x{MAX_LOOKS} looks, {available / 2**20:.0f} MB free. Use a smaller --bbox."
    )


def multilook(a: np.ndarray, looks: int) -> np.ndarray:
    """NxN block mean of linear power, NaN-aware (a block with no samples stays NaN). The ragged
    edge that does not fill a whole block is dropped. looks=1 returns the array itself."""
    if looks == 1:
        return a
    rows, cols = (a.shape[0] // looks) * looks, (a.shape[1] // looks) * looks
    blocks = a[:rows, :cols].reshape(rows // looks, looks, cols // looks, looks)
    counts = np.isfinite(blocks).sum(axis=(1, 3))
    sums = np.nansum(blocks, axis=(1, 3), dtype="float64")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, sums / counts, np.nan).astype("float32")


def multilook_grid(grid, looks: int):
    """The grid of a multilooked raster: pixel centres move to the centre of each block."""
    if looks == 1:
        return grid
    shift = (looks - 1) / 2
    return replace(
        grid,
        x0=grid.x0 + grid.dx * shift,
        y0=grid.y0 + grid.dy * shift,
        dx=grid.dx * looks,
        dy=grid.dy * looks,
    )
