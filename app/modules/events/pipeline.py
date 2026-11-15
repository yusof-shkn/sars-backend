"""Offline change detection on a pre/post pair of NISAR GCOV granules.

    uv run python -m app.modules.events.pipeline run --id <slug> \
        --pre <url|path> --post <url|path> --bbox W S E N \
        --title "..." --hazard flood --location "..." --description "..."
    uv run python -m app.modules.events.pipeline search --bbox W S E N [--start ... --end ...]
    uv run python -m app.modules.events.pipeline demo     # synthetic event, clearly labelled

Granule URLs are read in place over HTTPS range requests (only the bbox's chunks travel, not the
1–2 GB file); add --download to pull whole files into data/granules/ instead. Auth: see README.

Method, per pixel on the product's native UTM grid:
  1. 5x5 box mean of gamma0 (linear power) on each date — speckle reduction.
  2. d = 10*log10(post) - 10*log10(pre)  (the log-ratio, in dB).
  3. Pixels darker than the noise floor on BOTH dates (permanent water, radar shadow) carry no
     change information and are masked out.
  4. decrease = d <= -T, increase = d >= +T, each sieved with a 5x5 majority filter so lone
     speckle-sized blobs drop out.
  5. Stats on the native grid; rendering resampled onto a lon/lat grid for the web map.
A drop in L-band backscatter is what open water / burn scars / clearing look like; a rise is
what flooded vegetation (double bounce) or new structures look like. It is a change map, not a
damage or flood *prediction*.
"""

import argparse
import json
import math
import os
import re
import time
import urllib.request
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from pyproj import Transformer
from scipy import ndimage

from app.core.config import settings
from app.modules.events.ground import add_ground
from app.modules.events.resources import (
    MAX_LOOKS,
    available_memory,
    choose_looks,
    multilook,
    multilook_grid,
    read_cache_bytes,
)
from app.modules.events.schemas import Granule

# NISAR GCOV HDF5 layout per the product spec — verify on your first real granule
# (`h5ls -r file.h5 | grep GCOV`) and adjust here if a release moved it.
GCOV_GRID = "science/LSAR/GCOV/grids/frequencyA"
IDENT = "science/LSAR/identification"
POLARIZATIONS = ("HHHH", "VVVV")  # co-pol terms; first one present is used
CROSS_POL = {"HHHH": "HVHV", "VVVV": "VHVH"}  # matching cross-pol term, if the mode has one

# CMR collections, best processing first. Granule names look like
# NISAR_L2_PR_GCOV_<cycle>_<track>_<A|D>_<frame>_<mode>_<pols>_..._<start>_<end>_...
GCOV_COLLECTIONS = ("NISAR_L2_GCOV_V1", "NISAR_L2_GCOV_PROVISIONAL_V1", "NISAR_L2_GCOV_BETA_V1")

# NISAR's L-band noise-equivalent sigma0 is about -25 dB. A pixel this dark on both dates is
# noise on both dates, so its ratio is noise too.
NOISE_FLOOR_DB = -22.0

# Must match --change-decrease / --change-increase in sars-frontend/src/index.css.
DECREASE_RGB = (0x1D, 0x63, 0xC4)  # water / darker: blue
INCREASE_RGB = (0xB4, 0x2C, 0x7E)  # brighter, double bounce (generic brighter without HV): magenta
VOLUME_RGB = (0xE0, 0x7A, 0x1F)  # brighter in HH and HV, growth / rougher land: orange
ROUGH_RGB = (0x63, 0xB5, 0xC2)  # brighter but still water-dark: wind on water, pale teal
# NASA GIBS HLS (Harmonized Landsat + Sentinel-2, 30 m): a scene every 2-3 days, cloud permitting.
HLS_LAYERS = ("HLS_S30_Nadir_BRDF_Adjusted_Reflectance", "HLS_L30_Nadir_BRDF_Adjusted_Reflectance")
GIBS_TILE = "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/{layer}/default/{date}/GoogleMapsCompatible_Level12/12/{y}/{x}.png"
# Block size of the landscape-scale layer: coherent sub-threshold change over ~500 m.
SUBTLE_M = 500
# Change classes written to flags.npy / change.png
DEC, INC, VOLUME, ROUGH = 1, 2, 3, 4
# Longest side of the PNGs the browser loads; statistics always use the full grid.
MAX_DISPLAY_PX = 2048
# Open-water level when a scene has no water peak of its own (see water_line). Calm water at
# L-band HH sits well below this after the 5x5 filter; land well above it.
WATER_DB = -16.0
# A pixel that brightened but is still this dark is open water roughened by wind, not land.
ROUGH_MAX_DB = -9.0


@dataclass
class Grid:
    epsg: int
    x0: float  # top-left pixel centre
    y0: float
    dx: float
    dy: float  # negative: rows go north -> south


# ---------- maths (pure numpy, covered by tests/test_change.py) ----------


def _box_sum(m: np.ndarray, k: int) -> np.ndarray:
    """k x k moving sum via an integral image; windows are truncated at the edges."""
    p = k // 2
    c = np.pad(m, ((p + 1, p), (p + 1, p))).cumsum(0).cumsum(1)
    return c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]


def box_mean(a: np.ndarray, k: int = 5) -> np.ndarray:
    """NaN-aware k x k moving average (no k*k copies of the array)."""
    valid = np.isfinite(a) & (a > 0)
    n = _box_sum(valid.astype("float64"), k)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, _box_sum(np.where(valid, a, 0.0).astype("float64"), k) / n, np.nan)


def majority(mask: np.ndarray, k: int = 5) -> np.ndarray:
    """True where more than half of the k x k window is True — a sieve for speckle-sized blobs."""
    return _box_sum(mask.astype("float64"), k) * 2 > _box_sum(np.ones(mask.shape), k)


def to_db(a: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return 10 * np.log10(a)


def log_ratio(pre, post, floor_db: float = NOISE_FLOOR_DB):
    """Filtered dB images and their difference d; pixels dark on both dates are NaN in d."""
    pre_db, post_db = to_db(box_mean(pre)), to_db(box_mean(post))
    d = post_db - pre_db
    d[~np.isfinite(d)] = np.nan
    dark = np.isfinite(d) & (np.fmax(pre_db, post_db) < floor_db)
    d[dark] = np.nan
    return pre_db, post_db, d, dark


def noise_sigma(d) -> float:
    """Robust spread of the scene's ratio (MAD -> sigma). With real change on a minority of
    pixels this is the residual noise a threshold has to clear."""
    valid = np.isfinite(d)
    if not valid.any():
        return 0.0
    med = float(np.nanmedian(d))
    return 1.4826 * float(np.nanmedian(np.abs(d[valid] - med)))


def auto_threshold(d) -> float:
    """3 sigma of the scene, never below 3 dB (the usual L-band flood threshold)."""
    return max(3.0, round(3 * noise_sigma(d), 1))


def flag_change(d, threshold_db: float):
    valid = np.isfinite(d)
    return majority(valid & (d <= -threshold_db)), majority(valid & (d >= threshold_db))


def water_line(db: np.ndarray) -> float:
    """This scene's open-water level, from its own HH histogram (what NASA's ARSET flood method
    does by hand): halfway between the calm-water peak and the water/land split (Otsu). A fixed
    line misses water: on the Meghna the water peak sits at -16.6 dB, so a line at -16 dB catches
    only 72% of what NASA's optical water product (DSWx-HLS) calls open water; the halfway line
    (-13.2 dB) catches 93% at 83% precision. Scenes with no water peak keep WATER_DB."""
    from skimage.filters import threshold_otsu

    v = db[np.isfinite(db)]
    v = v[(v > -35) & (v < 5)]
    if v.size < 1000:
        return WATER_DB
    counts, edges = np.histogram(v, bins=80, range=(-35, 5))  # 0.5 dB bins
    counts = ndimage.uniform_filter1d(counts.astype("float64"), 3)
    centres = (edges[:-1] + edges[1:]) / 2
    peaks = [
        i
        for i in range(1, len(counts) - 1)
        if centres[i] < -14
        and counts[i] >= counts[i - 1]
        and counts[i] > counts[i + 1]
        and counts[i] >= 0.03 * counts.max()
    ]
    if not peaks:
        return WATER_DB
    split = float(threshold_otsu(v))
    return round(float(np.clip((centres[peaks[0]] + split) / 2, -20, -12)), 1)


def subtle_change(d, pre_db, post_db, grid: Grid, water_db: float = WATER_DB):
    """Coherent change below the pixel threshold: the block-mean log-ratio over land at ~500 m,
    relative to the scene's median change (so a season-wide shift does not light everything up),
    against a 3-sigma threshold. The 5x5 filter correlates neighbouring pixels, so a k x k block
    holds about (k/5)^2 independent samples and its noise is the pixel noise times 5/k. A boreal
    burn moves L-band HH by only ~+1 dB per pixel, invisible to the pixel test, but coherent
    here. Returns (block dB, threshold dB)."""
    k = max(3, int(round(SUBTLE_M / abs(grid.dx))) | 1)
    land = np.isfinite(d) & (np.fmax(pre_db, post_db) > water_db)
    blk = to_db(box_mean(np.where(land, 10 ** (d / 10), np.nan), k))
    blk[~land] = np.nan
    blk -= np.nanmedian(blk) if land.any() else 0.0
    # Floor 0.5 dB: the 3-sigma theory assumes independent noise, but real scenes carry smooth
    # calibration and moisture gradients. Measured on Thunder Bay: forest the same-day HLS photos
    # show unburned still varies by 0.75 dB across the scene (NISAR's PROVISIONAL known issues
    # list residual radiometric bands in 40 MHz products).
    return blk, max(0.5, round(3 * noise_sigma(d) * 5 / k, 2))


def save_subtle(blk: np.ndarray, thr: float, path: Path) -> None:
    """Diverging overlay: brightening in the increase hue, darkening in the decrease hue,
    opacity growing with magnitude beyond the threshold."""
    rgba = np.zeros((*blk.shape, 4), "uint8")
    with np.errstate(invalid="ignore"):
        strength = np.clip((np.abs(blk) - thr) / (2 * thr), 0.25, 1.0)
        up, down = blk > thr, blk < -thr
    for sel, rgb in ((up, VOLUME_RGB), (down, DECREASE_RGB)):
        rgba[sel, :3] = rgb
        rgba[sel, 3] = (strength[sel] * 200).astype("uint8")
    Image.fromarray(rgba, "RGBA").save(path)


def detect_change(pre, post, threshold_db: float, floor_db: float = NOISE_FLOOR_DB):
    """Returns (pre_db, post_db, d, decrease, increase, dark) — all filtered, same shape."""
    pre_db, post_db, d, dark = log_ratio(pre, post, floor_db)
    dec, inc = flag_change(d, threshold_db)
    return pre_db, post_db, d, dec, inc, dark


def summarize(d, dec, inc, dark, pixel_km2: float) -> dict:
    valid = np.isfinite(d)
    n = int(valid.sum()) or 1
    sigma = noise_sigma(d)
    counts, edges = np.histogram(np.clip(d[valid], -10, 10), bins=20, range=(-10, 10))
    return {
        "aoi_km2": round(n * pixel_km2, 2),
        "masked_km2": round(int(dark.sum()) * pixel_km2, 2),
        "decrease_km2": round(int(dec.sum()) * pixel_km2, 2),
        "increase_km2": round(int(inc.sum()) * pixel_km2, 2),
        "changed_pct": round(100 * int((dec | inc).sum()) / n, 2),
        "mean_change_db": round(float(np.nanmean(d)), 2) if valid.any() else 0.0,
        "noise_sigma_db": round(sigma, 2),
        "histogram": [{"db": float(e + 0.5), "pixels": int(c)} for e, c in zip(edges[:-1], counts)],
    }


# ---------- geometry ----------


def latlon_bounds(grid: Grid, shape) -> list[list[float]]:
    """Lon/lat box enclosing the UTM raster's four corners: [[south, west], [north, east]]."""
    rows, cols = shape
    xs = [grid.x0 - grid.dx / 2, grid.x0 + grid.dx * (cols - 0.5)]
    ys = [grid.y0 - grid.dy / 2, grid.y0 + grid.dy * (rows - 0.5)]
    inv = Transformer.from_crs(grid.epsg, 4326, always_xy=True)
    lon, lat = inv.transform([xs[0], xs[1], xs[0], xs[1]], [ys[0], ys[0], ys[1], ys[1]])
    return [[min(lat), min(lon)], [max(lat), max(lon)]]


def to_latlon(layers: list[np.ndarray], grid: Grid, shape, out_shape=None):
    """Nearest-neighbour resample from the product's UTM grid onto a regular lon/lat grid.

    Leaflet stretches an ImageOverlay between two lon/lat corners. A UTM raster is rotated
    relative to that by the grid convergence (about Δλ·sin φ — up to ~2° at a zone edge), which
    slides the corners of a 10–30 km scene 200 m to 1 km off the basemap. Resampling removes
    that exactly; statistics stay on the native grid. Returns (layers, bounds).
    """
    # One meshgrid for the whole scene: fine up to ~4k x 4k. Chunk by rows for a full frame.
    rows, cols = shape
    orows, ocols = out_shape or shape
    (south, west), (north, east) = latlon_bounds(grid, shape)
    lon = west + (np.arange(ocols) + 0.5) * (east - west) / ocols
    lat = north - (np.arange(orows) + 0.5) * (north - south) / orows
    x, y = Transformer.from_crs(4326, grid.epsg, always_xy=True).transform(*np.meshgrid(lon, lat))
    ri = np.rint((y - grid.y0) / grid.dy).astype(int)
    ci = np.rint((x - grid.x0) / grid.dx).astype(int)
    inside = (ri >= 0) & (ri < rows) & (ci >= 0) & (ci < cols)
    ri, ci = ri.clip(0, rows - 1), ci.clip(0, cols - 1)
    out = []
    for a in layers:
        r = a[ri, ci]
        out.append(r & inside if a.dtype == bool else np.where(inside, r, np.nan))
    return out, [[south, west], [north, east]]


# ---------- rendering ----------


def save_gray(db: np.ndarray, path: Path, stretch) -> None:
    lo, hi = stretch
    g = np.clip((db - lo) / (hi - lo) * 255, 0, 255)
    alpha = np.where(np.isfinite(db), 255, 0)
    Image.fromarray(np.dstack([np.nan_to_num(g), alpha]).astype("uint8"), "LA").save(path)


def save_rgb(hh, hv, path: Path, s_hh, s_hv, s_ratio) -> None:
    """Dual-pol false colour: red = HH, green = HV, blue = HH/HV ratio. Radar has no colour of its
    own; this is the standard composite that makes vegetation green-yellow, calm water dark and
    double bounce (flooded vegetation, buildings) red-magenta. Same stretches on both dates."""

    def ch(a, lo, hi):
        return np.nan_to_num(np.clip((a - lo) / (hi - lo) * 255, 0, 255))

    alpha = np.where(np.isfinite(hh) & np.isfinite(hv), 255, 0)
    rgba = np.dstack([ch(hh, *s_hh), ch(hv, *s_hv), ch(hh - hv, *s_ratio), alpha]).astype("uint8")
    Image.fromarray(rgba, "RGBA").save(path)


def save_change(kind: np.ndarray, path: Path) -> None:
    """kind: 0 none, DEC, INC (double bounce when HV exists), VOLUME, ROUGH."""
    rgba = np.zeros((*kind.shape, 4), "uint8")
    classes = ((DEC, DECREASE_RGB), (INC, INCREASE_RGB), (VOLUME, VOLUME_RGB), (ROUGH, ROUGH_RGB))
    for cls, rgb in classes:
        rgba[kind == cls] = (*rgb, 220)
    Image.fromarray(rgba, "RGBA").save(path)


CLASS_NAMES = {DEC: "decrease", INC: "increase", VOLUME: "volume", ROUGH: "rough_water"}


def top_regions(kind: np.ndarray, grid: Grid, n: int = 6, d=None, d_hv=None) -> list[dict]:
    """The n largest connected blobs of change on the native grid, with class, area and centre,
    so the page can number them and fly to them. Centres go through the UTM grid to lon/lat.
    With d / d_hv given, each region also carries its median HH and HV change; a large region
    whose two polarisations shifted alike is flagged `suspect`: real surface change rarely does
    that, a gain step or RFI block in one acquisition does (seen on a NISAR provisional granule)."""
    px = abs(grid.dx * grid.dy) / 1e6
    inv = Transformer.from_crs(grid.epsg, 4326, always_xy=True)
    found = []
    for cls in CLASS_NAMES:
        labels, count = ndimage.label(kind == cls, structure=np.ones((3, 3)))  # 8-connected
        if count == 0:
            continue
        sizes = np.bincount(labels.ravel())[1:]
        top = np.argsort(sizes)[::-1][:n]
        for idx in top:
            # Marker on the blob's deepest interior pixel, not its centroid: a curved lake or a
            # ragged blob has its centroid outside itself, and a click there would read nothing.
            sel = labels == idx + 1
            r, c = np.unravel_index(np.argmax(ndimage.distance_transform_edt(sel)), sel.shape)
            lon, lat = inv.transform(grid.x0 + c * grid.dx, grid.y0 + r * grid.dy)
            region = {
                "cls": CLASS_NAMES[cls],
                "km2": round(float(sizes[idx]) * px, 2),
                "lat": round(lat, 5),
                "lon": round(lon, 5),
            }
            if d is not None:
                region["d_hh"] = round(float(np.nanmedian(d[sel])), 1)
                if d_hv is not None:
                    region["d_hv"] = round(float(np.nanmedian(d_hv[sel])), 1)
                    region["suspect"] = bool(
                        region["km2"] >= 5 and abs(region["d_hh"] - region["d_hv"]) < 1.0
                    )
            found.append(region)
    # Under 0.05 km² (a few dozen pixels) is a speck, not a region worth sending someone to.
    found = [r for r in found if r["km2"] >= 0.05]
    found.sort(key=lambda r: -r["km2"])
    return [{"n": i + 1, **r} for i, r in enumerate(found[:n])]


def _tile_bytes(url: str) -> int:
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            return len(r.read())
    except Exception:  # noqa: BLE001 — a missing tile is just "no imagery"
        return 0


def _tile_xy(lon: float, lat: float, z: int = 12) -> tuple[int, int]:
    n = 2**z
    lat_r = math.radians(lat)
    x = int((lon + 180) / 360 * n)
    y = int((1 - math.log(math.tan(lat_r) + 1 / math.cos(lat_r)) / math.pi) / 2 * n)
    return x, y


def hls_dates(lat: float, lon: float, dates: list[str], max_days: int = 8) -> list[dict | None]:
    """For each radar date, the nearest day (within ±max_days) that has a real HLS tile at the
    box centre, and which HLS layer has it. An empty GIBS tile is a few hundred bytes."""
    x, y = _tile_xy(lon, lat)
    out: list[dict | None] = []
    for iso in dates:
        d0 = date.fromisoformat(iso[:10])
        hit = None
        for off in sorted(range(-max_days, max_days + 1), key=abs):
            d = (d0 + timedelta(days=off)).isoformat()
            for layer in HLS_LAYERS:
                if _tile_bytes(GIBS_TILE.format(layer=layer, date=d, x=x, y=y)) > 5000:
                    hit = {"date": d, "layer": layer}
                    break
            if hit:
                break
        out.append(hit)
    return out


# Areas of concentrated change. On a grid of AREA_BLOCK_M blocks, the share of the ground (land on
# at least one date) that changed is smoothed over AREA_WINDOW_M. Every block at least as dense as
# the scene as a whole (AREA_DENSITY_X) belongs to an area. That land is split into compact areas
# around its density peaks (at least AREA_PEAK_SPACING_M apart, watershed), and a piece holding
# less than the listing minimum is folded into the neighbour it shares most border with. Every area
# holding AREA_MIN_SHARE of all changed land (and AREA_MIN_KM2) is listed: no cap, so no
# concentration of that size is left without a number.
AREA_BLOCK_M = 50
AREA_WINDOW_M = 1500
AREA_DENSITY_X = 1.0
AREA_MIN_DENSITY = 0.02  # floor, so a scene with almost no change does not grow "areas" of noise
AREA_PEAK_SPACING_M = 1750
AREA_MIN_SHARE = 0.01
AREA_MIN_KM2 = 0.25
AREA_MAX = 60  # sanity cap only (labels are uint8)


def _odd(v: float) -> int:
    return max(3, int(round(v)) | 1)


def _merge_light(seg: np.ndarray, weight: np.ndarray, min_w: float) -> np.ndarray:
    """Fold each region lighter than min_w into the neighbour it shares the longest border with,
    lightest first, until no light region touches another region. Afterwards every region that
    touches another weighs at least min_w; a light region left over is a whole, isolated piece."""
    # Recomputes adjacency per merge (a few s for ~100 light pieces); use union-find if slow.
    seg = seg.copy()
    while True:
        w = ndimage.sum(weight, seg, index=np.arange(seg.max() + 1))
        a = np.concatenate([seg[:-1, :].ravel(), seg[:, :-1].ravel()])
        b = np.concatenate([seg[1:, :].ravel(), seg[:, 1:].ravel()])
        m = (a != b) & (a > 0) & (b > 0)
        if not m.any():
            return seg
        pairs, border = np.unique(np.sort(np.stack([a[m], b[m]], 1), 1), axis=0, return_counts=True)
        light = [(w[i], i) for i in np.unique(pairs) if w[i] < min_w]
        if not light:
            return seg
        i = min(light)[1]
        mine = (pairs[:, 0] == i) | (pairs[:, 1] == i)
        p = pairs[mine][np.argmax(border[mine])]
        seg[seg == i] = p[1] if p[0] == i else p[0]


def change_areas(kind: np.ndarray, ground: np.ndarray, grid: Grid):
    """Group scattered change into the compact areas where it concentrates.

    Change after a flood or fire is field-sized and scattered (thousands of small patches), so
    single patches, or a few "top" points, cover only a few percent of it. See the AREA_* notes for
    the rule. Wind on open water is not a change of the land and is not counted. Returns
    (areas, labels, rule): `labels` is the native-grid raster (0 = no listed area, n = area n),
    built from the same blocks the outline is traced from, so what is drawn and what is counted
    are the same pixels."""
    from skimage.feature import peak_local_max
    from skimage.measure import approximate_polygon, find_contours
    from skimage.segmentation import watershed

    dx = abs(grid.dx)
    px = abs(grid.dx * grid.dy) / 1e6
    land = (kind > 0) & (kind != ROUGH)
    ground = ground | land
    total = int(land.sum())
    mean = total / max(1, int(ground.sum()))
    thr = max(AREA_MIN_DENSITY, AREA_DENSITY_X * mean)
    min_px = max(AREA_MIN_KM2 / px, AREA_MIN_SHARE * total)
    rule = {
        "window_km": AREA_WINDOW_M / 1000,
        "density_x_mean": AREA_DENSITY_X,
        "scene_density_pct": round(100 * mean, 2),
        "threshold_density_pct": round(100 * thr, 2),
        "min_km2": round(min_px * px, 2),
        "total_km2": round(total * px, 2),
        "coverage_pct": 0.0,
    }
    rows, cols = kind.shape
    b = max(1, int(round(AREA_BLOCK_M / dx)))
    hb, wb = -(-rows // b), -(-cols // b)
    pad = ((0, hb * b - rows), (0, wb * b - cols))
    land_b = np.pad(land, pad).reshape(hb, b, wb, b).sum((1, 3)).astype("float64")
    ground_b = np.pad(ground, pad).reshape(hb, b, wb, b).sum((1, 3)).astype("float64")
    k = _odd(AREA_WINDOW_M / (b * dx))
    dens = _box_sum(land_b, k) / np.maximum(1.0, _box_sum(ground_b, k))
    mask = (dens >= thr) & (ground_b > 0)
    comp, ncomp = ndimage.label(mask)  # 4-connected, like the watershed
    if total == 0 or ncomp == 0:
        return [], np.zeros(kind.shape, "uint8"), rule
    seeds = np.zeros(mask.shape, "int32")
    spacing = max(1, int(round(AREA_PEAK_SPACING_M / (b * dx))))
    peaks = peak_local_max(dens, min_distance=spacing, labels=comp, exclude_border=False)
    for i, (r, c) in enumerate(peaks, start=1):
        seeds[r, c] = i
    # A component the peak finder skipped (flat, or narrower than the spacing) still needs a
    # seed, or the watershed would leave its change out.
    for ci, sl in enumerate(ndimage.find_objects(comp), start=1):
        inside = comp[sl] == ci
        if not seeds[sl][inside].any():
            r, c = np.unravel_index(np.argmax(np.where(inside, dens[sl], -1)), inside.shape)
            seeds[sl[0].start + r, sl[1].start + c] = seeds.max() + 1
    seg = _merge_light(watershed(-dens, seeds, mask=mask), land_b, min_px)
    # A pond or a thin spot enclosed by an area belongs to it: one clean outline per area. Only
    # unclaimed blocks are filled, so an area enclosed by another keeps its own number.
    for i, sl in enumerate(ndimage.find_objects(seg), start=1):
        if sl is not None:
            view = seg[sl]
            view[ndimage.binary_fill_holes(view == i) & (view == 0)] = i
    chg = ndimage.sum(land_b, seg, index=np.arange(seg.max() + 1))
    chg[0] = 0
    order = [int(i) for i in np.argsort(chg)[::-1] if chg[i] >= min_px][:AREA_MAX]
    seg_rank = np.zeros(seg.shape, "uint8")
    for rank, i in enumerate(order, start=1):
        seg_rank[seg == i] = rank
    labels = np.repeat(np.repeat(seg_rank, b, axis=0), b, axis=1)[:rows, :cols]
    inv = Transformer.from_crs(grid.epsg, 4326, always_xy=True)

    def block_ll(rr, cc):
        """Fractional block (row, col) -> (lon, lat); block edges land on pixel edges."""
        x = grid.x0 + (np.asarray(cc) * b + (b - 1) / 2) * grid.dx
        y = grid.y0 + (np.asarray(rr) * b + (b - 1) / 2) * grid.dy
        return inv.transform(x, y)

    nr = _odd(200 / dx)

    def marker(rank: int, rb: int, cb: int) -> tuple[int, int]:
        """Native pixel for the marker: in this area, within 150 m of block (rb, cb), the changed
        pixel with the most change around it (50 m, then 200 m), so it sits on a solid patch that
        survives the display resampling."""
        m = nr // 2
        r0, r1 = max(0, (rb - 1) * b - m), min(rows, (rb + 2) * b + m)
        c0, c1 = max(0, (cb - 1) * b - m), min(cols, (cb + 2) * b + m)
        win = land[r0:r1, c0:c1].astype("float64")
        score = _box_sum(win, 5) * 1e4 + _box_sum(win, nr)
        ok = (win > 0) & (labels[r0:r1, c0:c1] == rank)
        near = np.zeros(win.shape, bool)
        near[
            max(0, (rb - 1) * b) - r0 : (rb + 2) * b - r0,
            max(0, (cb - 1) * b) - c0 : (cb + 2) * b - c0,
        ] = True
        rr, cc = np.unravel_index(np.argmax(np.where(ok & near, score, -1.0)), win.shape)
        return r0 + int(rr), c0 + int(cc)

    areas = []
    for rank, i in enumerate(order, start=1):
        sel_b = seg_rank == rank
        rr, cc = np.nonzero(sel_b)
        wts = land_b[rr, cc]
        # Marker block: among blocks at least half as full as the area's typical changed block,
        # the one nearest the centroid of the area's changed land.
        cy, cx = (rr * wts).sum() / wts.sum(), (cc * wts).sum() / wts.sum()
        good = wts >= np.percentile(wts[wts > 0], 50)
        j = int(np.argmin(np.where(good, (rr - cy) ** 2 + (cc - cx) ** 2, np.inf)))
        mr, mc = marker(rank, int(rr[j]), int(cc[j]))
        mlon, mlat = inv.transform(grid.x0 + mc * grid.dx, grid.y0 + mr * grid.dy)
        # Outline: the boundary of the area's blocks, simplified by well under one block.
        r0, c0 = max(0, rr.min() - 1), max(0, cc.min() - 1)
        crop = np.pad(sel_b[r0 : rr.max() + 2, c0 : cc.max() + 2], 1)
        rings = []
        for ring in find_contours(crop.astype(float), 0.5):
            ring = approximate_polygon(ring, tolerance=0.6) - 1 + [r0, c0]
            lon, lat = block_ll(ring[:, 0], ring[:, 1])
            rings.append([[round(float(a), 5), round(float(o), 5)] for a, o in zip(lat, lon)])
        flat = [p for ring in rings for p in ring]
        sl = (slice(rr.min() * b, (rr.max() + 1) * b), slice(cc.min() * b, (cc.max() + 1) * b))
        zl = land[sl] & (labels[sl] == rank)
        counts = {c: int((kind[sl][zl] == c).sum()) for c in CLASS_NAMES if c != ROUGH}
        cnt = int(chg[i])
        ground_in = int(ground_b[sel_b].sum())
        areas.append(
            {
                "n": rank,
                "cls": CLASS_NAMES[max(counts, key=counts.get)],
                "km2": round(cnt * px, 2),
                "share_pct": round(100 * cnt / total, 1),
                "extent_km2": round(ground_in * px, 1),
                "density_pct": round(100 * cnt / max(1, ground_in), 1),
                "breakdown": {CLASS_NAMES[c]: round(v * px, 2) for c, v in counts.items() if v},
                "lat": round(mlat, 5),
                "lon": round(mlon, 5),
                "box": [
                    [min(p[0] for p in flat), min(p[1] for p in flat)],
                    [max(p[0] for p in flat), max(p[1] for p in flat)],
                ],
                "rings": rings,
            }
        )
    rule["coverage_pct"] = round(100 * sum(int(chg[i]) for i in order) / total, 1)
    return areas, labels, rule


@dataclass
class Classified:
    kind: np.ndarray  # 0 none, DEC, INC, VOLUME, ROUGH on the native grid
    pre_db: np.ndarray
    post_db: np.ndarray
    d: np.ndarray
    dark: np.ndarray
    dec: np.ndarray
    inc: np.ndarray
    hv_pre_db: np.ndarray | None
    hv_post_db: np.ndarray | None
    threshold_db: float
    auto: bool
    water_db: float = WATER_DB


def classify(pre, post, threshold_db: float | None = None, hv=None) -> Classified:
    """Every pixel of the scene into a change class. One function, used by the pipeline and by
    the tests, so what is drawn, counted and checked is always the same thing."""
    pre_db, post_db, d, dark = log_ratio(pre, post)
    if not np.isfinite(d).any():
        raise SystemExit("No radar samples on both dates inside the bbox - move it into the frame")
    auto = threshold_db is None
    if auto:
        threshold_db = auto_threshold(d)
    dec, inc = flag_change(d, threshold_db)
    kind = np.zeros(d.shape, "uint8")
    kind[dec], kind[inc] = DEC, INC
    # Dual-pol split of the brighter class, from what the data showed on the Meghna: wind on
    # open water lifts HH *and* HV but leaves the pixel water-dark, so the absolute level flags
    # it. Among pixels that became genuinely bright, a corner reflection between water and
    # vertical stems or walls lifts HH well above HV (double bounce = flooded vegetation,
    # structures), while growth or a rougher surface lifts both alike. One class without HV.
    hv_pre_db = hv_post_db = None
    if hv is not None:
        hv_pre_db, hv_post_db = to_db(box_mean(hv[0])), to_db(box_mean(hv[1]))
        ratio_gain = d - (hv_post_db - hv_pre_db)  # how much more HH rose than HV
        bright = post_db > ROUGH_MAX_DB
        kind[inc & ~bright] = ROUGH
        kind[inc & bright & ~(ratio_gain >= threshold_db / 2)] = VOLUME
    return Classified(
        kind,
        pre_db,
        post_db,
        d,
        dark,
        dec,
        inc,
        hv_pre_db,
        hv_post_db,
        threshold_db,
        auto,
        water_line(np.stack([pre_db, post_db])),
    )


def write_event(meta: dict, pre, post, grid: Grid, threshold_db: float | None, hv=None) -> Path:
    out = settings.events_dir / meta["id"]
    out.mkdir(parents=True, exist_ok=True)
    c = classify(pre, post, threshold_db, hv)
    kind, pre_db, post_db, d, dark, dec, inc = (
        c.kind,
        c.pre_db,
        c.post_db,
        c.d,
        c.dark,
        c.dec,
        c.inc,
    )
    hv_pre_db, hv_post_db, threshold_db, auto = c.hv_pre_db, c.hv_post_db, c.threshold_db, c.auto
    # One grayscale stretch for BOTH dates (2nd–98th percentile of the pre image), otherwise a
    # before/after comparison would show a contrast change that is not in the data.
    stretch = [round(float(v), 1) for v in np.nanpercentile(pre_db, [2, 98])]
    f = max(1, int(np.ceil(max(d.shape) / MAX_DISPLAY_PX)))
    blk, subtle_thr = subtle_change(d, pre_db, post_db, grid, c.water_db)
    # Ground = land on at least one date; the share of it that changed is what areas measure.
    ground = np.isfinite(d) & (np.fmax(pre_db, post_db) > c.water_db)
    d_hv = None if hv is None else hv_post_db - hv_pre_db
    areas, area_labels, area_rule = change_areas(kind, ground, grid)
    layers = [pre_db, post_db, kind.astype("float32"), blk, area_labels.astype("float32")]
    if hv is not None:
        layers += [hv_pre_db, hv_post_db]
    out_ll, bounds = to_latlon(layers, grid, d.shape, (d.shape[0] // f, d.shape[1] // f))
    pre_ll, post_ll = out_ll[0], out_ll[1]
    kind_ll = np.nan_to_num(out_ll[2], nan=0).astype("uint8")
    save_subtle(out_ll[3], subtle_thr, out / "subtle.png")
    areas_ll = np.nan_to_num(out_ll[4], nan=0).astype("uint8")
    np.save(out / "areas.npy", areas_ll)
    save_gray(pre_ll, out / "before.png", stretch)
    save_gray(post_ll, out / "after.png", stretch)
    save_change(kind_ll, out / "change.png")
    # Per-pixel values on the display grid for the click-to-probe endpoint (float16: 2 B/px).
    np.save(out / "pre_db.npy", pre_ll.astype("float16"))
    np.save(out / "post_db.npy", post_ll.astype("float16"))
    np.save(out / "flags.npy", kind_ll)
    layer_files = {
        "before": "before.png",
        "after": "after.png",
        "change": "change.png",
        "subtle": "subtle.png",
    }
    if hv is not None:
        np.save(out / "hv_pre_db.npy", out_ll[5].astype("float16"))
        np.save(out / "hv_post_db.npy", out_ll[6].astype("float16"))
        s_hv = np.nanpercentile(hv_pre_db, [2, 98])
        s_ratio = np.nanpercentile(pre_db - hv_pre_db, [2, 98])
        save_rgb(pre_ll, out_ll[5], out / "before_rgb.png", stretch, s_hv, s_ratio)
        save_rgb(post_ll, out_ll[6], out / "after_rgb.png", stretch, s_hv, s_ratio)
        layer_files |= {"before_rgb": "before_rgb.png", "after_rgb": "after_rgb.png"}
    stats = summarize(d, dec, inc, dark, abs(grid.dx * grid.dy) / 1e6)
    # Strict class for the headline: was land, is now at open-water darkness, and flagged.
    new_water = dec & (pre_db > c.water_db + 3) & (post_db < c.water_db)
    stats["open_water_km2"] = round(int(new_water.sum()) * abs(grid.dx * grid.dy) / 1e6, 2)
    stats["water_line_db"] = c.water_db
    px_km2 = abs(grid.dx * grid.dy) / 1e6
    stats["subtle_threshold_db"] = subtle_thr
    stats["subtle_increase_km2"] = round(int(np.nansum(blk > subtle_thr)) * px_km2, 2)
    stats["subtle_decrease_km2"] = round(int(np.nansum(blk < -subtle_thr)) * px_km2, 2)
    if hv is not None:
        px = abs(grid.dx * grid.dy) / 1e6
        stats["increase_double_bounce_km2"] = round(int((kind == INC).sum()) * px, 2)
        stats["increase_volume_km2"] = round(int((kind == VOLUME).sum()) * px, 2)
        stats["increase_rough_water_km2"] = round(int((kind == ROUGH).sum()) * px, 2)
    event = {
        **meta,
        "product": "NISAR L2 GCOV (L-band, RTC gamma0)",
        "threshold_db": threshold_db,
        "threshold_auto": auto,
        "cross_pol": hv is not None,
        "display_db": stretch,
        "bounds": bounds,
        "stats": stats,
        "regions": areas,
        "area_rule": area_rule,
        # Large single blobs that shifted alike in HH and HV: possible acquisition artefacts.
        "suspects": [r for r in top_regions(kind, grid, d=d, d_hv=d_hv) if r.get("suspect")],
        # Optical companions from NASA GIBS, one per radar date (None when nothing within ±8 days)
        "hls": None
        if meta.get("synthetic")
        else hls_dates(
            (bounds[0][0] + bounds[1][0]) / 2,
            (bounds[0][1] + bounds[1][1]) / 2,
            [meta["pre"]["datetime"], meta["post"]["datetime"]],
        ),
        "layers": layer_files,
    }
    (out / "event.json").write_text(json.dumps(event, indent=2))
    add_ground(out)  # ground on each date, so the before side of the swipe is coloured too
    return out


# ---------- real NISAR input ----------


def _login():
    """Earthdata Login through NASA's earthaccess: EARTHDATA_TOKEN from .env, else
    EARTHDATA_USERNAME/EARTHDATA_PASSWORD, else ~/.netrc, else an interactive prompt."""
    import earthaccess

    if settings.earthdata_token:
        os.environ.setdefault("EARTHDATA_TOKEN", settings.earthdata_token)
    if not earthaccess.login(persist=True).authenticated:
        raise SystemExit(
            "Earthdata Login failed. Put EARTHDATA_TOKEN=... (urs.earthdata.nasa.gov > Generate "
            "Token) or EARTHDATA_USERNAME= / EARTHDATA_PASSWORD= in sars-backend/.env"
        )
    return earthaccess


def open_gcov(src: str) -> h5py.File:
    """A local .h5 path, or an Earthdata Cloud URL read in place: h5py over an fsspec file, so
    only the HDF5 chunks under the bbox are fetched instead of the whole 1–2 GB granule."""
    if not src.startswith("http"):
        return h5py.File(src, "r")
    fobj = _login().open([src])[0]
    # NISAR HDF5 is cloud-optimised (paged); a large raw-chunk cache keeps the request count down,
    # sized to the memory this machine actually has free.
    cache = read_cache_bytes(available_memory())
    return h5py.File(fobj, "r", rdcc_nbytes=cache, rdcc_nslots=50_021)


def fetch(src: str) -> str:
    """--download: pull the whole granule once into data/granules/ and read it locally."""
    if not src.startswith("http"):
        return src
    cache = settings.data_dir / "granules"
    cache.mkdir(parents=True, exist_ok=True)
    out = cache / src.rsplit("/", 1)[-1]
    if not out.exists():
        _login().download([src], str(cache))
    return str(out)


CRID_MATURITY = {"P05023": "PROVISIONAL", "P05036": "VALIDATED"}
ORBIT_QUALITY = {"P": "precise", "M": "medium", "N": "near-real-time", "F": "forecast"}


def maturity(src: str) -> dict:
    """What the file name and its collection say about the product's maturity: the NISAR Data
    User Guide's naming convention (CRID = composite release id; A = orbit/pointing accuracy;
    PT = processing type) and the collection folder in the download URL."""
    name = src.rsplit("/", 1)[-1]
    m = re.search(r"_(?P<crid>[A-Z]\d{5})_(?P<a>[PMNF])_(?P<c>[FP])_[A-Z]_\d{3}", name)
    coll = re.search(r"NISAR_L2_GCOV_(PROVISIONAL|BETA)_V1", src)
    out: dict = {}
    if coll:
        out["maturity"] = coll.group(1)
    elif "NISAR_L2_GCOV_V1" in src:
        out["maturity"] = "VALIDATED"
    if m:
        out.setdefault("maturity", CRID_MATURITY.get(m.group("crid")))
        out["crid"] = m.group("crid")
        out["orbit_quality"] = ORBIT_QUALITY.get(m.group("a"))
        out["frame_coverage"] = {"F": "full", "P": "partial"}.get(m.group("c"))
    return out


def stack_key(name: str) -> str:
    """What repeat passes share: everything between the cycle number and the start time."""
    m = re.search(r"GCOV_\d+_(.+?)_\d{8}T\d{6}_", name)
    return m.group(1) if m else name


def search(a: argparse.Namespace) -> None:
    """List GCOV granules over a bbox grouped by pass geometry; pick a pair from ONE group."""
    import earthaccess  # CMR search is public: no login needed

    w, s, e, n = a.bbox
    groups: dict[str, list[tuple]] = {}
    footprints: dict[str, str] = {}  # frames are tilted; this is the enclosing box
    for coll in GCOV_COLLECTIONS:
        label = coll.removeprefix("NISAR_L2_GCOV_").removesuffix("_V1") or "VALIDATED"
        found = earthaccess.search_data(
            short_name=coll, bounding_box=(w, s, e, n), temporal=(a.start, a.end), count=500
        )
        for g in found:
            u = g["umm"]
            # CMR's STACK_ID is only track/dir/frame; one date can carry several modes and
            # polarization sets, and a pair must match those too, so group by the full key.
            key = stack_key(u["GranuleUR"])
            url = next((link for link in g.data_links() if link.endswith(".h5")), "?")
            start = u["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"][:16]
            size = g.size() if callable(g.size) else g.size
            groups.setdefault(key, []).append((start, round(float(size)), label, url))
            try:
                geom = u["SpatialExtent"]["HorizontalSpatialDomain"]["Geometry"]
                pts = geom["GPolygons"][0]["Boundary"]["Points"]
                lon = [q["Longitude"] for q in pts]
                lat = [q["Latitude"] for q in pts]
                footprints.setdefault(
                    key, f"{min(lon):.2f} {min(lat):.2f} {max(lon):.2f} {max(lat):.2f}"
                )
            except (KeyError, IndexError):
                pass
    if not groups:
        raise SystemExit("No GCOV granules there in that window (public data starts 2026-06-17).")
    lines = []
    for key, items in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        items.sort()
        lines.append(
            f"\n# {key}: {len(items)} pass(es), footprint W S E N = {footprints.get(key, '?')}. "
            "Pick a pair from ONE group and a bbox well inside the footprint."
        )
        lines += [f"  {start}  {mb:>5} MB  {label:<11} {url}" for start, mb, label, url in items]
    print("\n".join(lines))
    # Keep a copy with the project: the URLs are long and /tmp does not survive a reboot.
    out = settings.data_dir / "searches" / f"search_{'_'.join(f'{v:g}' for v in a.bbox)}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"\n(saved to {out})")


def eonet_rows(data: dict, pad: float = 0.15) -> list[tuple[str, str, str, str]]:
    """(date, categories, title, bbox) per EONET event; points and polygons become a padded box."""
    rows = []
    for ev in data.get("events", []):
        pts = []
        for geom in ev.get("geometry", [])[-1:]:  # latest position only (storm tracks move)
            coords = geom.get("coordinates", [])
            if coords and isinstance(coords[0], (int, float)):
                pts.append(coords)
            else:
                pts += [p for ring in coords for p in ring]
        if not pts or not ev.get("geometry"):
            continue
        lon = [p[0] for p in pts]
        lat = [p[1] for p in pts]
        bbox = (
            f"{min(lon) - pad:.2f} {min(lat) - pad:.2f} {max(lon) + pad:.2f} {max(lat) + pad:.2f}"
        )
        cats = ", ".join(c["title"] for c in ev.get("categories", []))
        rows.append((ev["geometry"][-1]["date"][:10], cats, ev["title"], bbox))
    return sorted(rows, reverse=True)


# Optical basemaps kept on disk: GIBS sends them with no-store, so without a local copy every
# view refetches them. NASA imagery is open data; the Esri and OpenStreetMap basemaps forbid
# offline copies, so those stay with the browser's own cache.
GIBS_URL = "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/{layer}/default/{date}/{tms}/{z}/{y}/{x}.{ext}"
MODIS_LAYER = "MODIS_Terra_CorrectedReflectance_TrueColor"


def tile_jobs(event: dict) -> list[tuple[str, str, str, int, str]]:
    """(layer, date, tile matrix set, max zoom, ext) for every dated optical layer the map shows:
    HLS on the scene each radar date matched, MODIS on the radar dates themselves."""
    jobs = [
        (h["layer"], h["date"], "GoogleMapsCompatible_Level12", 12, "png")
        for h in event.get("hls") or []
        if h
    ]
    for g in (event["pre"], event["post"]):
        jobs.append((MODIS_LAYER, g["datetime"][:10], "GoogleMapsCompatible_Level9", 9, "jpg"))
    return jobs


# The offline copy covers what a screen this size shows: at every zoom from two levels out of the
# event's opening view down to the layer's finest, the scene plus half a screen on every side (one
# tile more across, because the scene opens left of the sidebar, not centred).
SCREEN_PX = (1920, 1080)


def _world_px(lon: float, lat: float) -> tuple[float, float]:
    """Web Mercator position in pixels of the 256 px zoom-0 world."""
    lat_r = math.radians(lat)
    y = (1 - math.log(math.tan(lat_r) + 1 / math.cos(lat_r)) / math.pi) / 2
    return (lon + 180) / 360 * 256, y * 256


def tile_ranges(bounds, max_z: int) -> list[tuple[int, int, int, int, int]]:
    """(z, x0, x1, y0, y1) of the tiles a SCREEN_PX screen can show around the scene, per zoom."""
    (s, w), (n, e) = bounds
    x0, y0 = _world_px(w, n)
    x1, y1 = _world_px(e, s)
    # opening view: the scene fitted beside the sidebar (424 px) inside the map's padding
    free_w, free_h = SCREEN_PX[0] - 424 - 64, SCREEN_PX[1] - 168
    fit = math.log2(min(free_w / (x1 - x0), free_h / (y1 - y0)))
    hx, hy = SCREEN_PX[0] / 2 + 256, SCREEN_PX[1] / 2 + 128
    out = []
    for z in range(min(max(0, int(fit) - 2), max_z), max_z + 1):
        k, last = 2**z, 2**z - 1
        out.append(
            (
                z,
                max(0, int((x0 * k - hx) // 256)),
                min(last, int((x1 * k + hx) // 256)),
                max(0, int((y0 * k - hy) // 256)),
                min(last, int((y1 * k + hy) // 256)),
            )
        )
    return out


def download_tiles(event_id: str, workers: int = 4) -> tuple[int, int, int]:
    """Fetch the event's GIBS tiles into data/tiles/<layer>/<date>/<z>/<y>/<x>.<ext>, served at
    /files/tiles/... Existing tiles are kept. Returns (fetched, kept, missing)."""
    from concurrent.futures import ThreadPoolExecutor

    event = json.loads((settings.events_dir / event_id / "event.json").read_text())
    todo = []
    for layer, day, tms, max_z, ext in tile_jobs(event):
        for z, x0, x1, y0, y1 in tile_ranges(event["bounds"], max_z):
            for x in range(x0, x1 + 1):
                for y in range(y0, y1 + 1):
                    path = (
                        settings.data_dir / "tiles" / layer / day / str(z) / str(y) / f"{x}.{ext}"
                    )
                    url = GIBS_URL.format(layer=layer, date=day, tms=tms, z=z, y=y, x=x, ext=ext)
                    todo.append((url, path))

    def get(job) -> str:
        url, path = job
        if path.exists():
            return "kept"
        req = urllib.request.Request(url, headers={"User-Agent": "SARChangeWatch/0.1"})
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    body = r.read()
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(body)
                return "fetched"
            except urllib.error.HTTPError as err:
                if err.code in (400, 404):
                    return "missing"  # outside the layer's coverage
            except OSError:
                pass
            time.sleep(2 * (attempt + 1))
        return "missing"

    with ThreadPoolExecutor(workers) as pool:
        got = list(pool.map(get, todo))
    return got.count("fetched"), got.count("kept"), got.count("missing")


def tiles(a: argparse.Namespace) -> None:
    ids = a.id or sorted(
        d.name for d in settings.events_dir.iterdir() if (d / "event.json").exists()
    )
    for event_id in ids:
        fetched, kept, missing = download_tiles(event_id)
        print(
            f"{event_id}: {fetched} tiles fetched, {kept} already on disk, {missing} not available"
        )
    size = sum(f.stat().st_size for f in (settings.data_dir / "tiles").rglob("*") if f.is_file())
    print(f"data/tiles: {size / 2**20:.0f} MB")


def ground(a: argparse.Namespace) -> None:
    """Add the ground-on-each-date layer to processed events (no radar download needed)."""
    folders = (
        [settings.events_dir / i for i in a.id]
        if a.id
        else sorted(p.parent for p in settings.events_dir.glob("*/event.json"))
    )
    for folder in folders:
        if not (folder / "pre_db.npy").is_file():
            print(f"{folder.name}: no per-pixel data, skipped")
            continue
        stats = add_ground(folder)
        row = ", ".join(f"{k} {v['before_km2']:g} -> {v['after_km2']:g}" for k, v in stats.items())
        print(f"{folder.name}: km² {row}")


def eonet(a: argparse.Namespace) -> None:
    """NASA EONET: open natural events of the last N days, each with a ready-made --bbox."""
    url = f"https://eonet.gsfc.nasa.gov/api/v3/events?status=open&days={a.days}&limit=200"
    if a.category:
        url += f"&category={a.category}"
    with urllib.request.urlopen(url, timeout=60) as r:
        data = json.load(r)
    rows = eonet_rows(data)
    if not rows:
        raise SystemExit("EONET lists no open events for that window / category.")
    for when, cats, title, bbox in rows:
        print(f"{when}  {cats[:18]:<18} {title[:58]:<58}  --bbox {bbox}")
    print("\nNext: pipeline search --bbox <box> --start <about two months before the event date>")


def _ident(f: h5py.File, key: str):
    ds = f.get(f"{IDENT}/{key}")
    if ds is None:
        return None
    v = ds[()]
    v = v.decode() if isinstance(v, bytes) else v.item() if hasattr(v, "item") else v
    return v.decode() if isinstance(v, bytes) else v


def _footprint(f: h5py.File) -> str:
    """Lon/lat box of the granule's boundingPolygon (WKT, 2-D or 3-D vertices)."""
    pts = re.findall(
        r"(-?\d+\.?\d*)\s+(-?\d+\.?\d*)(?:\s+-?\d+\.?\d*)?", _ident(f, "boundingPolygon") or ""
    )
    if not pts:
        return "unknown"
    lon = [float(a) for a, _ in pts]
    lat = [float(b) for _, b in pts]
    return f"{min(lon):.2f} {min(lat):.2f} {max(lon):.2f} {max(lat):.2f}"


def _transient(e: BaseException) -> bool:
    """Network hiccups worth a retry (DNS, resets, timeouts) — not bad paths or bad data."""
    return (
        isinstance(e, (OSError, TimeoutError, ConnectionError)) or "aiohttp" in type(e).__module__
    )


def read_gcov(src: str, bbox: list[float]) -> tuple[np.ndarray, Grid, Granule]:
    """Cropped backscatter + grid + provenance for one granule. The subset is cached on disk,
    so a re-run (new threshold, dropped wifi on granule two) never streams the same bytes twice."""
    name = src.rsplit("/", 1)[-1].removesuffix(".h5")
    cache = settings.data_dir / "subsets" / f"{name}_{'_'.join(f'{v:g}' for v in bbox)}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        e, x0, y0, dx, dy = z["grid"]
        print(f"{name}: using cached subset {cache.name}")
        hv = z["hv"] if "hv" in z.files else None
        granule = Granule.model_validate_json(str(z["granule"]))
        return z["data"], Grid(int(e), x0, y0, dx, dy), granule, hv
    for attempt in range(1, 4):
        try:
            data, grid, granule, hv = _read_gcov(src, name, bbox)
            break
        except Exception as e:  # noqa: BLE001 — filtered right below
            if attempt == 3 or not _transient(e):
                raise
            print(f"{name}: {type(e).__name__}: {e} — retry {attempt}/2 in {15 * attempt}s")
            time.sleep(15 * attempt)
    cache.parent.mkdir(parents=True, exist_ok=True)
    grid_row = np.array([grid.epsg, grid.x0, grid.y0, grid.dx, grid.dy], "float64")
    extra = {"hv": hv} if hv is not None else {}
    np.savez_compressed(
        cache, data=data, grid=grid_row, granule=np.array(granule.model_dump_json()), **extra
    )
    return data, grid, granule, hv


def _read_gcov(src: str, name: str, bbox: list[float]) -> tuple[np.ndarray, Grid, Granule]:
    with open_gcov(src) as f:
        g = f[GCOV_GRID]
        pol = next((p for p in POLARIZATIONS if p in g), None)
        if pol is None:
            raise SystemExit(f"{name}: no {POLARIZATIONS} layer, found {list(g)}")
        x, y = g["xCoordinates"][:], g["yCoordinates"][:]
        epsg = int(g["projection"].attrs["epsg_code"])
        w, s, e, n = bbox
        fx, fy = Transformer.from_crs(4326, epsg, always_xy=True).transform(
            [w, e, w, e], [s, s, n, n]
        )
        cols = np.flatnonzero((x >= min(fx)) & (x <= max(fx)))
        rows = np.flatnonzero((y >= min(fy)) & (y <= max(fy)))
        if cols.size == 0 or rows.size == 0:
            raise SystemExit(f"{name}: bbox does not overlap this granule")
        window = (slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))
        print(f"{name}: reading {pol} rows {rows[0]}-{rows[-1]} cols {cols[0]}-{cols[-1]}")
        data = g[pol][window].astype("float32")
        # GCOV's `mask` is ISCE3's valid-samples sub-swath mask (GeocodeCov.cpp): 1..N = valid
        # sample from that sub-swath, 0 = invalid / partially focused, 255 = fill outside the
        # swath. Layover and shadow were already removed upstream (NaN in the backscatter).
        if "mask" in g:
            m = g["mask"][window]
            data[m == 0] = np.nan
            inside = float(np.mean((m >= 1) & (m < 255)))
            print(f"{name}: {inside:.0%} of the bbox has radar samples")
            if inside < 0.5:
                print(
                    f"{name}: WARNING most of the bbox is outside this frame; its footprint "
                    f"is about W S E N = {_footprint(f)}"
                )
        # Cross-pol term for the double-bounce test (HV for H transmit, VH for V transmit).
        cross = CROSS_POL.get(pol)
        hv = None
        if cross and cross in g:
            hv = g[cross][window].astype("float32")
            hv[~np.isfinite(data)] = np.nan
        x0, y0 = float(x[cols[0]]), float(y[rows[0]])
        grid = Grid(epsg, x0, y0, float(x[1] - x[0]), float(y[1] - y[0]))
        granule = Granule(
            id=name,
            datetime=str(_ident(f, "zeroDopplerStartTime") or ""),
            polarization=pol[:2],
            track=_ident(f, "trackNumber"),
            frame=_ident(f, "frameNumber"),
            orbit_direction=_ident(f, "orbitPassDirection"),
            product_version=_ident(f, "productVersion"),
        )
    return data, grid, granule, hv


def run(a: argparse.Namespace) -> None:
    # Measured before the radar is read, so the inputs are not counted twice.
    available = available_memory()
    pre_src, post_src = (fetch(a.pre), fetch(a.post)) if a.download else (a.pre, a.post)
    pre, g1, m1, hv1 = read_gcov(pre_src, a.bbox)
    post, g2, m2, hv2 = read_gcov(post_src, a.bbox)
    if (
        g1.epsg != g2.epsg
        or g1.dx != g2.dx
        or abs(g1.x0 - g2.x0) > abs(g1.dx) / 2
        or abs(g1.y0 - g2.y0) > abs(g1.dy) / 2
    ):
        raise SystemExit("Pre/post grids differ — pick two granules from the same track & frame")
    h, w = min(pre.shape[0], post.shape[0]), min(pre.shape[1], post.shape[1])
    # Full resolution on a capable machine; NxN multilook only when memory is short.
    looks = choose_looks(h * w, available, a.looks)
    grid = multilook_grid(g1, looks)
    if looks > 1:
        free = f"{available / 2**20:.0f} MB free" if available else "forced"
        print(
            f"{h * w / 1e6:.1f} M px: averaging {looks}x{looks} -> {abs(grid.dx):g} m pixels "
            f"({free}); a computer with more memory keeps {abs(g1.dx):g} m"
        )
    pre, post = multilook(pre[:h, :w], looks), multilook(post[:h, :w], looks)
    hv = None
    if hv1 is not None and hv2 is not None:
        hv = (multilook(hv1[:h, :w], looks), multilook(hv2[:h, :w], looks))
    meta = dict(
        id=a.id, title=a.title, hazard=a.hazard, location=a.location,
        description=a.description, synthetic=False,
        pre=m1.model_dump() | maturity(a.pre), post=m2.model_dump() | maturity(a.post),
        looks=looks, pixel_m=round(abs(grid.dx), 2),
    )  # fmt: skip
    print("wrote", write_event(meta, pre, post, grid, a.threshold, hv))


# ---------- synthetic demo (for UI work before real granules are processed) ----------


def synthetic_scene(seed: int = 7):
    """Speckled L-band look-alike with a known flood. Returns (pre, post, flood, trees):
    `flood` is new open water (backscatter drops), `trees` flooded vegetation (rises)."""
    rng = np.random.default_rng(seed)
    h, w = 400, 520
    yy, xx = np.mgrid[0:h, 0:w]
    centre = 200 + 40 * np.sin(xx / 60)
    river = np.abs(yy - centre) < 7
    flood = (np.abs(yy - centre) < 45) & (xx > 120) & (xx < 430) & ~river
    trees = (xx - 70) ** 2 + (yy - 80) ** 2 < 22**2
    land = 0.06 * (1 + 0.5 * np.sin(xx / 37) * np.cos(yy / 53))
    pre = np.where(river, 0.003, land)
    post = np.where(flood, 0.003, np.where(trees, land * 5, pre))

    def speckle():
        return rng.gamma(4, 1 / 4, (h, w))  # 4-look multiplicative speckle

    return pre * speckle(), post * speckle(), flood, trees


def demo(_: argparse.Namespace) -> None:
    pre, post, _, _ = synthetic_scene()
    grid = Grid(epsg=32646, x0=760_000.0, y0=2_640_000.0, dx=20.0, dy=-20.0)
    meta = dict(
        id="synthetic-demo", title="River flood (synthetic demo)", hazard="flood",
        location="Synthetic scene (placement arbitrary)",
        description="Simulated speckled L-band backscatter with an injected flood. Used only to "
        "develop the dashboard; replace with a real GCOV pair via `pipeline run`.",
        synthetic=True,
        pre=dict(id="SYNTHETIC_PRE", datetime="2026-07-01T00:00:00Z", polarization="HH"),
        post=dict(id="SYNTHETIC_POST", datetime="2026-07-13T00:00:00Z", polarization="HH"),
    )  # fmt: skip
    print("wrote", write_event(meta, pre, post, grid, 3.0))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(required=True)
    r = sub.add_parser("run", help="process a real pre/post GCOV pair")
    r.add_argument("--id", required=True)
    r.add_argument("--pre", required=True, help="GCOV .h5 URL (ASF) or local path")
    r.add_argument("--post", required=True)
    r.add_argument("--bbox", type=float, nargs=4, required=True, metavar=("W", "S", "E", "N"))
    r.add_argument("--title", required=True)
    r.add_argument("--hazard", default="flood")
    r.add_argument("--location", default="")
    r.add_argument("--description", default="")
    r.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="|change| in dB; default max(3, 3 x scene noise)",
    )
    r.add_argument("--download", action="store_true", help="fetch whole files, don't stream")
    r.add_argument(
        "--looks",
        type=int,
        choices=range(1, MAX_LOOKS + 1),
        default=None,
        help="NxN multilook; default: 1 (full resolution) unless memory is short",
    )
    r.set_defaults(fn=run)
    s = sub.add_parser("search", help="list GCOV granules over a bbox, grouped by pass")
    s.add_argument("--bbox", type=float, nargs=4, required=True, metavar=("W", "S", "E", "N"))
    s.add_argument("--start", default="2026-06-17", help="YYYY-MM-DD (public data starts here)")
    s.add_argument("--end", default=date.today().isoformat())
    s.set_defaults(fn=search)
    n = sub.add_parser("eonet", help="list open NASA EONET events (floods, fires, storms...)")
    n.add_argument("--days", type=int, default=60)
    n.add_argument("--category", default="", help="e.g. floods, wildfires, severeStorms, volcanoes")
    n.set_defaults(fn=eonet)
    sub.add_parser("demo", help="write a synthetic, clearly-labelled event").set_defaults(fn=demo)
    t = sub.add_parser("tiles", help="download the NASA GIBS photo tiles of events for offline use")
    t.add_argument("--id", nargs="*", help="event ids (default: every processed event)")
    t.set_defaults(fn=tiles)
    gr = sub.add_parser("ground", help="add the ground-on-each-date layer to processed events")
    gr.add_argument("--id", nargs="*", help="event ids (default: every processed event)")
    gr.set_defaults(fn=ground)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
