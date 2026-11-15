"""What the ground looked like to the radar on each date, in four plain classes.

The change layer only exists on the after date (it is a difference), so the before side of the
swipe used to be plain gray. This colours BOTH dates with the same classes and the same
thresholds, so dragging the divider shows what grew and what shrank: water spreading, mud
replacing fields, rubble appearing.

Works from the display-grid arrays every event already keeps (pre_db.npy / post_db.npy), so
existing events gain the layer without fetching the radar again (`pipeline ground`).
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image

# Classes written to before_ground.png / after_ground.png (0 = no radar sample).
WATER, SMOOTH, MEDIUM, BRIGHT = 1, 2, 3, 4
GROUND_NAMES = {WATER: "water", SMOOTH: "smooth", MEDIUM: "medium", BRIGHT: "bright"}

# Typical L-band HH levels after the 5x5 speckle filter. Must match LEVEL_DARK_DB /
# LEVEL_BRIGHT_DB in sars-frontend/src/lib/reading.ts (the hover card uses the same words).
GROUND_DARK_DB = -12.0  # below: smooth ground, bare soil, mud, short grass
GROUND_BRIGHT_DB = -6.0  # above: forest, buildings, rock, rubble, slopes facing the satellite
DEFAULT_WATER_DB = -16.0  # when an event did not record its own water line
# Water ruffled by wind rises above the water line but stays this dark; water on one date and at
# most this on the other is still water (pipeline.ROUGH_MAX_DB, the "wind on water" class).
ROUGH_MAX_DB = -9.0
# Change-layer flags (flags.npy) where that rule may apply: no significant change, or the
# change layer's own "wind on water" (pipeline.ROUGH). A flagged darker/brighter pixel is a real
# change and keeps each date's own class, so a flooded wet field still shows as new water.
NO_CHANGE, ROUGH_FLAG = 0, 4

# Must match --ground-* in sars-frontend/src/index.css.
GROUND_RGB = {
    WATER: (0x1B, 0x3F, 0x7A),  # deep blue
    SMOOTH: (0xC8, 0xA2, 0x6B),  # sand / mud
    MEDIUM: (0x7F, 0xAE, 0x5A),  # green
    BRIGHT: (0xF1, 0xEA, 0xD8),  # pale stone
}
GROUND_ALPHA = 230

EARTH_KM_PER_DEG = 111.32


def ground_classes(db: np.ndarray, water_db: float) -> np.ndarray:
    """uint8 class per pixel; 0 where there is no radar sample."""
    finite = np.isfinite(db)
    v = np.where(finite, db, 0.0)
    out = np.full(db.shape, BRIGHT, "uint8")
    out[v < GROUND_BRIGHT_DB] = MEDIUM
    out[v < GROUND_DARK_DB] = SMOOTH
    out[v < water_db] = WATER
    out[~finite] = 0
    return out


def ground_pair(pre_db: np.ndarray, post_db: np.ndarray, water_db: float, flags=None):
    """Ground classes of both dates.

    With the change flags (the normal case), the after date differs from the before date ONLY
    where the change layer measured a real change (darker / double bounce / rougher). Elsewhere
    it keeps the before class: a shift below the noise threshold that happens to cross a class
    line is not a change of ground (on Thunder Bay a +1 dB drift otherwise moved 200 km² from
    "fields" to "bright"), and wind on a lake is not a lake drying out.

    Without flags, each date gets its own class, except that water on one date which is still
    water-dark on the other stays water on both.
    """
    pre, post = ground_classes(pre_db, water_db), ground_classes(post_db, water_db)
    if flags is not None:
        changed = ~np.isin(flags, (NO_CHANGE, ROUGH_FLAG))
        keep = ~changed & (pre > 0) & (post > 0)
        return pre, np.where(keep, pre, post)
    with np.errstate(invalid="ignore"):
        ruffled_after = (pre == WATER) & (post > 0) & (post_db <= ROUGH_MAX_DB)
        ruffled_before = (post == WATER) & (pre > 0) & (pre_db <= ROUGH_MAX_DB)
    return np.where(ruffled_before, WATER, pre), np.where(ruffled_after, WATER, post)


def save_ground(cls: np.ndarray, path: Path) -> None:
    rgba = np.zeros((*cls.shape, 4), "uint8")
    for k, rgb in GROUND_RGB.items():
        rgba[cls == k] = (*rgb, GROUND_ALPHA)
    Image.fromarray(rgba, "RGBA").save(path)


def cell_km2(bounds, shape) -> np.ndarray:
    """Area of each display-grid cell (a regular lon/lat grid), per row: shrinks with cos(lat)."""
    (south, west), (north, east) = bounds
    rows, cols = shape
    dlat = (north - south) / rows
    dlon = (east - west) / cols
    lat = north - (np.arange(rows) + 0.5) * dlat
    return (dlat * EARTH_KM_PER_DEG) * (dlon * EARTH_KM_PER_DEG * np.cos(np.radians(lat)))


def ground_stats(pre: np.ndarray, post: np.ndarray, bounds) -> dict[str, dict[str, float]]:
    """km² of each class on each date, counted only where BOTH dates have radar, so the two
    numbers compare the same ground."""
    both = (pre > 0) & (post > 0)
    row_km2 = cell_km2(bounds, pre.shape)[:, None]
    out = {}
    for k, name in GROUND_NAMES.items():
        before = float(np.sum(row_km2 * ((pre == k) & both)))
        after = float(np.sum(row_km2 * ((post == k) & both)))
        out[name] = {"before_km2": round(before, 2), "after_km2": round(after, 2)}
    return out


def add_ground(folder: Path) -> dict:
    """Write before_ground.png / after_ground.png for one processed event and record the layer
    and its km² in event.json. Returns the per-class stats."""
    event = json.loads((folder / "event.json").read_text())
    water_db = event["stats"].get("water_line_db") or DEFAULT_WATER_DB
    pre, post = ground_pair(
        np.load(folder / "pre_db.npy").astype("float32"),
        np.load(folder / "post_db.npy").astype("float32"),
        water_db,
        np.load(folder / "flags.npy"),
    )
    save_ground(pre, folder / "before_ground.png")
    save_ground(post, folder / "after_ground.png")
    stats = ground_stats(pre, post, event["bounds"])
    updated = {
        **event,
        "stats": {**event["stats"], "ground": stats},
        "layers": {
            **event["layers"],
            "before_ground": "before_ground.png",
            "after_ground": "after_ground.png",
        },
    }
    (folder / "event.json").write_text(json.dumps(updated, indent=2))
    return stats
