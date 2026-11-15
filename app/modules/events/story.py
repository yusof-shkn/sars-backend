"""The story of an event: a few real places in order, each with what happened there, photos and
the radar reading at that spot.

Hand-written in `data/events/<id>/story.json`, so reprocessing the radar never erases it. The
radar reading of each point is computed here from the event's own arrays, never typed in, so a
story cannot claim a signal the data does not hold.
"""

import json
import math
from pathlib import Path

import numpy as np
from pydantic import ValidationError

from app.core.errors import AppError
from app.modules.events.schemas import Event, PointRadar, Story

# Neighbourhood of a story point: the share of it that changed says more than one pixel does.
RADIUS_M = 300
EARTH_M_PER_DEG = 111_320.0
FLAG_NAMES = {1: "decrease", 2: "increase", 3: "volume", 4: "rough_water"}
# Classes that count as change of the ground (wind on water does not).
CHANGED_FLAGS = (1, 2, 3)
NOT_COVERED = PointRadar(covered=False)


def point_radar(folder: Path, bounds, lat: float, lon: float) -> PointRadar:
    """Before / after / class at the point, and the share of ground within RADIUS_M that changed.
    `covered` is False outside the analysed box or where the radar has no sample on both dates."""
    if not (folder / "pre_db.npy").is_file():
        return NOT_COVERED
    (south, west), (north, east) = bounds
    pre = np.load(folder / "pre_db.npy", mmap_mode="r")
    rows, cols = pre.shape
    r = math.floor((north - lat) / (north - south) * rows)
    c = math.floor((lon - west) / (east - west) * cols)
    if not (0 <= r < rows and 0 <= c < cols):
        return NOT_COVERED
    post = np.load(folder / "post_db.npy", mmap_mode="r")
    flags = np.load(folder / "flags.npy", mmap_mode="r")
    a, b = float(pre[r, c]), float(post[r, c])
    if not (np.isfinite(a) and np.isfinite(b)):
        return NOT_COVERED
    # Window of the display grid (regular lon/lat) around the point, then a circle inside it.
    row_m = (north - south) / rows * EARTH_M_PER_DEG
    col_m = (east - west) / cols * EARTH_M_PER_DEG * math.cos(math.radians(lat))
    dr, dc = math.ceil(RADIUS_M / row_m), math.ceil(RADIUS_M / col_m)
    r0, r1, c0, c1 = max(0, r - dr), min(rows, r + dr + 1), max(0, c - dc), min(cols, c + dc + 1)
    yy, xx = np.mgrid[r0:r1, c0:c1]
    circle = ((yy - r) * row_m) ** 2 + ((xx - c) * col_m) ** 2 <= RADIUS_M**2
    seen = circle & np.isfinite(pre[r0:r1, c0:c1]) & np.isfinite(post[r0:r1, c0:c1])
    changed = seen & np.isin(flags[r0:r1, c0:c1], CHANGED_FLAGS)
    return PointRadar(
        covered=True,
        pre_db=round(a, 1),
        post_db=round(b, 1),
        delta_db=round(b - a, 1),
        flag=FLAG_NAMES.get(int(flags[r, c])),
        changed_pct_300m=round(100 * float(changed.sum()) / max(1, int(seen.sum())), 1),
    )


def load_story(folder: Path, event: Event) -> Story | None:
    """The event's story with photo URLs and radar readings filled in; None when it has none.
    A story that cannot be read is an error naming the file, never a silently dropped story."""
    path = folder / "story.json"
    if not path.is_file():
        return None
    try:
        story = Story.model_validate(json.loads(path.read_text()))
    except (json.JSONDecodeError, ValidationError) as e:
        raise AppError(500, f"{event.id}/story.json is not a valid story: {e}") from e
    points = []
    for p in story.points:
        photos = []
        for photo in p.photos:
            if not (folder / photo.file).is_file():
                raise AppError(500, f"{event.id}/story.json names a missing photo: {photo.file}")
            photos.append(
                photo.model_copy(update={"file": f"/files/events/{event.id}/{photo.file}"})
            )
        radar = point_radar(folder, event.bounds, p.lat, p.lon)
        points.append(p.model_copy(update={"photos": photos, "radar": radar}))
    return story.model_copy(update={"points": points})
