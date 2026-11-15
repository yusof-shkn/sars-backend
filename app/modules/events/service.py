import json
import math

import numpy as np

from app.core.config import settings
from app.core.errors import AppError
from app.modules.events.schemas import Event, EventSummary, Probe
from app.modules.events.story import load_story

# Events are read from disk on every request: a handful of JSON files, no DB needed.
# Add caching/DB only if events reach the hundreds.


def _versioned(folder, event_id: str, file: str) -> str:
    """URL of a layer file with its modification time as a version, so a browser that already
    holds the image fetches the new one after the event is reprocessed."""
    target = folder / file
    version = int(target.stat().st_mtime) if target.is_file() else 0
    return f"/files/events/{event_id}/{file}?v={version}"


def _load(path, with_story: bool = False) -> Event:
    event = Event.model_validate(json.loads(path.read_text()))
    event.layers = {
        name: _versioned(path.parent, event.id, file) for name, file in event.layers.items()
    }
    # The list needs no stories, and one broken story must not take the whole list down.
    if with_story:
        event.story = load_story(path.parent, event)
    return event


def list_events() -> list[EventSummary]:
    events = [_load(p) for p in sorted(settings.events_dir.glob("*/event.json"))]
    # Newest first, then (stable sort) real NISAR events ahead of the synthetic demo.
    events.sort(key=lambda e: e.post.datetime, reverse=True)
    events.sort(key=lambda e: e.synthetic)
    return [
        EventSummary(
            id=e.id,
            title=e.title,
            hazard=e.hazard,
            synthetic=e.synthetic,
            pre_date=e.pre.datetime[:10],
            post_date=e.post.datetime[:10],
            changed_pct=e.stats.changed_pct,
        )
        for e in events
    ]


def get_event(event_id: str) -> Event:
    path = settings.events_dir / event_id / "event.json"
    if not event_id.replace("-", "").replace("_", "").isalnum() or not path.is_file():
        raise AppError(404, "Event not found")
    return _load(path, with_story=True)


def probe(event_id: str, lat: float, lon: float) -> Probe:
    """Before / after / change at one point, from the per-pixel arrays the pipeline saved."""
    event = get_event(event_id)
    folder = settings.events_dir / event_id
    if not (folder / "pre_db.npy").is_file():
        raise AppError(404, "No per-pixel data for this event; re-run the pipeline")
    (south, west), (north, east) = event.bounds
    pre = np.load(folder / "pre_db.npy", mmap_mode="r")
    rows, cols = pre.shape
    # floor, not int(): int() rounds toward zero, so a point just north or west of the box
    # would read the edge pixel instead of being refused.
    r = math.floor((north - lat) / (north - south) * rows)
    c = math.floor((lon - west) / (east - west) * cols)
    if not (0 <= r < rows and 0 <= c < cols):
        raise AppError(404, "Outside the analysed area")
    post = np.load(folder / "post_db.npy", mmap_mode="r")
    flags = np.load(folder / "flags.npy", mmap_mode="r")
    a, b = float(pre[r, c]), float(post[r, c])
    if not (np.isfinite(a) and np.isfinite(b)):
        return Probe(lat=lat, lon=lon, pre_db=None, post_db=None, delta_db=None, flag=None)
    flag = {1: "decrease", 2: "increase", 3: "volume", 4: "rough_water"}.get(int(flags[r, c]))
    hv: dict = {}
    if (folder / "hv_pre_db.npy").is_file():
        ha = float(np.load(folder / "hv_pre_db.npy", mmap_mode="r")[r, c])
        hb = float(np.load(folder / "hv_post_db.npy", mmap_mode="r")[r, c])
        if np.isfinite(ha) and np.isfinite(hb):
            hv = {"hv_pre_db": round(ha, 1), "hv_post_db": round(hb, 1)}
    return Probe(
        lat=lat,
        lon=lon,
        pre_db=round(a, 1),
        post_db=round(b, 1),
        delta_db=round(b - a, 1),
        flag=flag,
        **hv,
    )
