"""Check every processed event against what the map shows.

    uv run python -m app.modules.events.verify          # all events, exit code 1 on any failure

Works on the display-grid files the browser loads (flags.npy = the change colours, areas.npy = the
area raster), not on the pipeline's own numbers, so it catches drawing and counting bugs:

  1. Each marker sits on a coloured change pixel, inside its own area and its own outline.
  2. Each outline drawn on the map (the rings) covers the same pixels as its area.
  3. Each area's km² matches the changed pixels inside it.
  4. Nothing is left out: the share-of-land-changed density, recomputed here on the display grid,
     finds no concentration of change outside the numbered areas that holds the listing minimum.
  5. The list is ordered, shares add up to the stated coverage, each breakdown adds up, and the
     class table adds up to the totals.
"""

import json
import math
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from app.core.config import settings
from app.modules.events.pipeline import ROUGH, WATER_DB, _box_sum, _odd

LAND = (1, 2, 3)
# The independent recomputation smooths per pixel, the pipeline per 50 m block, so their edges
# differ slightly; a cluster only counts as missed when clearly above the threshold.
DENSITY_MARGIN = 1.15


def _px_km2(bounds, shape) -> float:
    (s, w), (n, e) = bounds
    rows, cols = shape
    lat = math.radians((s + n) / 2)
    return ((e - w) * 111.32 * math.cos(lat) / cols) * ((n - s) * 110.57 / rows)


def _rc(bounds, shape, lat, lon):
    (s, w), (n, e) = bounds
    rows, cols = shape
    return int((n - lat) / (n - s) * rows), int((lon - w) / (e - w) * cols)


def _rasterize(rings, bounds, shape) -> np.ndarray:
    """The outline as drawn (even-odd fill of its rings) on the display grid."""
    (s, w), (n, e) = bounds
    rows, cols = shape
    out = np.zeros(shape, bool)
    for ring in rings:
        img = Image.new("1", (cols, rows), 0)
        pts = [((lon - w) / (e - w) * cols, (n - lat) / (n - s) * rows) for lat, lon in ring]
        ImageDraw.Draw(img).polygon(pts, fill=1)
        out ^= np.array(img, bool)
    return out


def check(event_dir) -> list[str]:
    e = json.loads((event_dir / "event.json").read_text())
    flags = np.load(event_dir / "flags.npy")
    areas = np.load(event_dir / "areas.npy")
    pre = np.load(event_dir / "pre_db.npy").astype("float32")
    post = np.load(event_dir / "post_db.npy").astype("float32")
    s, rule, regions = e["stats"], e.get("area_rule") or {}, e["regions"]
    px = _px_km2(e["bounds"], flags.shape)
    land = np.isin(flags, LAND)
    with np.errstate(invalid="ignore"):
        wl = s.get("water_line_db") or WATER_DB
        ground = (np.isfinite(pre) & np.isfinite(post) & (np.fmax(pre, post) > wl)) | land
    fails: list[str] = []

    def ok(cond, msg):
        if not cond:
            fails.append(msg)

    drawn = {r["n"]: _rasterize(r["rings"], e["bounds"], flags.shape) for r in regions}

    # 1. markers on colour, in their own area and outline (1-px slack for lat/lon rounding)
    for r in regions:
        rr, cc = _rc(e["bounds"], flags.shape, r["lat"], r["lon"])
        win = (slice(max(0, rr - 1), rr + 2), slice(max(0, cc - 1), cc + 2))
        ok(land[win].any(), f"area {r['n']}: marker not on a coloured change pixel")
        ok((areas[win] == r["n"]).any(), f"area {r['n']}: marker outside its own area")
        ok(drawn[r["n"]][win].any(), f"area {r['n']}: marker outside its own outline")
        (bs, bw), (bn, be) = r["box"]
        ok(bs <= r["lat"] <= bn and bw <= r["lon"] <= be, f"area {r['n']}: marker outside its box")

    # 2. outline drawn = area counted (overlap of the two pixel sets)
    worst_iou = 1.0
    for r in regions:
        a, d = areas == r["n"], drawn[r["n"]]
        iou = (a & d).sum() / max(1, (a | d).sum())
        worst_iou = min(worst_iou, iou)
        ok(iou >= 0.9, f"area {r['n']}: outline and area differ (overlap {iou:.2f})")

    # 3. km² = changed pixels inside the area (display grid; resampling tolerance)
    for r in regions:
        got = int((land & (areas == r["n"])).sum()) * px
        tol = max(0.06, 0.08 * r["km2"])
        ok(abs(got - r["km2"]) <= tol, f"area {r['n']}: {r['km2']} km² listed, {got:.2f} drawn")

    # 4. nothing left out
    largest_missed = 0.0
    if rule.get("threshold_density_pct") is not None:
        k = _odd(rule["window_km"] * 1000 / math.sqrt(px * 1e6))
        dens = _box_sum(land * 1.0, k) / np.maximum(1.0, _box_sum(ground * 1.0, k))
        dense = (dens >= DENSITY_MARGIN * rule["threshold_density_pct"] / 100) & ground
        missed, n = ndimage.label(dense & (areas == 0))
        if n:
            km = ndimage.sum(land, missed, index=np.arange(1, n + 1)) * px
            largest_missed = float(km.max())
        ok(
            largest_missed < rule["min_km2"],
            f"{largest_missed:.2f} km² of concentrated change has no number "
            f"(listing minimum {rule['min_km2']} km²)",
        )

    # 5. order, shares, breakdowns, class table
    km = [r["km2"] for r in regions]
    ok(km == sorted(km, reverse=True), "areas not ordered by changed land")
    if regions:
        ok(
            abs(sum(r["share_pct"] for r in regions) - rule.get("coverage_pct", 0)) <= 0.5,
            "shares do not add up to the stated coverage",
        )
    for r in regions:
        ok(
            abs(sum((r.get("breakdown") or {}).values()) - r["km2"]) <= 0.03,
            f"area {r['n']}: breakdown does not add up",
        )
        ok(
            r["density_pct"] <= 100 and r["km2"] <= r["extent_km2"] + 0.05,
            f"area {r['n']}: odd size",
        )
    if e.get("cross_pol"):
        parts = [
            s.get(k) or 0
            for k in (
                "increase_double_bounce_km2",
                "increase_volume_km2",
                "increase_rough_water_km2",
            )
        ]
        ok(abs(sum(parts) - s["increase_km2"]) <= 0.05, "brighter classes do not add up")
    ok(s["open_water_km2"] <= s["decrease_km2"] + 0.01, "open water larger than darker")
    ok(not np.isin(flags, (ROUGH,)).any() or e.get("cross_pol"), "wind on water without HV")

    print(
        f"{e['id']:16s} {len(regions):2d} areas, {rule.get('coverage_pct')}% of "
        f"{rule.get('total_km2')} km² changed land; outline/area overlap ≥ {worst_iou:.2f}; "
        f"largest unnumbered concentration {largest_missed:.2f} km² "
        f"(minimum {rule.get('min_km2')}) -> {'PASS' if not fails else 'FAIL'}"
    )
    for f in fails:
        print("   -", f)
    return fails


def main() -> None:
    failed = 0
    for d in sorted(settings.events_dir.iterdir()):
        if (d / "event.json").is_file() and (d / "areas.npy").is_file():
            failed += bool(check(d))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
