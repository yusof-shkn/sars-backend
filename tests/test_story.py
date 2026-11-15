import json

import numpy as np
import pytest

from app.core.errors import AppError
from app.modules.events import pipeline, service
from app.modules.events.pipeline import Grid, synthetic_scene


@pytest.fixture
def scene(tmp_path, monkeypatch):
    """A synthetic flood event on disk; returns (folder, lat/lon inside the flood, far from it)."""
    monkeypatch.setattr(pipeline.settings, "data_dir", tmp_path)
    monkeypatch.setattr(service.settings, "data_dir", tmp_path)
    pre, post, flood, _trees = synthetic_scene()
    grid = Grid(epsg=32646, x0=760_000.0, y0=2_640_000.0, dx=20.0, dy=-20.0)
    meta = dict(id="t", title="t", hazard="flood", location="", description="", synthetic=True,
                pre=dict(id="A", datetime="2026-07-01T00:00:00Z", polarization="HH"),
                post=dict(id="B", datetime="2026-07-13T00:00:00Z", polarization="HH"))  # fmt: skip
    folder = pipeline.write_event(meta, pre, post, grid, 3.0)
    (south, west), (north, east) = service.get_event("t").bounds
    rows, cols = pre.shape

    def ll(r, c):
        return (north - (r + 0.5) / rows * (north - south), west + (c + 0.5) / cols * (east - west))

    inside = np.argwhere(flood)
    r, c = inside[len(inside) // 2]
    # Quiet land well inside the scene: 1 km from the flood band and from the flooded trees.
    far = (350, 480)
    return folder, ll(r, c), ll(*far), (north + 1, west)


def write_story(folder, points, **extra):
    story = {"summary": "s", "points": points, **extra}
    (folder / "story.json").write_text(json.dumps(story))


def point(lat, lon, **extra):
    return {"id": "p", "title": "P", "lat": lat, "lon": lon, "what_happened": "w", **extra}


def test_event_without_story_loads_with_story_none(scene):
    assert service.get_event("t").story is None


def test_story_is_attached_and_photo_urls_point_at_files(scene):
    folder, flood, _far, _out = scene
    (folder / "photos").mkdir()
    (folder / "photos" / "a.jpg").write_bytes(b"x")
    photo = dict(file="photos/a.jpg", caption="c", credit="NASA", licence="Public domain",
                 source_url="https://example.org/a")  # fmt: skip
    write_story(folder, [point(*flood, photos=[photo], sources=[{"label": "L", "url": "https://e.org"}])],
                path={"label": "path", "points": [list(flood), list(flood)]})  # fmt: skip
    story = service.get_event("t").story
    assert story.summary == "s" and story.path.label == "path"
    assert story.points[0].photos[0].file == "/files/events/t/photos/a.jpg"
    assert story.points[0].sources[0].label == "L"


def test_point_radar_matches_probe(scene):
    folder, flood, _far, _out = scene
    write_story(folder, [point(*flood)])
    radar = service.get_event("t").story.points[0].radar
    probe = service.probe("t", *flood)
    assert radar.covered is True
    assert (radar.pre_db, radar.post_db, radar.delta_db, radar.flag) == (
        probe.pre_db,
        probe.post_db,
        probe.delta_db,
        probe.flag,
    )


def test_changed_pct_300m_is_high_inside_the_flood_and_zero_far_from_it(scene):
    folder, flood, far, _out = scene
    write_story(folder, [point(*flood), point(*far) | {"id": "q"}])
    a, b = (p.radar for p in service.get_event("t").story.points)
    assert a.changed_pct_300m > 50
    assert b.changed_pct_300m == 0


def test_point_outside_the_box_is_not_covered(scene):
    folder, _flood, _far, out = scene
    write_story(folder, [point(*out)])
    radar = service.get_event("t").story.points[0].radar
    assert radar.covered is False
    assert radar.pre_db is None and radar.flag is None and radar.changed_pct_300m is None


def test_malformed_story_raises_with_the_file_name(scene):
    folder, flood, _far, _out = scene
    (folder / "story.json").write_text(json.dumps({"points": [{"lat": 1}]}))
    with pytest.raises(AppError, match="story.json"):
        service.get_event("t")


def test_missing_photo_file_raises_with_the_file_name(scene):
    folder, flood, _far, _out = scene
    photo = dict(file="photos/gone.jpg", caption="c", credit="x", licence="y",
                 source_url="https://example.org")  # fmt: skip
    write_story(folder, [point(*flood, photos=[photo])])
    with pytest.raises(AppError, match="gone.jpg"):
        service.get_event("t")


def test_a_broken_story_does_not_break_the_event_list(scene):
    folder, _flood, _far, _out = scene
    (folder / "story.json").write_text("{not json")
    assert [e.id for e in service.list_events()] == ["t"]


def test_step_display_settings_pass_through(scene):
    folder, flood, _far, _out = scene
    show = {"view": "change", "subtle": True, "basemap": "hls", "radar_opacity": 0, "split": 0.5}
    video = {"title": "Watch", "url": "https://youtu.be/abc"}
    write_story(folder, [point(*flood, headline="Big", area=1, show=show, videos=[video])])
    p = service.get_event("t").story.points[0]
    assert p.videos[0].url == "https://youtu.be/abc"
    assert (p.headline, p.area, p.show.view, p.show.subtle, p.show.split) == (
        "Big", 1, "change", True, 0.5
    )  # fmt: skip
    assert p.show.radar_opacity == 0 and p.show.basemap == "hls"


def test_layer_urls_change_when_the_file_is_rewritten(scene):
    # A browser keeps an image it has already loaded; a reprocessed event must not show the old one.
    import os

    folder, _flood, _far, _out = scene
    before = service.get_event("t").layers["change"]
    assert before.startswith("/files/events/t/change.png?v=")
    stat = (folder / "change.png").stat()
    os.utime(folder / "change.png", (stat.st_atime, stat.st_mtime + 60))
    assert service.get_event("t").layers["change"] != before
