import numpy as np
import pytest
from pyproj import Transformer
from scipy import ndimage

from app.modules.events import pipeline
from app.modules.events.pipeline import (
    Grid,
    _merge_light,
    box_mean,
    change_areas,
    detect_change,
    eonet_rows,
    majority,
    stack_key,
    subtle_change,
    summarize,
    synthetic_scene,
    to_latlon,
    top_regions,
)


def test_stack_key_ignores_cycle_and_time():
    a = "NISAR_L2_PR_GCOV_033_137_D_073_2005_DHDH_A_20261021T170721_20261021T170733_P05036_N_P_J"
    b = a.replace("_033_", "_034_").replace("20261021", "20261102")
    assert stack_key(a) == stack_key(b) == "137_D_073_2005_DHDH_A"


def test_box_mean_matches_naive_and_skips_nan():
    rng = np.random.default_rng(0)
    a = rng.random((9, 11)) + 0.1
    a[4, 5] = np.nan
    got = box_mean(a, 3)
    i, j = 4, 6  # window includes the NaN
    win = a[i - 1 : i + 2, j - 1 : j + 2]
    assert np.isclose(got[i, j], np.nanmean(win))
    assert np.isclose(got[0, 0], a[:2, :2].mean())  # edge window is truncated, not padded


def test_majority_drops_specks_and_keeps_blocks():
    m = np.zeros((20, 20), bool)
    m[3, 3] = True  # lone speck
    m[8:16, 8:16] = True  # solid block
    out = majority(m, 5)
    assert not out[3, 3]
    assert out[10:14, 10:14].all()  # interior kept
    assert not out[8, 8] and out[8, 11]  # convex corners erode by ~1 px, edges stay
    assert not out[0, 0]  # truncated edge window still works


def test_detect_change_flags_drop_and_rise():
    pre = np.full((40, 40), 0.1)
    post = pre.copy()
    post[:, :20] = 0.01  # -10 dB
    post[:, 30:] = 1.0  # +10 dB
    _, _, d, dec, inc, dark = detect_change(pre, post, 3.0)
    assert dec[:, :18].all() and not dec[:, 22:].any()
    assert inc[:, 32:].all() and not inc[:, :28].any()
    assert not dark.any()
    s = summarize(d, dec, inc, dark, pixel_km2=0.0004)
    assert s["aoi_km2"] == round(1600 * 0.0004, 2)
    # 20 dropped cols + 10 raised cols, widened by 2 px where the 5x5 filter blurs the +10 dB edge
    assert s["changed_pct"] == 80.0


def test_both_dates_at_noise_floor_are_masked_not_flagged():
    rng = np.random.default_rng(1)
    pre = 0.002 * rng.gamma(4, 1 / 4, (60, 60))  # -27 dB: calm water / shadow
    post = 0.002 * rng.gamma(4, 1 / 4, (60, 60))
    _, _, d, dec, inc, dark = detect_change(pre, post, 3.0)
    assert dark.all() and np.isnan(d).all()
    assert not dec.any() and not inc.any()


def test_synthetic_flood_is_recovered():
    """The demo scene is also the method's ground-truth check."""
    pre, post, flood, trees = synthetic_scene()
    _, _, _, dec, inc, dark = detect_change(pre, post, 3.0)
    assert dec[flood].mean() > 0.97  # recall on new open water
    assert inc[trees].mean() > 0.90  # recall on flooded vegetation
    assert (dec | inc)[~flood & ~trees].mean() < 0.01  # false positives (edge blur only)
    assert dark.sum() > 0  # the permanent river is masked, not counted as change


def test_to_latlon_puts_pixels_where_they_belong():
    grid = Grid(epsg=32646, x0=760_000.0, y0=2_640_000.0, dx=20.0, dy=-20.0)
    rows, cols = 120, 150
    a = np.zeros((rows, cols))
    a[30, 100] = 1.0  # one bright pixel
    (out,), (sw, ne) = to_latlon([a], grid, a.shape)
    r, c = np.unravel_index(np.nanargmax(out), out.shape)
    # Centre of that output pixel, back to UTM, must land within one source pixel.
    lon = sw[1] + (c + 0.5) * (ne[1] - sw[1]) / cols
    lat = ne[0] - (r + 0.5) * (ne[0] - sw[0]) / rows
    x, y = Transformer.from_crs(4326, grid.epsg, always_xy=True).transform(lon, lat)
    assert abs(x - (grid.x0 + 100 * grid.dx)) < 20 and abs(y - (grid.y0 + 30 * grid.dy)) < 20
    # Corners outside the rotated UTM footprint are transparent (NaN), the inside is not.
    assert np.isnan(out).any() and np.isfinite(out[rows // 2, cols // 2])


def test_read_gcov_retries_transient_errors_and_caches(tmp_path, monkeypatch):
    from app.modules.events.schemas import Granule

    monkeypatch.setattr(pipeline.settings, "data_dir", tmp_path)
    monkeypatch.setattr(pipeline.time, "sleep", lambda _s: None)
    calls: list[str] = []

    def flaky(src, name, bbox):
        calls.append(src)
        if len(calls) == 1:
            raise ConnectionResetError("wifi blip")
        grid = Grid(32646, 1.0, 2.0, 20.0, -20.0)
        return (
            np.ones((4, 5), "float32"),
            grid,
            Granule(id=name, datetime="2026-07-01", polarization="HH"),
            None,  # no cross-pol term
        )

    monkeypatch.setattr(pipeline, "_read_gcov", flaky)
    a = pipeline.read_gcov("https://x/y/G1.h5", [1, 2, 3, 4])
    b = pipeline.read_gcov("https://x/y/G1.h5", [1, 2, 3, 4])  # served from the cache
    assert calls == ["https://x/y/G1.h5"] * 2
    assert np.array_equal(a[0], b[0]) and a[1] == b[1] and a[2] == b[2]
    assert b[2].id == "G1"

    def broken(src, name, bbox):
        raise KeyError("science/LSAR/GCOV")  # a wrong path must not be retried

    monkeypatch.setattr(pipeline, "_read_gcov", broken)
    with pytest.raises(KeyError):
        pipeline.read_gcov("https://x/y/G2.h5", [1, 2, 3, 4])


def test_probe_reads_the_flooded_pixel(tmp_path, monkeypatch):
    from app.core.errors import AppError
    from app.modules.events import service

    monkeypatch.setattr(pipeline.settings, "data_dir", tmp_path)
    monkeypatch.setattr(service.settings, "data_dir", tmp_path)
    pre, post, flood, trees = synthetic_scene()
    grid = Grid(epsg=32646, x0=760_000.0, y0=2_640_000.0, dx=20.0, dy=-20.0)
    meta = dict(id="t", title="t", hazard="flood", location="", description="", synthetic=True,
                pre=dict(id="A", datetime="2026-07-01T00:00:00Z", polarization="HH"),
                post=dict(id="B", datetime="2026-07-13T00:00:00Z", polarization="HH"))  # fmt: skip
    pipeline.write_event(meta, pre, post, grid, 3.0)
    event = service.get_event("t")
    (south, west), (north, east) = event.bounds
    rows, cols = pre.shape
    r, c = np.argwhere(flood)[len(np.argwhere(flood)) // 2]  # a pixel inside the injected flood
    lat = north - (r + 0.5) / rows * (north - south)
    lon = west + (c + 0.5) / cols * (east - west)
    got = service.probe("t", lat, lon)
    assert got.flag == "decrease" and got.post_db < got.pre_db - 3
    assert (
        event.stats.open_water_km2 > 0.5 * event.stats.decrease_km2
    )  # the demo flood IS open water
    with pytest.raises(AppError):
        service.probe("t", north + 1, lon)


def test_dual_pol_splits_brighter_by_mechanism(tmp_path, monkeypatch):
    from app.modules.events import service

    monkeypatch.setattr(pipeline.settings, "data_dir", tmp_path)
    monkeypatch.setattr(service.settings, "data_dir", tmp_path)
    pre, post, flood, trees = synthetic_scene()
    pre, post = pre.copy(), post.copy()
    growth = np.zeros(pre.shape, bool)
    growth[300:340, 50:90] = True  # a field where HH and HV both rise x5 -> volume
    rough = np.zeros(pre.shape, bool)
    rough[370:395, 100:400] = True  # calm water that gets wind-roughened: -25 -> -18 dB, HV flat
    pre[rough], post[rough] = 0.003, 0.015
    post[growth] *= 5
    hv_pre, hv_post = pre * 0.1, post * 0.1  # cross-pol 10 dB below co-pol
    hv_post[trees] = hv_pre[trees]  # trees: HH x5 but HV flat -> double bounce
    hv_post[rough] = hv_pre[rough]
    grid = Grid(epsg=32646, x0=760_000.0, y0=2_640_000.0, dx=20.0, dy=-20.0)
    meta = dict(id="dp", title="t", hazard="flood", location="", description="", synthetic=True,
                pre=dict(id="A", datetime="2026-07-01T00:00:00Z", polarization="HH"),
                post=dict(id="B", datetime="2026-07-13T00:00:00Z", polarization="HH"))  # fmt: skip
    pipeline.write_event(meta, pre, post, grid, 3.0, hv=(hv_pre, hv_post))
    e = service.get_event("dp")
    s = e.stats
    assert e.cross_pol
    assert abs(s.increase_double_bounce_km2 - trees.sum() * 4e-4) < 0.15
    assert abs(s.increase_volume_km2 - growth.sum() * 4e-4) < 0.15
    assert abs(s.increase_rough_water_km2 - rough.sum() * 4e-4) < 0.6  # sieve erodes edges
    (south, west), (north, east) = e.bounds
    rows, cols = pre.shape

    def at(r, c):
        lat = north - (r + 0.5) / rows * (north - south)
        lon = west + (c + 0.5) / cols * (east - west)
        return service.probe("dp", lat, lon)

    assert at(320, 70).flag == "volume" and at(320, 70).hv_post_db > at(320, 70).hv_pre_db + 3
    assert at(80, 70).flag == "increase"  # the trees: double bounce
    assert at(382, 250).flag == "rough_water"
    assert "before_rgb" in e.layers and (tmp_path / "events/dp/after_rgb.png").is_file()


def test_top_regions_finds_the_flood_first():
    pre, post, flood, trees = synthetic_scene()
    _, _, _, dec, inc, _ = detect_change(pre, post, 3.0)
    kind = np.zeros(pre.shape, "uint8")
    kind[dec], kind[inc] = 1, 2
    grid = Grid(epsg=32646, x0=760_000.0, y0=2_640_000.0, dx=20.0, dy=-20.0)
    regions = top_regions(kind, grid, n=3)
    # The permanent river (masked) splits the injected flood into a north and a south blob.
    assert [r["cls"] for r in regions[:2]] == ["decrease", "decrease"] and regions[0]["n"] == 1
    assert abs(regions[0]["km2"] + regions[1]["km2"] - flood.sum() * 4e-4) < 0.6
    (south, west), (north, east) = (23.776, 95.551), (23.851, 95.655)  # the demo scene's box
    assert south < regions[0]["lat"] < north and west < regions[0]["lon"] < east
    assert any(r["cls"] == "increase" for r in regions)


def test_eonet_rows_handles_points_and_polygons():
    point = {"date": "2026-08-01T00:00:00Z", "type": "Point", "coordinates": [90.5, 23.2]}
    ring = [[10.0, 40.0], [10.5, 40.0], [10.5, 40.4], [10.0, 40.4], [10.0, 40.0]]
    poly = {"date": "2026-09-01T00:00:00Z", "type": "Polygon", "coordinates": [ring]}
    data = {
        "events": [
            {"title": "Flood A", "categories": [{"title": "Floods"}], "geometry": [point]},
            {"title": "Fire B", "categories": [{"title": "Wildfires"}], "geometry": [poly]},
            {"title": "No geometry", "categories": [], "geometry": []},
        ]
    }
    rows = eonet_rows(data)
    assert [r[2] for r in rows] == ["Fire B", "Flood A"]  # newest first, empty one dropped
    assert rows[1][3] == "90.35 23.05 90.65 23.35"
    assert rows[0][3] == "9.85 39.85 10.65 40.55"


def test_hls_dates_picks_the_nearest_real_tile(monkeypatch):
    real = {("HLS_S30_Nadir_BRDF_Adjusted_Reflectance", "2026-08-18"): 130_000,
            ("HLS_L30_Nadir_BRDF_Adjusted_Reflectance", "2026-07-04"): 90_000}  # fmt: skip

    def fake(url):
        layer, d = url.split("/best/")[1].split("/default/")
        return real.get((layer, d[:10]), 334)

    monkeypatch.setattr(pipeline, "_tile_bytes", fake)
    got = pipeline.hls_dates(
        23.3, 90.7, ["2026-07-02T12:34:57Z", "2026-08-19T12:34:54Z", "2026-01-01"]
    )
    assert got[0] == {"date": "2026-07-04", "layer": "HLS_L30_Nadir_BRDF_Adjusted_Reflectance"}
    assert got[1] == {"date": "2026-08-18", "layer": "HLS_S30_Nadir_BRDF_Adjusted_Reflectance"}
    assert got[2] is None
    assert pipeline._tile_xy(90.7, 23.3) == (3079, 1775)


def test_region_polarimetry_flags_an_equal_shift_artefact():
    kind = np.zeros((300, 300), "uint8")
    kind[20:140, 20:140] = 1  # 120x120 px at 20 m = 5.76 km2: big enough to matter
    kind[200:230, 200:230] = 1  # small blob
    d = np.full(kind.shape, np.nan)
    d_hv = np.full(kind.shape, np.nan)
    d[kind == 1] = -5.0
    d_hv[20:140, 20:140] = -5.1  # same shift in both polarisations: not how ground behaves
    d_hv[200:230, 200:230] = -9.0  # HV dropped far more than HH: vegetation really went
    grid = Grid(epsg=32646, x0=760_000.0, y0=2_640_000.0, dx=20.0, dy=-20.0)
    big, small = top_regions(kind, grid, n=2, d=d, d_hv=d_hv)
    assert big["km2"] == 5.76 and big["d_hh"] == -5.0 and big["d_hv"] == -5.1 and big["suspect"]
    assert small["d_hv"] == -9.0 and not small["suspect"]
    assert "d_hv" not in top_regions(kind, grid, n=1, d=d)[0]  # single-pol mode: no HV fields


def test_subtle_change_recovers_a_one_db_offset_the_pixel_test_misses():
    rng = np.random.default_rng(3)
    land = 0.06 * np.ones((600, 600))
    burn = np.zeros(land.shape, bool)
    burn[150:450, 150:450] = True  # 6 km square brightened by only +1 dB
    pre = land * rng.gamma(4, 1 / 4, land.shape)
    post = land * np.where(burn, 10**0.1, 1.0) * rng.gamma(4, 1 / 4, land.shape)
    pre_db, post_db, d, _, _, _ = detect_change(pre, post, 3.0)
    assert (d <= -3).mean() + (d >= 3).mean() < 0.02  # pixel test: nothing
    grid = Grid(epsg=32646, x0=760_000.0, y0=2_640_000.0, dx=20.0, dy=-20.0)
    blk, thr = subtle_change(d, pre_db, post_db, grid)
    assert 0.3 <= thr < 1.0
    inner = np.zeros_like(burn)
    inner[180:420, 180:420] = True  # away from the block edges
    assert (blk[inner] > thr).mean() > 0.95 and (blk[~burn] > thr).mean() < 0.05


def _scatter(rng, shape, density):
    """Field-sized specks (3x3 px) at a given ground density, like scattered flooded fields."""
    seeds = rng.random(shape) < density / 9
    return ndimage.binary_dilation(seeds, np.ones((3, 3), bool))


GRID20 = Grid(epsg=32646, x0=760_000.0, y0=2_640_000.0, dx=20.0, dy=-20.0)


def test_change_areas_cover_every_cluster_and_count_exactly():
    rng = np.random.default_rng(11)
    kind = np.zeros((600, 600), "uint8")
    ground = np.ones(kind.shape, bool)
    kind[_scatter(rng, kind.shape, 0.004)] = 1  # thin background change everywhere
    a = np.zeros(kind.shape, bool)
    a[50:200, 50:250] = True  # dense cluster A: flooded fields
    b = np.zeros(kind.shape, bool)
    b[380:500, 350:520] = True  # dense cluster B: growth
    kind[a & _scatter(rng, kind.shape, 0.35)] = 1
    kind[b & _scatter(rng, kind.shape, 0.30)] = 3
    kind[300:330, :] = 4  # a lake rippled by wind: never an area
    ground[300:330, :] = False
    areas, labels, rule = change_areas(kind, ground, GRID20)
    land = (kind > 0) & (kind != 4)
    assert [x["n"] for x in areas] == list(range(1, len(areas) + 1))
    assert [x["km2"] for x in areas] == sorted((x["km2"] for x in areas), reverse=True)
    # Both clusters are inside numbered areas, and no area spans both.
    assert (labels[a] > 0).mean() > 0.95 and (labels[b] > 0).mean() > 0.95
    assert not set(np.unique(labels[a])) - {0} & set(np.unique(labels[b])) - {0}
    assert not (labels[300:330, :] > 0).any()  # the lake is not an area
    tr = Transformer.from_crs(4326, GRID20.epsg, always_xy=True)
    for x in areas:
        # counted = changed land inside the area, exactly; share and density agree with it
        n_px = int((land & (labels == x["n"])).sum())
        assert x["km2"] == round(n_px * 4e-4, 2)
        assert abs(x["share_pct"] - 100 * n_px / land.sum()) < 0.06
        # marker on changed land inside its own area, inside its box
        mx, my = tr.transform(x["lon"], x["lat"])
        r, c = int(round((my - GRID20.y0) / GRID20.dy)), int(round((mx - GRID20.x0) / GRID20.dx))
        assert land[r, c] and labels[r, c] == x["n"]
        (s, w), (n, e) = x["box"]
        assert s <= x["lat"] <= n and w <= x["lon"] <= e
        assert x["rings"] and all(len(ring) >= 4 for ring in x["rings"])
    covered = int((land & (labels > 0)).sum())
    assert abs(rule["coverage_pct"] - 100 * covered / land.sum()) < 0.06


def test_change_areas_none_when_change_is_tiny_or_only_wind_on_water():
    kind = np.zeros((300, 300), "uint8")
    ground = np.ones(kind.shape, bool)
    kind[100:104, 100:104] = 1  # 16 px = 0.0064 km²: a speck, not an area
    kind[200:260, :] = 4  # wind on water only
    areas, labels, rule = change_areas(kind, ground, GRID20)
    assert areas == [] and not labels.any()


def test_merge_light_leaves_no_light_piece_touching_another():
    seg = np.array(
        [
            [1, 1, 2, 2, 0, 5],
            [1, 1, 2, 3, 0, 5],
            [1, 1, 4, 3, 0, 0],
        ]
    )
    weight = np.array(
        [
            [9.0, 9, 1, 1, 0, 1],
            [9.0, 9, 1, 1, 0, 1],
            [9.0, 9, 1, 1, 0, 0],
        ]
    )
    out = _merge_light(seg, weight, min_w=6)
    # 2, 3 and 4 (5 together) stay below 6 however they merge, so all end up in region 1;
    # 5 is light but isolated, so it stays on its own.
    assert set(np.unique(out)) == {0, 1, 5}
    assert (out[:, :4] == 1).all() and (out[:2, 5] == 5).all()
    # The invariant on the real-sized case: no light piece touches another piece.
    rng = np.random.default_rng(3)
    seg = np.kron(rng.integers(0, 40, (12, 12)), np.ones((5, 5), int))  # touching pieces
    weight = rng.random((60, 60))
    assert ((seg[:-1] != seg[1:]) & (seg[:-1] > 0) & (seg[1:] > 0)).any()
    out = _merge_light(seg, weight, min_w=8)
    w_of = ndimage.sum(weight, out, index=np.arange(out.max() + 1))
    for x, y in ((out[:-1], out[1:]), (out[:, :-1], out[:, 1:])):
        touch = (x != y) & (x > 0) & (y > 0)
        assert (w_of[x[touch]] >= 8).all() and (w_of[y[touch]] >= 8).all()


def test_tile_ranges_cover_the_scene_and_its_surroundings():
    from app.modules.events.pipeline import SCREEN_PX, _tile_xy, _world_px, tile_jobs, tile_ranges

    bounds = [[23.1, 90.55], [23.45, 90.9]]
    ranges = {z: (x0, x1, y0, y1) for z, x0, x1, y0, y1 in tile_ranges(bounds, 12)}
    assert max(ranges) == 12 and min(ranges) <= 9  # two levels out of the ~11.7 opening view
    for z, (x0, x1, y0, y1) in ranges.items():
        for lat in (23.1, 23.45):
            for lon in (90.55, 90.9):  # every scene corner is inside the downloaded range
                x, y = _tile_xy(lon, lat, z)
                assert x0 <= x <= x1 and y0 <= y <= y1
        # ... and so is a whole screen centred on the scene
        cx, cy = (_world_px(90.725, 23.275)[i] * 2**z for i in (0, 1))
        assert x0 * 256 <= cx - SCREEN_PX[0] / 2 and (x1 + 1) * 256 >= cx + SCREEN_PX[0] / 2
        assert y0 * 256 <= cy - SCREEN_PX[1] / 2 and (y1 + 1) * 256 >= cy + SCREEN_PX[1] / 2
    event = {
        "hls": [{"layer": "HLS_S30", "date": "2026-07-04"}, None],
        "pre": {"datetime": "2026-07-02T12:34:57"},
        "post": {"datetime": "2026-08-19T12:34:54"},
    }
    jobs = tile_jobs(event)
    assert [j[:2] for j in jobs] == [
        ("HLS_S30", "2026-07-04"),
        ("MODIS_Terra_CorrectedReflectance_TrueColor", "2026-07-02"),
        ("MODIS_Terra_CorrectedReflectance_TrueColor", "2026-08-19"),
    ]


def test_water_line_sits_between_the_water_peak_and_the_land_split():
    from app.modules.events.pipeline import WATER_DB, maturity, water_line

    rng = np.random.default_rng(2)
    water = rng.normal(-17.0, 1.2, 40_000)
    land = rng.normal(-6.0, 2.5, 160_000)
    line = water_line(np.concatenate([water, land]))
    assert -15.5 < line < -11.5  # below the land tail, above the water peak
    assert np.mean(water < line) > 0.97  # nearly all calm water is below it
    assert np.mean(land < line) < 0.03  # almost no land is
    assert water_line(land) == WATER_DB  # no water peak: the fixed floor
    src = (
        "https://nisar.asf.earthdatacloud.nasa.gov/NISAR/NISAR_L2_GCOV_PROVISIONAL_V1/x/"
        "NISAR_L2_PR_GCOV_028_091_D_077_4005_DHDH_A_20260819T123454_20260819T123511_P05023_N_P_J_001.h5"
    )
    assert maturity(src) == {
        "maturity": "PROVISIONAL",
        "crid": "P05023",
        "orbit_quality": "near-real-time",
        "frame_coverage": "partial",
    }
    name = "NISAR_L2_PR_GCOV_034_047_A_027_4005_DHDH_A_20261027T105124_20261027T105159_"
    assert maturity(name + "P05036_N_F_J_001.h5")["maturity"] == "VALIDATED"


def test_probe_refuses_points_just_outside_the_box(tmp_path, monkeypatch):
    import json

    from app.core.errors import AppError
    from app.modules.events import service

    monkeypatch.setattr(service.settings, "data_dir", tmp_path)
    folder = tmp_path / "events" / "box"
    folder.mkdir(parents=True)
    grid = np.full((100, 100), -10.0, "float16")
    for name in ("pre_db", "post_db"):
        np.save(folder / f"{name}.npy", grid)
    np.save(folder / "flags.npy", np.zeros((100, 100), "uint8"))
    event = {
        "id": "box", "title": "t", "hazard": "flood", "product": "p", "threshold_db": 3.0,
        "location": "", "description": "", "synthetic": False,
        "bounds": [[23.1, 90.55], [23.45, 90.9]], "stats": {
            "aoi_km2": 1, "masked_km2": 0, "decrease_km2": 0, "increase_km2": 0,
            "changed_pct": 0, "mean_change_db": 0, "noise_sigma_db": 1, "histogram": []},
        "pre": {"id": "a", "datetime": "2026-07-02T00:00:00", "polarization": "HH"},
        "post": {"id": "b", "datetime": "2026-08-19T00:00:00", "polarization": "HH"},
        "layers": {},
    }  # fmt: skip
    (folder / "event.json").write_text(json.dumps(event))
    step = 0.35 / 100
    assert service.probe("box", 23.45 - 0.5 * step, 90.6).pre_db == -10.0  # inside, first row
    for lat, lon in ((23.45 + 0.3 * step, 90.6), (23.2, 90.55 - 0.3 * step)):
        with pytest.raises(AppError):
            service.probe("box", lat, lon)  # a third of a pixel outside: refused
