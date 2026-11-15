import numpy as np
import pytest

from app.modules.events import resources
from app.modules.events.ground import ground_classes, ground_stats
from app.modules.events.pipeline import Grid
from app.modules.events.resources import (
    BASE_BYTES,
    BYTES_PER_PIXEL,
    INPUT_BYTES_PER_PIXEL,
    MAX_LOOKS,
    available_memory,
    choose_looks,
    multilook,
    multilook_grid,
)

GB = 2**30


def test_capable_machine_keeps_full_resolution():
    assert choose_looks(6_000_000, available=8 * GB) == 1


def test_small_machine_averages_just_enough():
    # 6 M px at full resolution needs ~2 GB; 800 MB free fits 2x2 (1.5 M px) but not 1x1.
    looks = choose_looks(6_000_000, available=800 * 2**20)
    assert looks == 2
    need = BASE_BYTES + 6_000_000 * INPUT_BYTES_PER_PIXEL + 6_000_000 / looks**2 * BYTES_PER_PIXEL
    assert need <= 800 * 2**20


def test_unknown_memory_means_full_resolution():
    assert choose_looks(50_000_000, available=None) == 1


def test_too_little_memory_even_at_max_looks_fails_clearly():
    with pytest.raises(SystemExit, match="smaller --bbox"):
        choose_looks(400_000_000, available=500 * 2**20)


def test_forced_looks_are_used_as_given():
    assert choose_looks(6_000_000, available=8 * GB, forced=3) == 3


def test_multilook_averages_power_and_skips_nan():
    a = np.array([[1, 3, 10, np.nan], [1, 3, 10, np.nan], [5, 5, 7, 7], [5, 5, 7, 7]], "float32")
    out = multilook(a, 2)
    assert out.shape == (2, 2)
    np.testing.assert_allclose(out, [[2, 10], [5, 7]])


def test_multilook_of_all_nan_block_stays_nan_and_drops_ragged_edge():
    a = np.full((5, 5), np.nan, "float32")
    a[:2, :2] = 4
    out = multilook(a, 2)
    assert out.shape == (2, 2)
    assert out[0, 0] == 4 and np.isnan(out[1, 1])


def test_multilook_one_is_a_no_op():
    a = np.arange(6, dtype="float32").reshape(2, 3)
    assert multilook(a, 1) is a


def test_multilook_grid_moves_to_block_centres():
    g = Grid(32645, x0=100.0, y0=500.0, dx=10.0, dy=-10.0)
    m = multilook_grid(g, 2)
    # First output pixel is the centre of the first 2x2 block of input pixel centres.
    assert (m.x0, m.y0, m.dx, m.dy) == (105.0, 495.0, 20.0, -20.0)
    assert multilook_grid(g, 1) is g


def test_available_memory_respects_a_cgroup_cap(monkeypatch):
    monkeypatch.setattr(resources, "_meminfo_available", lambda: 4 * GB)
    monkeypatch.setattr(resources, "_cgroup_headroom", lambda: 600 * 2**20)
    assert available_memory() == 600 * 2**20
    monkeypatch.setattr(resources, "_cgroup_headroom", lambda: None)
    assert available_memory() == 4 * GB


def test_max_looks_is_a_sane_ceiling():
    assert 2 <= MAX_LOOKS <= 8


def test_ground_classes_and_areas_compare_the_same_ground():
    pre = np.array([[-20, -14], [-8, -3]], "float32")
    post = np.array([[-20, -20], [np.nan, -3]], "float32")
    a, b = ground_classes(pre, -16), ground_classes(post, -16)
    assert a.tolist() == [[1, 2], [3, 4]] and b.tolist() == [[1, 1], [0, 4]]
    stats = ground_stats(a, b, [[0.0, 0.0], [0.02, 0.02]])
    # The pixel with no radar after is left out of BOTH dates.
    assert stats["medium"] == {"before_km2": 0.0, "after_km2": 0.0}
    assert stats["water"]["after_km2"] == pytest.approx(2 * stats["water"]["before_km2"], rel=1e-3)


def test_read_cache_is_full_size_on_a_capable_machine():
    assert resources.read_cache_bytes(8 * GB) == resources.READ_CACHE_MAX
    assert resources.read_cache_bytes(None) == resources.READ_CACHE_MAX


def test_read_cache_shrinks_to_a_quarter_of_free_memory_but_not_below_the_floor():
    assert resources.read_cache_bytes(400 * 2**20) == 100 * 2**20
    assert resources.read_cache_bytes(50 * 2**20) == resources.READ_CACHE_MIN
