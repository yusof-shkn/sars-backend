import numpy as np

from app.modules.events import ground, pipeline
from app.modules.events.ground import BRIGHT, MEDIUM, SMOOTH, WATER, ground_pair

WATER_DB = -16.0


def test_wind_ruffled_lake_stays_water_on_both_dates():
    # Calm lake before (-20 dB); wind on the after date lifts it to -12 dB, still water-dark.
    pre = np.array([[-20.0]], "float32")
    post = np.array([[-12.0]], "float32")
    a, b = ground_pair(pre, post, WATER_DB)
    assert a[0, 0] == WATER and b[0, 0] == WATER


def test_ruffled_on_the_before_date_works_the_other_way_round():
    a, b = ground_pair(np.array([[-11.0]]), np.array([[-21.0]]), WATER_DB)
    assert a[0, 0] == WATER and b[0, 0] == WATER


def test_water_that_really_became_bright_ground_is_not_kept_as_water():
    # Rubble or a building where a pond was: far above the water-dark line.
    a, b = ground_pair(np.array([[-20.0]]), np.array([[-5.0]]), WATER_DB)
    assert a[0, 0] == WATER and b[0, 0] == BRIGHT


def test_land_that_was_never_water_is_untouched():
    a, b = ground_pair(np.array([[-13.0, -8.0]]), np.array([[-13.0, -8.0]]), WATER_DB)
    assert a.tolist() == b.tolist() == [[SMOOTH, MEDIUM]]


def test_no_radar_on_one_date_stays_empty():
    a, b = ground_pair(np.array([[-20.0]]), np.array([[np.nan]]), WATER_DB)
    assert a[0, 0] == WATER and b[0, 0] == 0


def test_ruffled_water_line_matches_the_change_classifier():
    # The ground view and the change layer must call the same pixels "wind on water".
    assert ground.ROUGH_MAX_DB == pipeline.ROUGH_MAX_DB
    assert ground.ROUGH_FLAG == pipeline.ROUGH


def test_a_real_flood_is_not_hidden_by_the_wind_rule():
    # Wet field (-10 dB) that flooded (-20 dB): the change layer flags it darker (1). It must stay
    # land before and water after, or the flood disappears from the before -> after numbers.
    pre, post = np.array([[-10.0]]), np.array([[-20.0]])
    flags = np.array([[pipeline.DEC]], "uint8")
    a, b = ground_pair(pre, post, WATER_DB, flags)
    assert a[0, 0] == MEDIUM and b[0, 0] == WATER


def test_wind_flagged_pixels_stay_water_when_flags_are_given():
    flags = np.array([[pipeline.ROUGH, 0]], "uint8")
    a, b = ground_pair(np.array([[-20.0, -20.0]]), np.array([[-12.0, -14.0]]), WATER_DB, flags)
    assert a.tolist() == [[WATER, WATER]] and b.tolist() == [[WATER, WATER]]


def test_a_small_unflagged_shift_across_a_class_line_does_not_change_the_class():
    # -6.5 -> -5.5 dB crosses the medium/bright line, but 1 dB is below the change threshold
    # (flag 0): the ground did not measurably change, so the after date keeps the before class.
    flags = np.array([[0]], "uint8")
    a, b = ground_pair(np.array([[-6.5]]), np.array([[-5.5]]), WATER_DB, flags)
    assert a[0, 0] == MEDIUM and b[0, 0] == MEDIUM


def test_a_flagged_change_takes_the_after_dates_own_class():
    flags = np.array([[pipeline.VOLUME]], "uint8")
    a, b = ground_pair(np.array([[-9.0]]), np.array([[-4.0]]), WATER_DB, flags)
    assert a[0, 0] == MEDIUM and b[0, 0] == BRIGHT
