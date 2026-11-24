import math

import pytest

from research.harness.interaction_geometry import closest_approach, constant_velocity_ttc, footprint_clearance, world_footprint
from research.tests.outcome_helpers import actor


def test_rear_end_uses_surface_clearance_and_relative_velocity():
    a, b = actor(vx=10), actor("other", x=24, vx=0)
    assert footprint_clearance(a, b) == 20
    assert constant_velocity_ttc(a, b, 3).ttc_s == 2
    assert constant_velocity_ttc(b, a, 3).ttc_s == 2


@pytest.mark.parametrize("b", [actor("b", x=20, vx=20), actor("b", x=20, vx=10),
                                  actor("b", x=20, y=3, vx=0)])
def test_receding_same_speed_and_parallel_miss_have_no_ttc(b):
    assert constant_velocity_ttc(actor(), b, 10).status == "no_contact_within_horizon"


def test_crossing_paths_only_conflict_when_time_intervals_overlap():
    a = actor(x=-10, vx=10)
    b = actor("b", x=0, y=-10, yaw_rad=math.pi / 2, vx=0, vy=10)
    assert constant_velocity_ttc(a, b, 3).ttc_s == pytest.approx(.7)
    late = actor("late", x=0, y=-40, yaw_rad=math.pi / 2, vx=0, vy=10)
    assert constant_velocity_ttc(a, late, 5).ttc_s is None


def test_rotated_local_box_offset_and_gap():
    a = actor(yaw_rad=math.pi / 2, bbox_offset_x=3, bbox_yaw_rad=math.pi / 2)
    points = world_footprint(a)
    assert sum(x for x, _ in points) / 4 == pytest.approx(0)
    assert sum(y for _, y in points) / 4 == pytest.approx(3)
    assert max(x for x, _ in points) == pytest.approx(2)
    b = actor("b", y=10)
    assert footprint_clearance(a, b) == pytest.approx(5)


def test_overlap_is_prediction_not_sensor_collision_and_decks_are_separate():
    a, b = actor(), actor("b", x=1)
    assert constant_velocity_ttc(a, b, 3).status == "already_overlapping"
    assert footprint_clearance(a, b) == 0
    overpass = actor("above", z_min_m=5, z_max_m=7)
    assert footprint_clearance(a, overpass) is None
    assert constant_velocity_ttc(a, overpass, 3).ttc_s is None


def test_swept_sat_does_not_miss_short_contact_between_discrete_steps():
    a = actor(x=-50, vx=1000, half_length_m=.1, half_width_m=.1)
    b = actor("b", vx=0, half_length_m=.1, half_width_m=.1)
    assert constant_velocity_ttc(a, b, 1).ttc_s == pytest.approx(.0498)


def test_horizon_and_tangent():
    a, b = actor(), actor("b", x=24, y=2, vx=0)
    assert constant_velocity_ttc(a, b, 1).ttc_s is None
    assert constant_velocity_ttc(a, b, 2).ttc_s == pytest.approx(2)
    with pytest.raises(ValueError):
        constant_velocity_ttc(a, b, float("nan"))


def test_closest_approach_diagnoses_near_miss_paths_that_never_intersect():
    # Ego moving along y=0 at vx=10; other stopped at x=10, y=3 (half-widths=1 each -> gap=1 m at t=1.0 s)
    a = actor(x=0, y=0, vx=10, vy=0, half_length_m=2, half_width_m=1)
    b = actor("b", x=10, y=3, vx=0, vy=0, half_length_m=2, half_width_m=1)
    assert constant_velocity_ttc(a, b, 3.0).ttc_s is None
    ca = closest_approach(a, b, 3.0)
    assert ca.status == "closest_approach_within_horizon"
    assert ca.min_clearance_m == pytest.approx(1.0, abs=1e-4)
    assert 0.6 <= ca.time_s <= 1.4
    # Vertical separation yields vertical_separation with None clearance
    above = actor("c", x=10, y=0, vx=0, z_min_m=5, z_max_m=7)
    ca_sep = closest_approach(a, above, 3.0)
    assert ca_sep.status == "vertical_separation" and ca_sep.min_clearance_m is None
