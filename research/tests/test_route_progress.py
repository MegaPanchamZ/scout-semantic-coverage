import pytest

from research.harness.route_progress import RouteProgressTracker
from research.tests.outcome_helpers import actor


def test_arc_length_reversal_and_high_water():
    tracker = RouteProgressTracker([(0, 0), (10, 0), (10, 10)])
    assert tracker.update(actor(), 0).s_m == 0
    assert tracker.update(actor(x=6), .1).s_m == 6
    assert tracker.update(actor(x=10, y=2), .2).s_m == 12
    reversed_position = tracker.update(actor(x=8), .3)
    assert reversed_position.s_m == 8
    assert reversed_position.high_water_m == 12
    assert tracker.summary() == reversed_position
    assert tracker.provenance()["route_id"] == reversed_position.route_id
    assert tracker.provenance()["threshold_status"] == "experimental"


def test_self_intersection_and_loop_do_not_jump_to_later_branch():
    tracker = RouteProgressTracker([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0), (10, 0)])
    assert tracker.update(actor(), 0).s_m == 0
    assert tracker.update(actor(x=5), .1).s_m == 5  # not s=45 on identical geometry
    assert tracker.update(actor(x=10), .2).s_m == 10
    tracker.reset(40)
    assert tracker.update(actor(), .3).s_m == 40
    assert tracker.update(actor(x=5), .4).s_m == 45


@pytest.mark.parametrize("loss", ["gap", "off_route", "discontinuity", "teleport"])
def test_loss_requires_explicit_anchor_and_changes_continuity_id(loss):
    tracker = RouteProgressTracker([(0, 0), (100, 0)])
    first = tracker.update(actor(), 0)
    time = .3 if loss == "gap" else .1
    position = actor(x=50 if loss == "teleport" else 1, y=10 if loss == "off_route" else 0)
    assert tracker.update(position, time, discontinuity=loss == "discontinuity").status == "unknown"
    assert tracker.update(actor(x=1), time + .1).status == "unknown"
    tracker.reset(1)
    reacquired = tracker.update(actor(x=1), time + .2)
    assert reacquired.status == "available"
    assert reacquired.continuity_id != first.continuity_id


def test_route_points_offsets_and_timestamps_validated():
    with pytest.raises(ValueError):
        RouteProgressTracker([(0, 0), (0, 0)])
    with pytest.raises(ValueError):
        RouteProgressTracker([(0, 0), (float("nan"), 0)])
    tracker = RouteProgressTracker([(0, 0), (0, 0), (10, 0)])
    assert tracker.update(actor(y=2), 0).cross_track_m == 2
    with pytest.raises(ValueError):
        tracker.update(actor(), 0)
    with pytest.raises(ValueError):
        tracker.reset(11)


def test_first_projection_respects_supplied_anchor():
    assert RouteProgressTracker([(0, 0), (100, 0)]).update(actor(x=50), 0).status == "unknown"
    assert RouteProgressTracker([(0, 0), (100, 0)], initial_s_m=50).update(actor(x=50), 0).s_m == 50
