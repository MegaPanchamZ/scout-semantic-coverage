from types import SimpleNamespace
import math
import pytest
from research.harness.route_completion import route_goal_reached


def location(x, y):
    point = SimpleNamespace(x=x, y=y)
    point.distance = lambda other: math.hypot(x-other.x, y-other.y)
    return point


@pytest.mark.parametrize('index,yaw,lateral,expected', [
    (0, 0, 0, False),  # Close in space, but hundreds of route points remain.
    (599, 180, 0, False),  # Wrong direction at the endpoint.
    (599, 0, 3.5, False),  # Adjacent lane still falls inside old 5 m radius.
    (599, 0, 0.3, True),
])
def test_route_endpoint_requires_progress_lane_and_heading(index, yaw, lateral, expected):
    route = [location(i*.1, 0) for i in range(600)]
    ego = SimpleNamespace(get_transform=lambda: SimpleNamespace(
        location=location(59, lateral), rotation=SimpleNamespace(yaw=yaw)))
    assert route_goal_reached(ego, location(60, 0), route, index) is expected
