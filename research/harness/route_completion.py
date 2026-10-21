"""Route-aware goal checks for model adapters, including adjacent-lane goals."""
import math
from typing import Any


def route_goal_reached(ego: Any, goal: Any, locations: list[Any], next_index: int) -> bool:
    if goal is None or len(locations) < 2 or len(locations) - next_index > 3:
        return False
    transform = ego.get_transform()
    position = transform.location
    if position.distance(goal) >= 5.0:
        return False
    end = locations[-1]
    # Find the last nondegenerate segment, ignoring duplicated endpoint samples.
    for start in reversed(locations[:-1]):
        dx, dy = end.x - start.x, end.y - start.y
        length = math.hypot(dx, dy)
        if length > 0.01:
            lateral = abs((position.x - end.x) * dy - (position.y - end.y) * dx) / length
            heading = math.degrees(math.atan2(dy, dx))
            error = abs((transform.rotation.yaw - heading + 180) % 360 - 180)
            return lateral < 1.75 and error < 30.0
    return False
