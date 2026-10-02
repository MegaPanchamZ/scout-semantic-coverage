from __future__ import annotations

from types import SimpleNamespace

from research.harness.scenario_runtime import LeadVehicleBrakingController


class _Location:
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        self.x, self.y, self.z = x, y, z

    def distance(self, other: "_Location") -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5


class _Control:
    def __init__(self, throttle: float = 0.0, brake: float = 0.0, steer: float = 0.0, hand_brake: bool = False) -> None:
        self.throttle, self.brake, self.steer, self.hand_brake = throttle, brake, steer, hand_brake


class _Actor:
    def __init__(self, location: _Location) -> None:
        self.location = location
        self.controls: list[_Control] = []

    def apply_control(self, control: _Control) -> None:
        self.controls.append(control)

    def get_location(self) -> _Location:
        return self.location

    def get_transform(self):
        return SimpleNamespace(location=self.location, rotation=SimpleNamespace(yaw=0.0))


def _controller(release_after_s):
    params = {"trigger_location": {"x": 0.0, "y": 0.0, "z": 0.0}, "trigger_radius_m": 8.0,
              "pre_brake_throttle": 0.0, "post_trigger_brake": 1.0}
    if release_after_s is not None:
        params["release_after_s"] = release_after_s
    controller = LeadVehicleBrakingController(params)
    controller._actor = _Actor(_Location(20.0, 0.0))
    controller._pre_brake_control = _Control(throttle=0.0)
    controller._post_brake_control = _Control(brake=1.0)
    carla = SimpleNamespace(Location=_Location, VehicleControl=_Control)
    world = SimpleNamespace(get_map=lambda: SimpleNamespace(get_waypoint=lambda loc: SimpleNamespace(next=lambda d: [])))
    context = {"carla": carla, "world": world, "ego_vehicle": _Actor(_Location(1.0, 0.0))}
    return controller, context


def test_lead_releases_brake_after_hold():
    controller, context = _controller(4.0)
    for tick in range(60):
        controller.on_tick(tick, context)
    controls = controller._actor.controls
    assert controls[0].brake == 1.0          # brakes when triggered
    assert controls[39].brake == 1.0         # still braking at 3.9 s
    assert controls[40].brake == 0.0 and controls[40].throttle >= 0.5  # drives off at 4 s
    assert context["lead_vehicle_braking_active"] is False


def test_lead_without_release_holds_brake():
    controller, context = _controller(None)
    for tick in range(60):
        controller.on_tick(tick, context)
    assert all(control.brake == 1.0 for control in controller._actor.controls)
