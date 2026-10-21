from types import SimpleNamespace

import numpy as np
import pytest

from research.harness.gnss_coordinates import GnssRouteNormalizer
from research.harness.config import AppConfig
from research.harness.runner import _control_cadence
from research.harness.pcla_bridge import PclaAdapter


class Location:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = x, y, z


def route_gps(location):
    return {"lat": 42 - location.y / 100000, "lon": 2 + location.x / 80000}


@pytest.mark.parametrize("latitude_sign", [1, -1])
def test_gnss_matches_route_coordinates_without_mutating_sensor_data(latitude_sign):
    def geolocation(location):
        return SimpleNamespace(latitude=42 + latitude_sign * location.y / 100000,
                               longitude=2 + location.x / 80000)

    normalizer = GnssRouteNormalizer(
        SimpleNamespace(transform_to_geolocation=geolocation),
        Location(300, -250), route_gps,
    )
    # Test a point away from calibration, with noisy GPS and unchanged altitude.
    location = Location(315, -230)
    geo = geolocation(location)
    sample = np.array([geo.latitude + 1e-6, geo.longitude, 7.0])
    original = sample.copy()
    result = normalizer.normalize(sample)
    expected = route_gps(location)
    assert result[0] == pytest.approx(expected["lat"] - latitude_sign * 1e-6)
    assert result[1] == pytest.approx(expected["lon"])
    assert result[2] == 7.0
    np.testing.assert_array_equal(sample, original)


def test_interfuser_uses_two_native_steps_per_campaign_tick():
    config = AppConfig()
    config.agent.kind = "pcla"
    config.agent.pcla_agent_name = "if_if"
    assert _control_cadence(config) == (0.05, 2)
    config.harness.fixed_delta_seconds = 0.05
    assert _control_cadence(config) == (0.05, 1)


def test_other_agents_keep_configured_timestep():
    config = AppConfig()
    assert _control_cadence(config) == (0.1, 1)


def test_pcla_sensor_interface_normalizes_gnss_preserving_frame_and_other_sensors(monkeypatch):
    raw_gps = np.array([42.001, 2.002, 8.0])
    raw = {"gps": (123, raw_gps), "rgb": (123, object())}
    interface = SimpleNamespace(get_data=lambda: raw)
    carla_map = SimpleNamespace(transform_to_geolocation=lambda loc: SimpleNamespace(
        latitude=42 + loc.y / 100000, longitude=2 + loc.x / 80000,
    ))
    route_module = SimpleNamespace(
        _get_latlon_ref=lambda world: (42, 2),
        _location_to_gps=lambda lat, lon, loc: route_gps(loc),
    )
    monkeypatch.setattr("research.harness.pcla_bridge.importlib.import_module", lambda name: route_module)
    adapter = PclaAdapter.__new__(PclaAdapter)
    adapter._world = SimpleNamespace(get_map=lambda: carla_map)
    adapter._ego_vehicle = SimpleNamespace(get_location=lambda: Location())
    adapter._pcla = SimpleNamespace(agent_instance=SimpleNamespace(sensor_interface=interface))
    adapter._normalize_interfuser_gnss()
    result = interface.get_data()
    assert result["gps"][0] == 123
    np.testing.assert_allclose(result["gps"][1], [41.999, 2.002, 8.0])
    assert result["rgb"] is raw["rgb"]
    np.testing.assert_array_equal(raw_gps, [42.001, 2.002, 8.0])


def test_interfuser_rejects_incompatible_logical_timestep():
    config = AppConfig()
    config.agent.kind = "pcla"
    config.agent.pcla_agent_name = "if_if"
    config.harness.fixed_delta_seconds = 0.075
    with pytest.raises(ValueError, match="multiple of 0.05"):
        _control_cadence(config)
