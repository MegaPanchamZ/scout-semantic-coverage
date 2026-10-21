from threading import Thread
import time

import pytest
from research.harness.leaderboard_bridge import LeaderboardSensorSuite


def test_frame_wait_does_not_mix_old_camera_with_current_gnss():
    suite = LeaderboardSensorSuite(None, None)
    suite._store_data("rgb", 0.4, 8, "old-image")
    suite._store_data("gps", 0.5, 10, "current-position")
    def deliver_camera():
        time.sleep(0.01)
        suite._store_data("rgb", 0.5, 10, "current-image")
    thread = Thread(target=deliver_camera)
    thread.start()
    suite.wait_for_frame(["rgb", "gps"], 10, timeout=1)
    data = suite.read_input_data(0.5, frame=10)
    thread.join()
    assert data["rgb"] == (10, "current-image")
    assert data["gps"] == (10, "current-position")


def test_missing_current_camera_fails_instead_of_reusing_stale_data():
    suite = LeaderboardSensorSuite(None, None)
    suite._store_data("rgb", 0.4, 8, "old-image")
    with pytest.raises(TimeoutError, match="rgb"):
        suite.wait_for_frame(["rgb"], 10, timeout=0.01)


@pytest.mark.parametrize("native,expected_tick", [(False, "0.05"), (True, "0")])
def test_native_imu_uses_every_frame_as_official_wrapper(monkeypatch, native, expected_tick):
    from types import SimpleNamespace
    import research.harness.leaderboard_bridge as bridge
    attributes = {}
    blueprint = SimpleNamespace(set_attribute=lambda key, value: attributes.update({key: value}))
    sensor = SimpleNamespace(listen=lambda callback: None)
    world = SimpleNamespace(
        get_blueprint_library=lambda: SimpleNamespace(find=lambda name: blueprint),
        spawn_actor=lambda *args, **kwargs: sensor,
    )
    carla = SimpleNamespace(Location=lambda **kwargs: None, Rotation=lambda **kwargs: None,
                            Transform=lambda *args: None)
    original = bridge.importlib.import_module
    monkeypatch.setattr(bridge.importlib, "import_module", lambda name: carla if name == "carla" else original(name))
    suite = LeaderboardSensorSuite(world, None, native_leaderboard_defaults=native)
    suite._spawn_imu({"id": "imu", "sensor_tick": 0.05})
    assert attributes["sensor_tick"] == expected_tick
    if native:
        assert attributes["noise_accel_stddev_z"] == "0.015"
        assert attributes["noise_gyro_stddev_z"] == "0.001"
