import numpy as np
import pytest
from types import SimpleNamespace
import math

from research.harness.autovla_inference import validate_trajectory
from research.harness.autovla_sensors import CameraHistory
from research.harness.autovla_bridge import AutoVlaAdapter


def test_camera_history_rejects_mismatched_and_stale_frames():
    history = CameraHistory(["front", "left", "right"], stride=5)
    for frame in (5, 10, 15, 20):
        for camera in ("front", "left"):
            history.add(camera, frame, (camera, frame))
    for frame in (5, 10, 15, 25):
        history.add("right", frame, ("right", frame))
    assert history.snapshot(20, wait_seconds=0) is None
    history.add("right", 20, ("right", 20))
    images, frames = history.snapshot(20, wait_seconds=0)
    assert frames == [5, 10, 15, 20]
    assert all([frame for _, frame in values] == frames for values in images.values())
    assert history.snapshot(25, wait_seconds=0) is None


@pytest.mark.parametrize("poses", [np.zeros((9, 3)), np.zeros((10, 2)), np.full((10, 3), np.nan)])
def test_incomplete_or_nonfinite_trajectory_is_rejected(poses):
    with pytest.raises(ValueError, match="ten finite"):
        validate_trajectory(poses)


def test_valid_stop_trajectory_remains_a_stop():
    np.testing.assert_array_equal(validate_trajectory(np.zeros((10, 3))), np.zeros((10, 3)))


def test_delayed_acceleration_plan_does_not_deadlock_at_rest(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "carla", SimpleNamespace(
        VehicleControl=lambda **values: SimpleNamespace(**values)))
    adapter = AutoVlaAdapter.__new__(AutoVlaAdapter)
    adapter._tick = adapter._last_inference_tick = 1
    adapter._fixed_delta_seconds = 0.1
    adapter._trajectory_origin = (0.0, 0.0, 0.0)
    adapter._target_speed = 5.5
    adapter._ego = SimpleNamespace(
        get_transform=lambda: SimpleNamespace(location=SimpleNamespace(x=0.0, y=0.0), rotation=SimpleNamespace(yaw=0.0)),
        get_velocity=lambda: SimpleNamespace(x=0.0, y=0.0, z=0.0),
    )
    poses = np.zeros((10, 3))
    poses[:, 0] = [0.003, 0.006, 0.08, 0.48, 1.3, 2.5, 4.1, 5.9, 7.9, 9.8]
    control = adapter._control_from_poses(poses)
    assert control.brake == 0.0
    assert control.throttle > 0.05


def test_steering_uses_vehicle_wheel_limits_and_world_space_wheelbase(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "carla", SimpleNamespace(
        VehicleControl=lambda **values: SimpleNamespace(**values)))
    adapter = AutoVlaAdapter.__new__(AutoVlaAdapter)
    wheels = [SimpleNamespace(max_steer_angle=angle, position=SimpleNamespace(x=x, y=y))
              for angle, x, y in [(70, 10000, 20200), (70, 10150, 20200),
                                  (0, 10000, 19900), (0, 10150, 19900)]]
    adapter._ego = SimpleNamespace(
        get_physics_control=lambda: SimpleNamespace(wheels=wheels,
            steering_curve=[SimpleNamespace(x=0, y=1), SimpleNamespace(x=20, y=.9)]),
        get_transform=lambda: SimpleNamespace(location=SimpleNamespace(x=0., y=0.), rotation=SimpleNamespace(yaw=0.)),
        get_velocity=lambda: SimpleNamespace(x=0., y=0., z=0.),
    )
    adapter._configure_vehicle_geometry()
    adapter._tick = adapter._last_inference_tick = 1
    adapter._fixed_delta_seconds = .1
    adapter._trajectory_origin = (0., 0., 0.)
    adapter._target_speed = 5.5
    poses = np.array([[2.5 * i, .25 * i, 0.] for i in range(1, 11)])
    control = adapter._control_from_poses(poses)
    # Lateral target at 1 s is (5, .5): wheel angle is ~6.8 degrees,
    # requiring about 9.7% of this car's 70-degree steering range.
    assert -.099 < control.steer < -.095
    assert adapter._wheelbase_m == pytest.approx(3.0)


@pytest.mark.parametrize("invalid", [False, True])
@pytest.mark.parametrize("mode", ["direct", "adaptive"])
def test_decoder_bounds_generation_and_rejects_invalid_action_ids(invalid, mode):
    import torch
    from research.harness.autovla_inference import AutoVlaInference

    calls = []
    processor = SimpleNamespace(
        tokenizer=SimpleNamespace(convert_tokens_to_ids=lambda token: 10 + int(token[8:-1])),
        decode=lambda ids: "actions",
    )
    def generate(**kwargs):
        calls.append(kwargs)
        scores = torch.ones(1, 30)
        restricted = kwargs["logits_processor"][0](kwargs["input_ids"], scores)
        if mode == "direct":
            assert torch.isneginf(restricted[0, :10]).all()
            assert torch.isneginf(restricted[0, 14:]).all()
        else:
            assert torch.isfinite(restricted).all()
            after_action = kwargs["logits_processor"][0](torch.tensor([[11]]), torch.ones(1, 30))
            assert torch.isneginf(after_action[0, :10]).all()
            assert torch.isneginf(after_action[0, 14:]).all()
        tokens = torch.full((1, 10), 29 if invalid else 11)
        if mode == "adaptive":
            tokens = torch.cat([torch.tensor([[4, 5]]), tokens], dim=1)
        assert bool(kwargs["stopping_criteria"][0](torch.tensor([[11] * 10]), None))
        assert not bool(kwargs["stopping_criteria"][0](torch.tensor([[11] * 9]), None))
        return torch.cat([kwargs["input_ids"], tokens], dim=1)

    model = SimpleNamespace(
        processor=processor, action_start_id=10, device="cpu",
        vlm=SimpleNamespace(set_attn_implementation=lambda name: None, generate=generate),
        action_tokenizer=SimpleNamespace(n_bins=4,
            decode_token_ids_to_trajectory=lambda ids: np.zeros((1, 11, 3))),
    )
    runtime = AutoVlaInference(model, decoding_mode=mode)
    class Inputs(dict):
        @property
        def input_ids(self):
            return self["input_ids"]
    runtime._prompt = lambda features: Inputs(input_ids=torch.tensor([[1, 2, 3]]))
    if invalid:
        with pytest.raises(ValueError, match="ten valid action tokens"):
            runtime.predict({})
    else:
        poses, _ = runtime.predict({})
        assert poses.shape == (10, 3)
        assert calls[0]["max_new_tokens"] == (10 if mode == "direct" else 384)
        assert calls[0]["do_sample"] is (mode == "adaptive")


def test_nuplan_camera_selection_and_straight_command_match_released_features():
    from research.harness.autovla_bridge import CAMERA_SPECS
    assert [camera[-1] for camera in CAMERA_SPECS] == [0., -90., 90.]
    adapter = AutoVlaAdapter.__new__(AutoVlaAdapter)
    adapter._route = []
    assert adapter._driving_command() == "keep forward"
