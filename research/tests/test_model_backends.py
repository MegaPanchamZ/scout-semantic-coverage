"""Offline tests for the pluggable VLA model backends (no torch / no network)."""
from __future__ import annotations

import numpy as np
import pytest

from research.harness import model_backends as mb


def _features(tmp_path):
    images = {}
    for cam in ("front_camera", "front_left_camera", "front_right_camera"):
        paths = []
        for i in range(4):
            p = tmp_path / f"{cam}_{i}.png"
            p.write_bytes(b"\x89PNG\r\n\x1a\n" + bytes([i]))
            paths.append(str(p))
        images[cam] = paths
    return {
        "images": images,
        "vehicle_velocity": [3.0, 4.0],
        "vehicle_acceleration": [1.0, 0.0],
        "driving_command": "forward",
        "sensor_data_path": None,
    }


def test_text_to_bins():
    text = "blah <action_0><action_2047> trailing <action_12>"
    assert mb.text_to_bins(text) == [0, 2047, 12]


def test_bins_to_trajectory_shape_and_forward():
    code_book = np.zeros((mb.NBINS, 6, 4, 2), dtype=np.float64)
    # point all agents/modes forward at +x for every bin -> straight line
    code_book[:, :, :, 0] = 1.0
    traj = mb.bins_to_trajectory([1] * 10, code_book)
    assert traj.shape == (10, 3)
    assert np.all(np.isfinite(traj))
    # x increases monotonically for a forward codebook
    assert np.all(np.diff(traj[:, 0]) > 0)


def test_bins_pad_and_truncate():
    code_book = np.zeros((mb.NBINS, 6, 4, 2), dtype=np.float64)
    assert mb.bins_to_trajectory([1, 2, 3], code_book).shape == (10, 3)
    assert mb.bins_to_trajectory(list(range(50)), code_book).shape == (10, 3)


def test_http_backend_payload_and_decode(tmp_path, monkeypatch):
    captured = {}

    def fake_post(url, payload, timeout):
        captured["url"] = url
        captured["payload"] = payload
        return {"trajectory": [[1.0, 0.0, 0.0]] * 10, "text": "ok", "action_bins": [1] * 10}

    monkeypatch.setattr(mb, "_post_json", fake_post)
    backend = mb.HttpPlanBackend("http://host:9999/")
    traj, text = backend.plan(_features(tmp_path))
    assert captured["url"] == "http://host:9999/plan"
    assert captured["payload"]["command"] == "forward"
    assert captured["payload"]["velocity"] == 5.0  # hypot(3,4)
    assert captured["payload"]["acceleration"] == 1.0
    assert len(captured["payload"]["frames"]["front_camera"]) == 4
    assert traj.shape == (10, 3)
    assert text == "ok"
    assert backend.supports_activation_hooks is False


def test_http_backend_missing_trajectory(tmp_path, monkeypatch):
    monkeypatch.setattr(mb, "_post_json", lambda *a, **k: {"action_bins": []})
    backend = mb.HttpPlanBackend("http://host:9999")
    with pytest.raises(mb.BackendUnavailable):
        backend.plan(_features(tmp_path))


def test_openai_backend_messages(tmp_path, monkeypatch):
    code_book = np.zeros((mb.NBINS, 6, 4, 2), dtype=np.float64)
    code_book[:, :, :, 0] = 1.0
    backend = mb.OpenAIChatBackend(
        "http://host:8100", model="autovla", codebook_path=_write_codebook(tmp_path, code_book)
    )
    msgs = backend._messages(_features(tmp_path))
    assert msgs[0]["role"] == "system"
    content = msgs[1]["content"]
    assert sum(1 for c in content if c["type"] == "image_url") == 12
    assert any("velocity of the vehicle is 5.000" in c.get("text", "") for c in content)

    def fake_post(url, payload, timeout):
        assert url == "http://host:8100/v1/chat/completions"
        return {"choices": [{"message": {"content": "<action_1>" * 10}}]}

    monkeypatch.setattr(mb, "_post_json", fake_post)
    traj, text = backend.plan(_features(tmp_path))
    assert traj.shape == (10, 3)


def test_openai_backend_no_actions(tmp_path, monkeypatch):
    code_book = np.zeros((mb.NBINS, 6, 4, 2), dtype=np.float64)
    backend = mb.OpenAIChatBackend("http://host:8100", codebook_path=_write_codebook(tmp_path, code_book))
    monkeypatch.setattr(mb, "_post_json", lambda *a, **k: {"choices": [{"message": {"content": "no tokens"}}]})
    with pytest.raises(mb.BackendUnavailable):
        backend.plan(_features(tmp_path))


def test_make_backend_selection():
    with pytest.raises(ValueError):
        mb.make_backend("nope")
    with pytest.raises(ValueError):
        mb.make_backend("http")  # needs endpoint
    assert isinstance(mb.make_backend("http", endpoint="http://x"), mb.HttpPlanBackend)
    assert isinstance(mb.make_backend("sglang", endpoint="http://x"), mb.HttpPlanBackend)
    assert isinstance(mb.make_backend("openai", endpoint="http://x"), mb.OpenAIChatBackend)
    assert isinstance(mb.make_backend("gguf", endpoint="http://x"), mb.OpenAIChatBackend)


def test_plan_backend_has_no_torch_model():
    backend = mb.HttpPlanBackend("http://x")
    with pytest.raises(mb.BackendUnavailable):
        _ = backend.torch_model


def _write_codebook(tmp_path, code_book):
    import pickle

    p = tmp_path / "agent_vocab.pkl"
    with open(p, "wb") as f:
        pickle.dump({"token_all": {"veh": code_book}}, f)
    return p
