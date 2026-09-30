from __future__ import annotations

import os
from pathlib import Path


class GlobalConfig:
    turn_KP = 1.25
    turn_KI = 0.75
    turn_KD = 0.3
    turn_n = 40

    speed_KP = 5.0
    speed_KI = 0.5
    speed_KD = 1.0
    speed_n = 40

    max_throttle = 0.75
    brake_speed = 0.1
    brake_ratio = 1.1
    clip_delta = 0.35

    max_speed = 5
    collision_buffer = [2.5, 1.2]
    momentum = 0
    skip_frames = 1
    detect_threshold = 0.04
    model = "interfuser_baseline"

    def __init__(self, **kwargs):
        workspace_root = Path(__file__).resolve().parents[2]
        default_model_path = workspace_root / "research" / "models" / "InterFuser" / "leaderboard" / "team_code" / "interfuser.pth.tar"
        self.model_path = os.environ.get("INTERFUSER_MODEL_PATH", str(default_model_path))
        for key, value in kwargs.items():
            setattr(self, key, value)