from dataclasses import replace

from research.harness.outcome_models import ActorState, CaptureManifest, ChannelCoverage, FrameEvidence

CHANNELS = ("actors", "ego_motion", "collision", "acceleration", "controls")


def actor(actor_id="ego", **kwargs):
    base = ActorState(actor_id, "vehicle", 0, 0, 0, 2, 1, 10, 0, 0, 2)
    return replace(base, **kwargs)


def frame(i, **kwargs):
    base = FrameEvidence("run", i, round(i * 0.1, 8), actor(x=i), available_channels=CHANNELS,
                         longitudinal_acceleration_mps2=0, brake=0)
    return replace(base, **kwargs)


def manifest(end=1.0, **kwargs):
    base = CaptureManifest("run", 0, end, "completed", "horizon_exhausted",
                           {name: ChannelCoverage("complete", 0, end) for name in CHANNELS},
                           reached_goal=False, scan_radius_m=50, projection_horizon_s=3, max_sample_interval_s=.1)
    return replace(base, **kwargs)
