import torch
from research.harness.leaderboard_bridge import LeaderboardModuleAdapter


def test_transfuser_coverage_observes_features_before_waypoint_decoder():
    adapter = LeaderboardModuleAdapter.__new__(LeaderboardModuleAdapter)
    adapter._agent = type("TransFuserAgent", (), {})()
    adapter.torch_model = torch.nn.Module()
    adapter.torch_model.encoder = torch.nn.Linear(4, 16)
    adapter.torch_model.output = torch.nn.Linear(16, 2)
    seen = []
    hook = adapter.register_activation_hook(lambda module, inputs, output: seen.append(output.shape))
    adapter.torch_model.output(adapter.torch_model.encoder(torch.ones(1, 4)))
    assert hook["layer_name"] == "encoder"
    assert seen == [torch.Size([1, 16])]
    hook["handle"].remove()
