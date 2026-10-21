import pytest
from research.harness.config import AppConfig
from research.harness.runner import _control_cadence

@pytest.mark.parametrize('agent',['lav_lav','lav_fast','simlingo_simlingo'])
def test_documented_twenty_hz_agents_keep_native_control_cadence(agent):
    c=AppConfig();c.agent.kind='pcla';c.agent.pcla_agent_name=agent
    assert _control_cadence(c)==(.05,2)

@pytest.mark.parametrize('module', ['submission_agent', 'transfuser_agent'])
def test_transfuser_releases_keep_native_control_cadence(module):
    c=AppConfig();c.agent.kind='leaderboard-module';c.agent.module_name=module
    assert _control_cadence(c)==(.05,2)
