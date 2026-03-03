from LLM4sim.simulations.mail_flow import MailFlowEnvironment, MailFlowSimulation


def test_mail_flow_runs_with_defaults():
    sim = MailFlowSimulation(config={})
    env = MailFlowEnvironment(sim)

    state, _ = env.reset(seed=123)
    assert "queue_size" in state

    for _ in range(5):
        result = env.step(action=0)
        assert isinstance(result.reward, float)
