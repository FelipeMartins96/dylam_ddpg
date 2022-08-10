from gym.envs.registration import register

register(
    id="vssStrat-v0",
    entry_point="envs.vss_strat:VSSStratEnv",
    kwargs={},
    max_episode_steps=1200,
)