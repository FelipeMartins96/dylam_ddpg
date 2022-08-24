from gym.envs.registration import register

register(
    id="vssStrat-v0",
    entry_point="envs.vss_strat:VSSStratEnv",
    kwargs={},
    max_episode_steps=1200,
)

register(
    id="vssStrat1v1-v0",
    entry_point="envs.vss_strat:VSSStratEnv",
    kwargs={'n_robots_blue': 1, 'n_robots_yellow': 1},
    max_episode_steps=1200,
)

register(
    id="vssStrat2v1-v0",
    entry_point="envs.vss_strat:VSSStratEnv",
    kwargs={'n_robots_blue': 2, 'n_robots_yellow': 1},
    max_episode_steps=1200,
)

register(
    id="vssStrat1v2-v0",
    entry_point="envs.vss_strat:VSSStratEnv",
    kwargs={'n_robots_blue': 1, 'n_robots_yellow': 2},
    max_episode_steps=1200,
)
