from gym.envs.registration import register

register(
    id="vssOri-v0",
    entry_point="envs.vss_strat:VSSStratEnv",
    kwargs={"stratified": False},
    max_episode_steps=1200,
)

register(
    id="vssStrat-v0",
    entry_point="envs.vss_strat:VSSStratEnv",
    max_episode_steps=1200,
)