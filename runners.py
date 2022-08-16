# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ddpg/#ddpg_continuous_actionpy
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from hydra.utils import instantiate

import envs

def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

def runner(cfg):
    run_name = f"{cfg.env_id}__{cfg.exp_name}__{cfg.seed}__{int(time.time())}"
    if cfg.track:
        import wandb

        wandb.init(
            project=cfg.wandb.project_name,
            entity=cfg.wandb.entity,
            sync_tensorboard=True,
            config=dict(cfg),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in dict(cfg).items()])),
    )
    

    # TRY NOT TO MODIFY: seeding
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.cuda else "cpu")

    # env setup
    envs = gym.vector.AsyncVectorEnv(
        [make_env(cfg.env_id, cfg.seed + i, i, cfg.capture_video, run_name) 
        for i in range(cfg.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = instantiate(cfg.agent, envs=envs, device=device)

    start_time = time.time()
    # TRY NOT TO MODIFY: start the game
    obs = envs.reset()
    for envs_step in tqdm(range(0, int(cfg.total_timesteps), envs.num_envs)):
        global_step = envs_step / envs.num_envs

        # ALGO LOGIC: put action logic here
        if envs_step < agent.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions = agent.sample_actions(obs)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        for info in infos:
            if "episode" in info.keys():
                writer.add_scalar("rw_total/episodic_return", info["episode"]["r"], envs_step)
                writer.add_scalar("extra/episodic_length", info["episode"]["l"], envs_step)
                ep_rws = info['rewards']['ep']
                [writer.add_scalar(f'rw_{n}/episodic_return', ep_rws[i], envs_step) for i, n in enumerate(envs.metadata['rewards_names'])]
                [writer.add_scalar(f'extra/{k}', v, envs_step) for k, v in info['rewards']['extra'].items()]
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(dones):
            if d:
                real_next_obs[idx] = infos[idx]["terminal_observation"]

        agent.observe(obs, real_next_obs, actions, rewards, dones, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        update_info = agent.update(global_step)

        if update_info is not None and global_step % 100 == 0:
            writer.add_scalar("charts/SPS", int(envs_step / (time.time() - start_time)), envs_step)
            for k, i in update_info.items():
                writer.add_scalar(k, i, envs_step)

    envs.close()
    writer.close()
