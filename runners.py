# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ddpg/#ddpg_continuous_actionpy
from collections import namedtuple
from copy import deepcopy
import os
import random
import time
from distutils.util import strtobool
import hydra

import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from hydra.utils import instantiate

import envs

Agent = namedtuple("Agent", ["name", "actors", "instance"])

def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/")
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

def runner(cfg):
    run_name = f"{cfg.env_id}__{cfg.exp_name}__{cfg.seed}"
    
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
    
    writer = SummaryWriter(f"logs/{run_name}")
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
    # Change vector env spaces to non dict, as its the same for all actors
    dummy_env = gym.make(cfg.env_id)
    env_actors_keys = dummy_env.metadata['actors_keys']
    envs.single_action_space = dummy_env.action_space[env_actors_keys[0]]
    envs.single_observation_space = dummy_env.observation_space[env_actors_keys[0]]
    dummy_env.close()
    del dummy_env

    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # List of agents and its controlled actors
    agents = []
    for name, value in cfg.agents.items():
        agents.append(
            Agent(
                name=name, 
                actors=value.actors,
                instance=instantiate(cfg[value.agent], envs=envs, device=device)
            )
        )
        if 'net' in value.keys():
            agents[-1].instance.load_actor(os.path.join(hydra.utils.get_original_cwd(), value.net))

    start_time = time.time()
    
    # TRY NOT TO MODIFY: start the game
    obs = envs.reset()
    for envs_step in tqdm(range(0, int(cfg.total_timesteps), envs.num_envs)):
        global_step = envs_step / envs.num_envs

        # Fill an dict with actions for every actor by looping every agent
        actions = {}
        for agent in agents:
            for actor in agent.actors:
                actions[actor] = agent.instance.sample_actions(obs[actor])

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        for info in infos:
            if "episode" in info.keys():
                writer.add_scalar("extra/episodic_length", info["episode"]["l"], envs_step)
                [writer.add_scalar(f'extra/{k}', v, envs_step) for k, v in info['extra'].items()]
                
                for actor_key in env_actors_keys:
                    ep_rws = info[actor_key]['rewards']['ep']
                    [writer.add_scalar(f'actor_{actor_key}/rw_{n}/episodic_return', ep_rws[i], envs_step) for i, n in enumerate(envs.metadata['rewards_names'])]
                    writer.add_scalar(f'actor_{actor_key}/rw_sum/episodic_return', ep_rws.sum(), envs_step)
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        real_next_obs = deepcopy(next_obs)
        for idx, d in enumerate(dones):
            if d:
                for actor_key in env_actors_keys:
                    real_next_obs[actor_key][idx] = infos[idx]["terminal_observation"][actor_key]
                    dones[idx] = not infos[idx].get("TimeLimit.truncated", False)

        for agent in agents:
            for actor in agent.actors:
                agent.instance.observe(obs[actor], real_next_obs[actor], actions[actor], rewards, dones, infos, actor)

            update_info = agent.instance.update(global_step)

            if update_info is not None and global_step % 100 == 0:
                for k, i in update_info.items():
                    writer.add_scalar(k, i, envs_step)

        writer.add_scalar("charts/SPS", int(envs_step / (time.time() - start_time)), envs_step)
        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

    for agent in agents:
        agent.instance.save_actor(f'model_{agent.name}.pt')

    envs.close()
    writer.close()
