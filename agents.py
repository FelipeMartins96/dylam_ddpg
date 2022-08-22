from networks import DylamQNetwork, QNetwork, Actor
import torch.optim as optim
from buffers import StratReplayBuffer
import numpy as np
import torch
import torch.nn.functional as F

class StratLastRewards:
    def __init__(self, size, n_rewards):
        self.pos = 0
        self.size = size
        self._can_do = False
        self.rewards = np.zeros((size, n_rewards))

    def add(self, rewards):
        self.rewards[self.pos] = rewards
        if self.pos == self.size - 1:
            self._can_do = True
        self.pos = (self.pos + 1) % self.rewards.shape[0]

    def can_do(self):
        return self._can_do

    def mean(self):
        return self.rewards.mean(0)

class AgentRandomNormal:
    def __init__(
            self, 
            envs, 
            device,
            sigma
    ):
        self.device = device
        self.envs = envs
        self.sigma = sigma
        self.action_bias = np.tile((self.envs.single_action_space.high + self.envs.single_action_space.low) / 2.0, (self.envs.num_envs, 1))
        self.action_scale = np.tile((self.envs.single_action_space.high - self.envs.single_action_space.low) / 2.0, (self.envs.num_envs, 1))
    
    def sample_actions(self, obs):
        return np.random.normal(self.action_bias, self.action_scale * self.sigma).clip(self.envs.single_action_space.low, self.envs.single_action_space.high)

    def observe(self, obs, _obs, actions, rws, dones, infos, actor_key):
        pass
    
    def update(self, global_step):
        return None

class AgentRandomOU:
    def __init__(
            self, 
            envs, 
            device,
            sigma,
            theta
    ):
        self.device = device
        self.envs = envs
        self.sigma = sigma
        self.theta = theta
        self.action_bias = np.tile((self.envs.single_action_space.high + self.envs.single_action_space.low) / 2.0, (self.envs.num_envs, 1))
        self.action_scale = np.tile((self.envs.single_action_space.high - self.envs.single_action_space.low) / 2.0, (self.envs.num_envs, 1))
        self.noise = self.action_bias.copy()
    
    def sample_actions(self, obs):
        self.noise += self.theta * (self.action_bias - self.noise) + np.random.normal(self.action_bias, self.action_scale * self.sigma)
        self.noise = np.clip(self.noise, self.envs.single_action_space.low, self.envs.single_action_space.high)
        return self.noise

    def observe(self, obs, _obs, actions, rws, dones, infos, actor_key):
        for idx, info in enumerate(infos):
            if "episode" in info.keys():
                self.noise[idx] = self.action_bias[idx]
        pass
    
    def update(self, global_step):
        return None


class AgentDDPG:
    def __init__(
            self, 
            envs, 
            device,
            batch_size,
            buffer_size,
            exploration_noise,
            gamma,
            learning_rate_q,
            learning_rate_actor,
            learning_starts,
            policy_frequency,
            tau,
        ):
        self.actor = Actor(envs).to(device)
        self.qf1 = QNetwork(envs).to(device)
        self.qf1_target = QNetwork(envs).to(device)
        self.target_actor = Actor(envs).to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.q_optimizer = optim.Adam(list(self.qf1.parameters()), lr=learning_rate_q)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=learning_rate_actor)

        self.num_rewards = 1

        envs.single_observation_space.dtype = np.float32
        self.rb = StratReplayBuffer(
            int(buffer_size),
            envs.single_observation_space,
            envs.single_action_space,
            self.num_rewards,
            device,
            handle_timeout_termination=False,
        )

        self.device = device
        self.envs = envs
        self.exploration_noise = exploration_noise
        self.learning_starts = int(learning_starts)
        self.batch_size = int(batch_size)
        self.gamma = gamma
        self.tau = tau
        self.policy_frequency = int(policy_frequency)

    def sample_actions(self, obs):
        with torch.no_grad():
            if self.rb.size() != 0 and self.rb.size() < self.learning_starts:
                actions = self.actor.action_bias.expand(self.envs.num_envs,-1).clone()
            else:
                actions = self.actor(torch.Tensor(obs).to(self.device))
            actions += torch.normal(self.actor.action_bias.expand(self.envs.num_envs,-1), self.actor.action_scale * self.exploration_noise)
            actions = actions.cpu().numpy().clip(self.envs.single_action_space.low, self.envs.single_action_space.high)
        return actions

    def observe(self, obs, _obs, actions, rws, dones, infos, actor_key):
        for idx, info in enumerate(infos):
            s = slice(idx,idx+1)
            self.rb.add(
                obs[s],
                _obs[s],
                actions[s],
                info[actor_key]["rewards"]["step"].sum().reshape(1,-1),
                dones[s],
                infos[s]
            )
    
    def update(self, global_step):
        if self.rb.size() < self.learning_starts or self.rb.size() == 0:
            return None

        info = {}
        data = self.rb.sample(self.batch_size)

        with torch.no_grad():
            next_state_actions = self.target_actor(data.next_observations)
            qf1_next_targets = self.qf1_target(data.next_observations, next_state_actions)
            next_q_values = data.rewards + (1 - data.dones.expand(-1, self.num_rewards)) * self.gamma * qf1_next_targets

        qf1_a_values = self.qf1(data.observations, data.actions)
        qf1_losses = torch.nn.MSELoss(reduction='none')(qf1_a_values, next_q_values).mean(0)
        qf1_loss = torch.sum(qf1_losses)

        # optimize the model
        self.q_optimizer.zero_grad()
        qf1_loss.backward()
        self.q_optimizer.step()

        info['losses/qf1_loss'] = qf1_loss.item()
        with torch.no_grad():
            qf1_a_values_mean = qf1_a_values.mean(0)
        info[f'rw_total/qf1_a_value'] = qf1_a_values_mean[0].item()

        if global_step % self.policy_frequency == 0:
            qf1s = self.qf1(data.observations, self.actor(data.observations))
            actor_loss = -qf1s.mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # update the target network
            for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            info.update({
                "losses/actor_loss": actor_loss.item(),
            })

        return info

    def save_actor(self, path):
        torch.save(self.actor.state_dict(), path)

class AgentDylamDDPG:
    def __init__(
            self, 
            envs, 
            device,
            batch_size,
            buffer_size,
            dynamic,
            exploration_noise,
            gamma,
            learning_rate_q,
            learning_rate_actor,
            learning_starts,
            n_last_ep_rewards,
            policy_frequency,
            rew_tau,
            tau,
        ):
        self.actor = Actor(envs).to(device)
        self.qf1 = DylamQNetwork(envs).to(device)
        self.qf1_target = DylamQNetwork(envs).to(device)
        self.target_actor = Actor(envs).to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.q_optimizer = optim.Adam(list(self.qf1.parameters()), lr=learning_rate_q)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=learning_rate_actor)
        
        self.num_rewards = envs.metadata['num_rewards']

        envs.single_observation_space.dtype = np.float32
        self.rb = StratReplayBuffer(
            int(buffer_size),
            envs.single_observation_space,
            envs.single_action_space,
            self.num_rewards,
            device,
            handle_timeout_termination=False,
        )

        self.device = device
        self.envs = envs

        self.batch_size = batch_size
        self.dynamic = dynamic
        self.exploration_noise = exploration_noise
        self.gamma = gamma
        self.learning_starts = learning_starts
        self.policy_frequency = policy_frequency
        self.rew_tau =rew_tau
        self.tau = tau

        self.last_epi_rewards = StratLastRewards(n_last_ep_rewards, self.num_rewards)
        self.r_min = torch.tensor(envs.metadata['r_min'], dtype=torch.float, device=self.device)
        self.r_max = torch.tensor(envs.metadata['r_max'], dtype=torch.float, device=self.device)
        self.rw_names = envs.metadata['rewards_names']
        self.last_rew_mean = None

    def sample_actions(self, obs):
        with torch.no_grad():
            if self.rb.size() != 0 and self.rb.size() < self.learning_starts:
                actions = self.actor.action_bias.expand(self.envs.num_envs,-1).clone()
            else:
                actions = self.actor(torch.Tensor(obs).to(self.device))
            actions += torch.normal(self.actor.action_bias.expand(self.envs.num_envs,-1), self.actor.action_scale * self.exploration_noise)
            actions = actions.cpu().numpy().clip(self.envs.single_action_space.low, self.envs.single_action_space.high)
        return actions

    def observe(self, obs, _obs, actions, rws, dones, infos, actor_key):
        for idx, info in enumerate(infos):
            if "episode" in info.keys():
                self.last_epi_rewards.add(info[actor_key]["rewards"]["ep"])
            s = slice(idx,idx+1)
            self.rb.add(
                obs[s],
                _obs[s],
                actions[s],
                info[actor_key]["rewards"]["step"].reshape(1,-1),
                dones[s],
                infos[s]
            )
    
    def update(self, global_step):
        if self.rb.size() < self.learning_starts or self.rb.size() == 0:
            return None
        
        if self.dynamic and self.last_epi_rewards.can_do():
            rew_mean_t = torch.Tensor(self.last_epi_rewards.mean()).to(self.device)
            if self.last_rew_mean is not None:
                rew_mean_t = rew_mean_t + (self.last_rew_mean - rew_mean_t) * (1- self.rew_tau)
            dQ = torch.clamp((self.r_max - rew_mean_t) / (self.r_max - self.r_min), 0, 1)
            expdQ = torch.exp(dQ) - 1
            lambdas = expdQ / (torch.sum(expdQ, 0) + 1e-4)
            self.last_rew_mean = rew_mean_t
        else:
            lambdas = torch.ones(self.num_rewards, dtype=torch.float, device=self.device) / self.num_rewards

        info = {}
        data = self.rb.sample(self.batch_size)

        with torch.no_grad():
            next_state_actions = self.target_actor(data.next_observations)
            qf1_next_targets = self.qf1_target(data.next_observations, next_state_actions)
            next_q_values = data.rewards + (1 - data.dones.expand(-1, self.num_rewards)) * self.gamma * qf1_next_targets

        qf1_a_values = self.qf1(data.observations, data.actions)

        qf1_losses = torch.nn.MSELoss(reduction='none')(qf1_a_values, next_q_values).mean(0)
        qf1_loss = torch.sum(qf1_losses)

        # optimize the model
        self.q_optimizer.zero_grad()
        qf1_loss.backward()
        self.q_optimizer.step()

        info['losses/qf1_loss'] = qf1_loss.item()
        with torch.no_grad():
            qf1_a_values_mean = qf1_a_values.mean(0)
        for idx, n in enumerate(self.rw_names):
            info[f'rw_{n}/qf1_a_value'] = qf1_a_values_mean[idx].item()
            info[f'rw_{n}/lambda'] = lambdas[idx].item()

        if global_step % self.policy_frequency == 0:
            qf1s = self.qf1(data.observations, self.actor(data.observations))
            actor_loss = -(qf1s * lambdas).sum(1).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # update the target network
            for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            info.update({
                "losses/actor_loss": actor_loss.item(),
            })

        return info

    def save_actor(self, path):
        torch.save(self.actor.state_dict(), path)
