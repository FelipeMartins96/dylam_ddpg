from networks import QNetwork, Actor
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

class AgentDDPG:
    def __init__(self, envs, device, args):
        self.actor = Actor(envs).to(device)
        self.qf1 = QNetwork(envs).to(device)
        self.qf1_target = QNetwork(envs).to(device)
        self.target_actor = Actor(envs).to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.q_optimizer = optim.Adam(list(self.qf1.parameters()), lr=args.learning_rate)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=args.learning_rate)

        envs.single_observation_space.dtype = np.float32
        self.rb = StratReplayBuffer(
            args.buffer_size,
            envs.single_observation_space,
            envs.single_action_space,
            envs.metadata['num_rewards'],
            device,
            handle_timeout_termination=True,
        )

        self.device = device
        self.envs = envs
        self.args = args

        self.num_rewards = envs.metadata['num_rewards']
        self.r_min = torch.tensor(envs.metadata['r_min'], dtype=torch.float, device=self.device)
        self.r_max = torch.tensor(envs.metadata['r_max'], dtype=torch.float, device=self.device)
        self.rw_names = envs.metadata['rewards_names']
        self.last_epi_rewards = StratLastRewards(10, self.num_rewards) # TODO: move 10 to args
        self.rew_tau = 0.995 # TODO: move to args
        self.last_rew_mean = None

    def sample_actions(self, obs):
        with torch.no_grad():
            actions = self.actor(torch.Tensor(obs).to(self.device))
            actions += torch.normal(self.actor.action_bias.expand(self.envs.num_envs,-1), self.actor.action_scale * self.args.exploration_noise)
            actions = actions.cpu().numpy().clip(self.envs.single_action_space.low, self.envs.single_action_space.high)
        return actions

    def observe(self, obs, _obs, actions, rws, dones, infos):
        for idx, d in enumerate(dones):
            if d:
                self.last_epi_rewards.add(infos[idx]["rewards"]["ep"])
            s = slice(idx,idx+1)
            self.rb.add(
                obs[s],
                _obs[s],
                actions[s],
                infos[idx]["rewards"]["step"].reshape(1,-1),
                dones[s],
                infos[s]
            )
    
    def update(self, global_step):
        if self.rb.size() < self.args.learning_starts:
            return None
        
        if self.last_epi_rewards.can_do():
            rew_mean_t = torch.Tensor(self.last_epi_rewards.mean()).to(self.device)
            if self.last_rew_mean is not None:
                rew_mean_t = rew_mean_t + (self.last_rew_mean - rew_mean_t) * self.rew_tau
            dQ = torch.clamp((self.r_max - rew_mean_t) / (self.r_max - self.r_min), 0, 1)
            expdQ = torch.exp(dQ) - 1
            lambdas = expdQ / (torch.sum(expdQ, 0) + 1e-4)
            self.last_rew_mean = rew_mean_t
        else:
            lambdas = torch.ones(self.num_rewards, dtype=torch.float, device=self.device) / self.num_rewards

        info = {}
        data = self.rb.sample(self.args.batch_size)

        with torch.no_grad():
            next_state_actions = self.target_actor(data.next_observations)
            qf1_next_targets = self.qf1_target(data.next_observations, next_state_actions)
            next_q_values = data.rewards + (1 - data.dones.expand(-1, self.num_rewards)) * self.args.gamma * qf1_next_targets

        qf1_a_values = self.qf1(data.observations, data.actions)

        qf1_losses = torch.nn.MSELoss(reduction='none')(qf1_a_values, next_q_values).mean(0)
        qf1_loss = torch.sum(qf1_losses)

        # optimize the model
        self.q_optimizer.zero_grad()
        qf1_loss.backward()
        self.q_optimizer.step()

        info['qf1_loss'] = qf1_loss.item()
        with torch.no_grad():
            qf1_a_values_mean = qf1_a_values.mean(0)
        for r in range(self.num_rewards):
            info[f'r{r}_qf1_a_value'] = qf1_a_values_mean[r].item()
            info[f'r{r}_lambda'] = lambdas[r].item()

        if global_step % self.args.policy_frequency == 0:
            qf1s = self.qf1(data.observations, self.actor(data.observations))
            actor_loss = -(qf1s * lambdas).sum(1).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # update the target network
            for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
                target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)
            for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)

            info.update({
                "actor_loss": actor_loss.item(),
            })

        return info
