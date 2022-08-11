from networks import QNetwork, Actor
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
import numpy as np
import torch
import torch.nn.functional as F

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
        self.rb = ReplayBuffer(
            args.buffer_size,
            envs.single_observation_space,
            envs.single_action_space,
            device,
            handle_timeout_termination=True,
        )

        self.device = device
        self.envs = envs
        self.args = args
    
    def sample_actions(self, obs):
        with torch.no_grad():
            actions = self.actor(torch.Tensor(obs).to(self.device))
            actions += torch.normal(self.actor.action_bias.expand(self.envs.num_envs,-1), self.actor.action_scale * self.args.exploration_noise)
            actions = actions.cpu().numpy().clip(self.envs.single_action_space.low, self.envs.single_action_space.high)
        return actions

    def observe(self, obs, _obs, actions, rws, dones, infos):
        for i in range(len(dones)):
            s = slice(i,i+1)
            self.rb.add(
                obs[s],
                _obs[s],
                actions[s],
                rws[s],
                dones[s],
                infos[s]
            )
    
    def update(self, global_step):
        if self.rb.size() < self.args.learning_starts:
            return None
        
        info = {}
        data = self.rb.sample(self.args.batch_size)
        with torch.no_grad():
            next_state_actions = self.target_actor(data.next_observations)
            qf1_next_target = self.qf1_target(data.next_observations, next_state_actions)
            next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * self.args.gamma * (qf1_next_target).view(-1)

        qf1_a_values = self.qf1(data.observations, data.actions).view(-1)
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)

        # optimize the model
        self.q_optimizer.zero_grad()
        qf1_loss.backward()
        self.q_optimizer.step()

        info['qf1_loss'] = qf1_loss.item()

        if global_step % self.args.policy_frequency == 0:
            actor_loss = -self.qf1(data.observations, self.actor(data.observations)).mean()
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
                "qf1_values": qf1_a_values.mean().item(),
            })

        return info
