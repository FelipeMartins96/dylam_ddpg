import random

import gym
import numpy as np
from gym.spaces import Box, Dict
from rsoccer_gym.Entities import Ball, Frame, Robot
from rsoccer_gym.Utils import KDTree
from rsoccer_gym.Utils.Utils import OrnsteinUhlenbeckAction
from rsoccer_gym.vss.vss_gym_base import VSSBaseEnv


class VSSStratEnv(VSSBaseEnv):
    VSSBaseEnv.metadata['num_rewards'] = 4
    VSSBaseEnv.metadata['r_min'] = np.array([0.0*0.66, 0.0*0.32, -2.0*0.0053, 0.0*0.008])
    VSSBaseEnv.metadata['r_max'] = np.array([0.5*0.66, 1.0*0.32, -1.0*0.0053, 1.0*0.008])
    VSSBaseEnv.metadata['rewards_names'] = ['move', 'ball_grad', 'energy', 'goal']
    VSSBaseEnv.metadata["video.frames_per_second"] = 40

    def __init__(self, n_robots_blue=3, n_robots_yellow=3):
        super().__init__(
            field_type=0, n_robots_blue=n_robots_blue, n_robots_yellow=n_robots_yellow, time_step=0.025
        )

        self.num_actors = n_robots_blue + n_robots_yellow
        self.actors_keys = [f'b_{idx}' for idx in range(n_robots_blue)] + [f'y_{idx}' for idx in range(n_robots_yellow)]
        self.metadata['actors_keys'] = self.actors_keys
        self.mirror_entity = np.array([-1, -1, -1, -1, -1, -1, 1], dtype=np.float32)
        self.num_obs = 4 + 7*n_robots_blue + 7*n_robots_yellow
        self.action_space = Dict({actor_key: Box(low=-1, high=1, shape=(2,), dtype=np.float32) for actor_key in self.actors_keys})
        self.observation_space = Dict({actor_key: Box(low=-self.NORM_BOUNDS, high=self.NORM_BOUNDS, shape=(self.num_obs,), dtype=np.float32) for actor_key in self.actors_keys})

        # Initialize Class Atributes
        self.previous_ball_potential = None
        self.cumulative_reward = np.tile(np.array([0.0, 0.0, 0.0, 0.0]), (self.num_actors, 1))
        self.v_wheel_deadzone = 0.05
        self.move_scale = (120 / 0.66) / 40
        self.grad_scale = 0.75 / 0.32
        self.energy_scale = 40000 / 0.0053
        self.goal_scale = 1 / 0.008

        print("dylam_ddpg/envs/vss_strat Environment initialized")

    def reset(self):
        self.previous_ball_potential = None
        self.cumulative_reward = np.tile(np.array([0.0, 0.0, 0.0, 0.0]), (self.num_actors, 1))

        return super().reset()

    def step(self, action):
        observation, strat_reward, done, _ = super().step(action)

        original_reward = strat_reward.sum(axis=1)

        info = {actor: {'rewards': {}} for actor in self.actors_keys}
        for idx, actor in enumerate(self.actors_keys):
            info[actor]['rewards']['ep'] = self.cumulative_reward[idx]
            info[actor]['rewards']['step'] = strat_reward[idx]
        info['extra'] = {
            'goal_blue': 1 if strat_reward[0][3] > 0 else 0,
            'goal_yellow': 1 if strat_reward[0][3] < 0 else 0,
        }
        
        return observation, original_reward[0], done, info

    def _frame_to_observations(self):
        
        ball_np = np.array(
            [
                self.norm_pos(self.frame.ball.x),
                self.norm_pos(self.frame.ball.y),
                self.norm_v(self.frame.ball.v_x),
                self.norm_v(self.frame.ball.v_y),
            ],
            dtype=np.float32
        )
        blue_np = np.array(
            [
                [
                    self.norm_pos(self.frame.robots_blue[i].x),
                    self.norm_pos(self.frame.robots_blue[i].y),
                    np.sin(np.deg2rad(self.frame.robots_blue[i].theta)),
                    np.cos(np.deg2rad(self.frame.robots_blue[i].theta)),
                    self.norm_v(self.frame.robots_blue[i].v_x),
                    self.norm_v(self.frame.robots_blue[i].v_y),
                    self.norm_w(self.frame.robots_blue[i].v_theta),
                ]
                for i in range(self.n_robots_blue)
            ],
            dtype=np.float32
        )
        yellow_np = np.array(
            [
                [
                    self.norm_pos(self.frame.robots_yellow[i].x),
                    self.norm_pos(self.frame.robots_yellow[i].y),
                    np.sin(np.deg2rad(self.frame.robots_yellow[i].theta)),
                    np.cos(np.deg2rad(self.frame.robots_yellow[i].theta)),
                    self.norm_v(self.frame.robots_yellow[i].v_x),
                    self.norm_v(self.frame.robots_yellow[i].v_y),
                    self.norm_w(self.frame.robots_yellow[i].v_theta),
                ]
                for i in range(self.n_robots_yellow)
            ],
            dtype=np.float32
        )

        obs = {}
        for actor in self.actors_keys:
            c, i = actor.split('_')
            idx = int(i)

            if c == 'b':
                team_idxs = list(range(self.n_robots_blue))
                team_idxs.pop(idx)
                team_obs = np.stack([blue_np[[idx]], blue_np[team_idxs]]) if len(team_idxs) else blue_np
                obs[actor] = np.concatenate(
                    [
                        ball_np,
                        team_obs.flatten(),
                        yellow_np.flatten(),
                    ]
                )
            if c == 'y':
                team_idxs = list(range(self.n_robots_yellow))
                team_idxs.pop(idx)
                team_obs = np.stack([yellow_np[[idx]], yellow_np[team_idxs]]) if len(team_idxs) else yellow_np
                obs[actor] = np.concatenate(
                    [
                        -ball_np,
                        (self.mirror_entity * team_obs).flatten(),
                        (self.mirror_entity * blue_np).flatten(),
                    ]
                )

        return obs

    def _get_commands(self, actions):
        commands = []

        for actor in self.actors_keys:
            c, i = actor.split('_')
            is_yellow = c == 'y'
            idx = int(i)

            # Process deadzone
            v_wheel0, v_wheel1 = self._actions_to_v_wheels(actions[actor])

            commands.append(
                Robot(
                    yellow=is_yellow,
                    id=idx,
                    v_wheel0=v_wheel0,
                    v_wheel1=v_wheel1,
                )
            )

        return commands

    def _calculate_reward_and_done(self):
        rewards = np.zeros((self.num_actors, 4))
        
        goal_reward = self.__goal_reward()
        ball_grad_reward = self.__ball_grad()

        for i, actor in enumerate(self.actors_keys):
            c, id = actor.split('_')
            idx = int(id)

            if c == 'b':
                robot = self.frame.robots_blue[idx]
                last_robot = self.last_frame.robots_blue[idx]
                rewards[i] = np.array([
                    self.__move_reward(robot, last_robot),
                    ball_grad_reward,
                    self.__energy_penalty(i),
                    goal_reward,
                ])
            if c == 'y':
                robot = self.frame.robots_yellow[idx]
                last_robot = self.last_frame.robots_yellow[idx]
                rewards[i] = np.array([
                    self.__move_reward(robot, last_robot),
                    -ball_grad_reward,
                    self.__energy_penalty(i),
                    -goal_reward,
                ])

        self.cumulative_reward += rewards
        
        return rewards, (goal_reward != 0)

    def _get_initial_positions_frame(self):
        """Returns the position of each robot and ball for the initial frame"""
        field_half_length = self.field.length / 2
        field_half_width = self.field.width / 2

        def x():
            return random.uniform(-field_half_length + 0.1, field_half_length - 0.1)

        def y():
            return random.uniform(-field_half_width + 0.1, field_half_width - 0.1)

        def theta():
            return random.uniform(0, 360)

        pos_frame: Frame = Frame()

        pos_frame.ball = Ball(x=x(), y=y())

        min_dist = 0.1

        places = KDTree()
        places.insert((pos_frame.ball.x, pos_frame.ball.y))

        for i in range(self.n_robots_blue):
            pos = (x(), y())
            while places.get_nearest(pos)[1] < min_dist:
                pos = (x(), y())

            places.insert(pos)
            pos_frame.robots_blue[i] = Robot(x=pos[0], y=pos[1], theta=theta())

        for i in range(self.n_robots_yellow):
            pos = (x(), y())
            while places.get_nearest(pos)[1] < min_dist:
                pos = (x(), y())

            places.insert(pos)
            pos_frame.robots_yellow[i] = Robot(x=pos[0], y=pos[1], theta=theta())

        return pos_frame

    def _actions_to_v_wheels(self, actions):
        left_wheel_speed = actions[0] * self.max_v
        right_wheel_speed = actions[1] * self.max_v

        left_wheel_speed, right_wheel_speed = np.clip(
            (left_wheel_speed, right_wheel_speed), -self.max_v, self.max_v
        )

        # Deadzone
        if -self.v_wheel_deadzone < left_wheel_speed < self.v_wheel_deadzone:
            left_wheel_speed = 0

        if -self.v_wheel_deadzone < right_wheel_speed < self.v_wheel_deadzone:
            right_wheel_speed = 0

        # Convert to rad/s
        left_wheel_speed /= self.field.rbt_wheel_radius
        right_wheel_speed /= self.field.rbt_wheel_radius

        return left_wheel_speed, right_wheel_speed

    def __goal_reward(self):
        goal = 0
        if self.frame.ball.x > self.field.length / 2:
            goal = 1
        elif self.frame.ball.x < -self.field.length / 2:
            goal = -1
        return goal / self.goal_scale

    def __ball_grad(self):
        assert self.last_frame is not None

        # Calculate previous ball dist
        last_ball = self.last_frame.ball
        last_ball_pos = np.array([last_ball.x, last_ball.y])
        goal_pos = np.array([self.field.length / 2, 0])
        last_ball_dist = np.linalg.norm(goal_pos - last_ball_pos)

        # Calculate new ball dist
        ball = self.frame.ball
        ball_pos = np.array([ball.x, ball.y])
        ball_dist = np.linalg.norm(goal_pos - ball_pos)

        ball_dist_rw = last_ball_dist - ball_dist
        ball_dist_rw = ball_dist_rw / self.grad_scale

        return ball_dist_rw

    def __move_reward(self, robot, last_robot):
        """Calculate Move to ball reward

        Cosine between the robot vel vector and the vector robot -> ball.
        This indicates rather the robot is moving towards the ball or not.
        """

        ball = np.array([self.frame.ball.x, self.frame.ball.y])
        robot_pos = np.array([robot.x, robot.y])
        last_robot_pos = np.array([last_robot.x, last_robot.y])
        
        last_ball_dist = np.linalg.norm(ball - last_robot_pos)
        ball_dist = np.linalg.norm(ball - robot_pos)

        move_reward = last_ball_dist - ball_dist

        return move_reward / self.move_scale

    def __energy_penalty(self, id):
        """Calculates the energy penalty"""
        command = self.sent_commands[id]
        en_penalty_1 = abs(command.v_wheel0)
        en_penalty_2 = abs(command.v_wheel1)
        energy_penalty = -(en_penalty_1 + en_penalty_2)

        return energy_penalty / self.energy_scale
