from robosuite.wrappers import VisualizationWrapper
import robosuite as suite
from robosuite.wrappers import GymWrapper
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import numpy as np
from typing import Callable, List, Optional, Tuple, Union
import os
import gym
import numpy as np


class mainUtils(object):
    def __init__(self,
                 simulation_total_time=5,  # sec
                 control_freq=50,
                 ):
        self.simulation_total_time = simulation_total_time  # sec
        self.control_freq = control_freq
        self.Horizon = self.control_freq * self.simulation_total_time

    def get_control_param(self, Use_video_for_simulation):
        plotting = Use_video_for_simulation
        collect_data = Use_video_for_simulation
        Control_param = dict(type='JOINT_TORQUE', input_max=1, input_min=-1,
                             output_max=[0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
                             output_min=[-0.05, -0.05, -0.05, -0.5, -0.5, -0.5], kp=700, damping_ratio=np.sqrt(2),
                             impedance_mode='fixed', kp_limits=[0, 100000], damping_ratio_limits=[0, 10],
                             position_limits=None,
                             orientation_limits=None, uncouple_pos_ori=True, control_delta=True, interpolation=None,
                             ramp_ratio=0.2,
                             control_dim=108,
                             plotting=plotting,
                             collect_data=collect_data,
                             simulation_total_time=self.simulation_total_time)
        return Control_param

    def get_gym_like_env(self, Use_video_for_simulation):
        has_renderer = Use_video_for_simulation
        Env = GymWrapper(
            suite.make(
                "PegInHole",
                # "Lift",
                robots="UR5e",  # use UR5e robot
                # user_init_qpos=[-0.470, -1.735, 2.480, -2.275, -1.590, -1.991],# EC - set initial position of the joints
                user_init_qpos=[-0.09525062, -0.8314843, 1.20965442, -1.94896648, -1.57079633, -1.66604693],
                # EC - set initial position of the joints
                use_camera_obs=False,  # do not use pixel observations
                has_offscreen_renderer=False,  # not needed since not using pixel obs
                has_renderer=has_renderer,
                reward_shaping=True,  # use dense rewards
                ignore_done=False,
                horizon=self.Horizon,
                control_freq=self.control_freq,
                controller_configs=control_param,
                Peg_density=1,

            )
        )
        # make sure the env is valid for stable_baselines3
        # check_env(Env, warn=True)

        return Env

    def execute_learning(self, Model_name, Env):
        model = PPO("MlpPolicy", Env, verbose=1)
        model.learn(total_timesteps=3)
        model.save(Model_name)

    def model_test(self, Model_name, Env):
        model = PPO.load(Model_name)
        Obs = Env.reset()
        Done = False
        while not Done:
            Action, _states = model.predict(Obs)
            Obs, Rewards, Done, Info = Env.step(Action)
            Env.render()

    def simulation(self, Env, Use_video_for_simulation):
        Env.reset()
        Done = False
        Action = np.random.randn(Env.robots[0].action_dim)  # the impedance parameters

        while not Done:
            Obs, Reward, Done, Info = Env.step(Action)  # take action in the environment
            if Use_video_for_simulation:
                Env.render()  # render on display
        #
        # # EC -  get all path information
        # info = Env.get_path_info()
        #
        # # Wrap this environment in a visualization wrapper
        # Env = VisualizationWrapper(Env, indicator_configs=None)


if __name__ == "__main__":
    # 'use_video_for_simulation' variable decide whether to do RL or to display the simulation
    # where True means display simulation and False means do RL

    # use_video_for_simulation = True
    use_video_for_simulation = False

    M = mainUtils()
    # control definitions
    control_param = M.get_control_param(use_video_for_simulation)
    # environment definitions
    env = M.get_gym_like_env(use_video_for_simulation)
    # start RL
    model_name = "IMP_model_v1"
    M.execute_learning(model_name, env)
    # # test the model
    # M.model_test(model_name, env)
    # run the simulation without a model
    # M.simulation(env, use_video_for_simulation)
