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
import pygad
from scipy.optimize import differential_evolution as de
from scipy.optimize import Bounds


class Simulation(object):
    def __init__(self,
                 simulation_total_time=6,  # sec
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
                             control_dim=16,
                             plotting=plotting,
                             collect_data=collect_data,
                             simulation_total_time=self.simulation_total_time)
        return Control_param

    def get_gym_like_env(self, Use_video_for_simulation, control_param, radial_error=0.0, angular_error=0.0):
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
                radial_error=radial_error,
                angular_error=angular_error,

            )
        )
        # make sure the env is valid for stable_baselines3
        # print("action space")
        # print(Env.action_space)
        # print("observation space")
        # print(Env.observation_space)
        # # check_env(Env, warn=True)

        return Env

    def simulation(self, Env, Use_video_for_simulation):
        Env.reset()
        Done = False
        Action = np.random.randn(Env.robots[0].action_dim)  # the impedance parameters

        while not Done:
            Obs, Reward, Done, Info = Env.step(Action)  # take action in the environment
            if Use_video_for_simulation:
                Env.render()  # render on display
        return


class RlClass(object):

    def execute_learning(self, Model_name, Env):
        model = PPO("MlpPolicy", Env, n_steps=20, batch_size=5, verbose=1)
        model.learn(total_timesteps=10000)
        model.save(Model_name)

    def model_test(self, Model_name, Env):
        model = PPO.load(Model_name)
        Obs = Env.reset()
        Done = False
        while not Done:
            Action, _states = model.predict(Obs)
            Obs, Rewards, Done, Info = Env.step(Action)
            Env.render()


class GaClass(object):
    def __init__(self,
                 num_generations=5,  # sec
                 num_parents_mating=4,
                 sol_per_pop=8,
                 num_genes=6,
                 init_range_low=-2,
                 init_range_high=5,
                 parent_selection_type="sss",
                 keep_parents=1,
                 crossover_type="single_point",
                 mutation_type=None,
                 mutation_percent_genes=1,
                 Env=None,
                 ):
        function_inputs = [4, -2, 3.5, 5, -11, -4.7]  # Function inputs.
        desired_output = 44  # Function output.

        def fitness_func(solution, solution_idx):
            a = Env.action_space
            print(a)
            output = np.sum(solution * function_inputs)
            fitness = 1.0 / (np.abs(output - desired_output) + 0.000001)
            return fitness

        self.ga_instance = pygad.GA(num_generations=num_generations,
                                    num_parents_mating=num_parents_mating,
                                    fitness_func=fitness_func,
                                    sol_per_pop=sol_per_pop,
                                    num_genes=num_genes,
                                    init_range_low=init_range_low,
                                    init_range_high=init_range_high,
                                    parent_selection_type=parent_selection_type,
                                    keep_parents=keep_parents,
                                    crossover_type=crossover_type,
                                    mutation_type=mutation_type,
                                    mutation_percent_genes=mutation_percent_genes)
