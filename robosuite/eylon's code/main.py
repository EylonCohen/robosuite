from robosuite.wrappers import VisualizationWrapper
import robosuite as suite
from robosuite.wrappers import GymWrapper
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from multiprocessing import Pool

import numpy as np
from typing import Callable, List, Optional, Tuple, Union
import os
import gym

import numpy as np
import time
from robosuite.utils.learning_class import RlClass, Simulation, GaClass
from scipy.optimize import differential_evolution as de
import pickle
from robosuite.utils.control_utils import build_imp_matrices_circular_peg, is_pos_def, is_stable_system

if __name__ == "__main__":
    # 'use_video_for_simulation' variable decide whether to do RL or to display the simulation
    # where True means display simulation and False means do RL
    #
    # use_video_for_simulation = True
    # sim = Simulation(simulation_total_time=10, control_freq=50)
    # # control definitions
    # control_param = sim.get_control_param(use_video_for_simulation)
    # # environment definitions
    # start = time.time()
    # env = sim.get_gym_like_env(use_video_for_simulation, control_param, radial_error=0.0025, angular_error=np.deg2rad(0))
    # end = time.time()
    # print(end - start)
    # sim.simulation(env, use_video_for_simulation)

    use_video_for_simulation = False
    sim = Simulation(simulation_total_time=10, control_freq=1)
    # control definitions
    control_param = sim.get_control_param(use_video_for_simulation)
    # environment definitions
    env_number = 1
    min_radial_error = 0.0012  # [m]
    max_radial_error = 0.0025  # [m]
    radial_error_vec = np.linspace(min_radial_error, max_radial_error, num=env_number)
    # radial_error_vec = np.random.uniform(high=max_radial_error, low=min_radial_error, size=env_number)
    min_angular_error = 0.0  # [m]
    rounds = 5
    max_angular_error = np.deg2rad(360 * rounds)  # [rad]
    angular_error_vec = np.linspace(min_angular_error, max_angular_error, num=env_number)
    # envs = [sim.get_gym_like_env(use_video_for_simulation, control_param,
    #                              radial_error=radial_error_vec[i], angular_error=angular_error_vec[i])
    #         for i in range(env_number)]
    env = sim.get_gym_like_env(use_video_for_simulation, control_param,
                               radial_error=0.0012, angular_error=0)


    def fitness_func(action):
        total_cost = 6000 * env_number  # this is for the case A has positive eig
        K_imp, C_imp, M_imp, A, A_d, B_d = build_imp_matrices_circular_peg(action)
        if is_stable_system(A):
            total_cost = 0.0
            # idx = 0
            action = np.concatenate((action, np.array([1])))  # the 1 is for the gripper to be closed
            for i in range(env_number):
                env.reset()
                Obs, cost, Done, Info = env.step(action)  # take action in the environment
                if not env.robots[0].controller.is_robot_stable_bool:
                    # total_cost += (env_number - idx) * 5000  # get a better result if the peg did get in i previous runs
                    total_cost += 5000  # add cost but keep checking other solutions
                    # print(total_cost)
                    # return total_cost
                else:
                    total_cost += cost
                    # idx += 1
        total_cost = total_cost / env_number  # to reduce the numbers value
        print(total_cost)
        return total_cost


    # x = np.arange(20*12).reshape(20, 12)
    #
    # with Pool(4) as pwl:
    #     output = pwl.map(fitness_func, x)
    #
    # print(output)

    # # K_bounds = [K1, K2, K3, K4, K5, K6]
    # K_bounds = [(0, 10000), (0.0001, 0.01), (0, 10000),
    #             (0, 100), (0, 10000), (0, 10000)]
    # # M_bounds = [M1, M2, M3, M4, M5, M6]
    # M_bounds = [(0, 100), (0, 100), (0, 100),
    #             (0, 100), (0, 100), (0, 100)]
    # # bounds = [K_bounds, C_bounds, M_bounds]
    # bounds = [K_bounds, M_bounds]
    # x0 = np.array([7.55022625e+03, 6.78138403e-03, 3.97646868e+03, 6.41925579e+01,
    #                7.48784099e+03, 8.32513966e+03, 6.53180852e+01, 9.70823481e+01,
    #                1.40992887e+00, 5.85738701e+01, 5.00239987e+01, 1.51216855e+01])

    # K_bounds = [K1, K2, K3, K4, K5, K6, K7, K8, K9, K10]
    K_bounds = [(0, 10000), (0, 10000), (0.0001, 0.01), (0, 10000), (0, 10000),
                (0, 100), (0, 10000), (-10000, 0), (-10000, 0), (0, 10000)]
    # M_bounds = [M1, M2, M3, M4, M5, M6]
    M_bounds = [(0, 100), (0, 100), (0, 100),
                (0, 100), (0, 100), (0, 100)]
    # bounds = [K_bounds, C_bounds, M_bounds]
    bounds = [K_bounds, M_bounds]

    # # K_bounds = [K1, K2, K3, K4, K5, K6]
    # K_bounds = [(0, 10000), (0, 10000), (100, 10000), (0, 10000), (0, 10000), (0, 10)]
    # # M_bounds = [M1, M2, M3, M4, M5, M6]
    # M_bounds = [(0, 100), (0, 100), (0, 100), (0, 100), (0, 100), (0, 100)]
    # # C_bounds = [C1, C2, C3, C4, C5, C6]
    # C_bounds = [(0, 10000), (0, 10000), (0, 10000), (0, 10000), (0, 10000), (0, 10000)]
    # # W_bounds = [W1, W2, W4, W5, K7, W8, W9, W10]
    # W_bounds = [(0, 10000), (0, 10000), (0, 10000), (0, 10000), (0, 10000), (-10000, 0),
    #             (-10000, 0), (0, 10000)]
    # bounds = [K_bounds, M_bounds, C_bounds, W_bounds]

    # flattened the list
    bounds = [val for sublist in bounds for val in sublist]
    result = de(func=fitness_func, bounds=bounds, maxiter=1000, popsize=10, disp=True, workers=-1,
                updating='deferred', strategy='best2bin')
    pickle.dump(result, open("result.p", "wb"))
    saved_result = pickle.load(open("result.p", "rb"))
    print(saved_result)

    # RL = RlClass()
    # # start RL
    # model_name = "IMP_model_v1"
    # RL.execute_learning(model_name, env)
    # # # test the model
    # # RL.model_test(model_name, env)

    # Ga = GaClass(Env=env)
    # Ga.ga_instance.run()

    # solution, solution_fitness, solution_idx = Ga.ga_instance.best_solution()
    # print("Parameters of the best solution : {solution}".format(solution=solution))
    # print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
    # function_inputs = [4, -2, 3.5, 5, -11, -4.7]
    # prediction = np.sum(np.array(function_inputs) * solution)
    # print("Predicted output based on the best solution : {prediction}".format(prediction=prediction))

    # function_inputs = [4, -2, 3.5, 5, -11, -4.7]  # Function inputs.
    # desired_output = 44  # Function output.
    #
    #
    # def fitness_func(solution, *args):
    #     a = env.action_space
    #     # print(a)
    #     output = np.sum(solution * function_inputs)
    #     fitness = 1.0 / (np.abs(output - desired_output) + 0.000001)
    #     return -fitness
    #
    #
    # bounds = [(-10, 10), (-10, 10), (-10, 10), (-10, 10), (-10, 10), (-10, 10)]
    # result = de(func=fitness_func, bounds=bounds, updating='deferred', workers=-1)
    #
    # # function_inputs = np.array(function_inputs)
    # # temp = DEClass()
    # # result = temp.DE
    # r = np.dot(np.array(result.x), function_inputs)
    # print(result)
    # print(r)
    # pickle.dump(result, open("save.p", "wb"))
    # saved_result = pickle.load(open("save.p", "rb"))
    # print(saved_result)
