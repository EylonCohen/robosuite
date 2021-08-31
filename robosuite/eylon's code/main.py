from robosuite.wrappers import VisualizationWrapper
import robosuite as suite
from robosuite.wrappers import GymWrapper
import numpy as np
from typing import Callable, List, Optional, Tuple, Union
import os
import gym
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.interpolate import interp1d


def plotter(info):
    time = np.array(info.get("time"))
    min_jerk_position = np.array(info.get("min_jerk_position_vec"))
    min_jerk_velocity = np.array(info.get("min_jerk_velocity_vec"))
    min_jerk_acceleration = np.array(info.get("min_jerk_acceleration_vec"))
    min_jerk_orientation = np.array(info.get("min_jerk_orientation_vec"))
    min_jerk_orientation_dot = np.array((info.get("min_jerk_orientation_dot")))
    min_jerk_angle_velocity = np.array(info.get("min_jerk_angle_velocity_vec"))

    impedance_position = np.array(info.get("impedance_position_vec"))
    impedance_velocity = np.array(info.get("impedance_velocity_vec"))
    impedance_orientation = np.array(info.get("impedance_orientation_vec"))
    impedance_angle_velocity = np.array(info.get("impedance_angle_velocity_vec"))

    real_position = np.array(info.get("real_position_vec"))
    real_velocity = np.array(info.get("real_velocity_vec"))
    real_orientation = np.array(info.get("real_orientation_vec"))
    real_angle_velocity = np.array(info.get("real_angle_velocity_vec"))
    interaction_forces = np.array(info.get("interaction_forces_vec"))
    new_torque = np.array((info.get("new_torque")))

    plt.figure()

    ax1 = plt.subplot(311)
    ax1.plot(time, min_jerk_position[:, 0], 'g', label=" X reference")
    ax1.plot(time, impedance_position[:, 0], 'b--', label=" X impedance")
    ax1.plot(time, real_position[:, 0], 'r--', label=" X real")
    ax1.legend()
    ax1.set_title("X [m]")

    ax2 = plt.subplot(312)
    ax2.plot(time, min_jerk_position[:, 1], 'g', label=" Y reference")
    ax2.plot(time, impedance_position[:, 1], 'b--', label=" Y impedance")
    ax2.plot(time, real_position[:, 1], 'r--', label=" Y real")
    ax2.legend()
    ax2.set_title("Y [m]")

    ax3 = plt.subplot(313)
    ax3.plot(time, min_jerk_position[:, 2], 'g', label=" Z reference")
    ax3.plot(time, impedance_position[:, 2], 'b--', label=" Z impedance")
    ax3.plot(time, real_position[:, 2], 'r--', label=" Z real")
    ax3.legend()
    ax3.set_title("Z [m]")

    plt.tight_layout()

    # ----------------------------------------------------------------------
    plt.figure()

    ax1 = plt.subplot(311)
    ax1.plot(time, min_jerk_velocity[:, 0], 'g', label="$\dot X$ minimum jerk")
    ax1.plot(time, real_velocity[:, 0], 'r--', label=" $\dot X$ real")
    ax1.legend()
    ax1.set_title("$\dot X$ [m/s]")

    ax2 = plt.subplot(312)
    ax2.plot(time, min_jerk_velocity[:, 1], 'g', label=" $\dot Y$ minimum jerk")
    ax2.plot(time, real_velocity[:, 1], 'r--', label=" $\dot Y$ real")
    ax2.legend()
    ax2.set_title("$\dot Y$ [m/s]")

    ax3 = plt.subplot(313)
    ax3.plot(time, min_jerk_velocity[:, 2], 'g', label=" $\dot Z$ minimum jerk")
    ax3.plot(time, real_velocity[:, 2], 'r--', label=" $\dot Z$ real")
    ax3.legend()
    ax3.set_title("$\dot Z$ [m/s]")

    plt.tight_layout()
    # ----------------------------------------------------------------------
    plt.figure()

    ax1 = plt.subplot(311)
    ax1.plot(time, min_jerk_orientation[:, 0], 'g', label="orientation 1st element - minimum jerk")
    ax1.plot(time, impedance_orientation[:, 0], 'b--', label="orientation 1st element - impedance")
    ax1.plot(time, real_orientation[:, 0], 'r--', label="orientation 1st element - real")
    ax1.legend()
    ax1.set_title("orientation 1st element")

    ax2 = plt.subplot(312)
    ax2.plot(time, min_jerk_orientation[:, 1], 'g', label="orientation 2nd element - minimum jerk")
    ax2.plot(time, impedance_orientation[:, 1], 'b--', label="orientation 2nd element - impedance")
    ax2.plot(time, real_orientation[:, 1], 'r--', label="orientation 2nd element - real")
    ax2.legend()
    ax2.set_title("orientation 2nd element")

    ax3 = plt.subplot(313)
    ax3.plot(time, min_jerk_orientation[:, 2], 'g', label="orientation 3rd element - minimum jerk")
    ax3.plot(time, impedance_orientation[:, 2], 'b--', label="orientation 3rd element - impedance")
    ax3.plot(time, real_orientation[:, 2], 'r--', label="orientation 3rd element - real")
    ax3.legend()
    ax3.set_title("orientation 3rd element")

    plt.tight_layout()
    # -------------------------------------------------------------
    plt.figure()

    ax1 = plt.subplot(311)
    ax1.plot(time, min_jerk_angle_velocity[:, 0], 'g', label="Wx - minimum jerk")
    # ax1.plot(time, impedance_orientation[:, 0], 'b--', label="orientation 1st element - impedance")
    ax1.plot(time, real_angle_velocity[:, 0], 'r--', label="Wx  - real")
    ax1.legend()
    ax1.set_title("Wx")

    ax2 = plt.subplot(312)
    ax2.plot(time, min_jerk_angle_velocity[:, 1], 'g', label="Wy - minimum jerk")
    # ax2.plot(time, impedance_orientation[:, 1], 'b--', label="orientation 2nd element - impedance")
    ax2.plot(time, real_angle_velocity[:, 1], 'r--', label="Wy - real")
    ax2.legend()
    ax2.set_title("Wy")

    ax3 = plt.subplot(313)
    ax3.plot(time, min_jerk_angle_velocity[:, 2], 'g', label="Wz - minimum jerk")
    # ax3.plot(time, impedance_orientation[:, 2], 'b--', label="orientation 3rd element - impedance")
    ax3.plot(time, real_angle_velocity[:, 2], 'r--', label="Wz - real")
    ax3.legend()
    ax3.set_title("Wz")

    plt.tight_layout()
    # -------------------------------------------------------------
    plt.figure()

    ax1 = plt.subplot(311)
    ax1.plot(time, interaction_forces[:, 0], 'r--')
    ax1.set_title("Fx [N]")

    ax2 = plt.subplot(312)
    ax2.plot(time, interaction_forces[:, 1], 'r--')
    ax2.set_title("Fy [N]")

    ax2 = plt.subplot(313)
    ax2.plot(time, interaction_forces[:, 2], 'r--')
    ax2.set_title("Fz [N]")

    plt.tight_layout()
    # -------------------------------------------------------------
    plt.figure()

    ax1 = plt.subplot(311)
    ax1.plot(time, interaction_forces[:, 3], 'r--')
    ax1.set_title("Mx [Nm]")

    ax2 = plt.subplot(312)
    ax2.plot(time, interaction_forces[:, 4], 'r--')
    ax2.set_title("My [Nm]")

    ax2 = plt.subplot(313)
    ax2.plot(time, interaction_forces[:, 5], 'r--')
    ax2.set_title("Mz [Nm]")

    plt.tight_layout()
    plt.show()

    contact_time = float(info.get("contact_time"))
    contact_index = np.where(np.array(time) == contact_time)
    contact_index = int(contact_index[0])
    time = time[contact_index:]
    t = np.array(time)
    K = float(info.get("K_imp"))
    M = float(info.get("M_imp"))
    C = float(info.get("C_imp"))
    # F = interaction_forces[contact_index:, 4]
    # F = -interaction_forces[contact_index:, 0]
    F = -new_torque[:, 1]
    K = 0.1*K

    # y0 = [min_jerk_orientation[contact_index, 1] - impedance_orientation[contact_index, 1]
    #     , min_jerk_angle_velocity[contact_index, 1] - impedance_angle_velocity[contact_index, 1]]
    # y0 = [min_jerk_position[contact_index, 0] - impedance_position[contact_index, 0]
    #     , min_jerk_velocity[contact_index, 0] - impedance_velocity[contact_index, 0]]
    y0 = [min_jerk_orientation[contact_index, 1] - impedance_orientation[contact_index, 1]
        , min_jerk_orientation_dot[contact_index, 1] - impedance_angle_velocity[contact_index, 1]]


    sol = odeint(pend, y0, t, args=(F, time, K, C, M))
    plt.figure()
    plt.plot(t, sol[:, 0], 'b', label='pos from ODE')
    plt.plot(time, min_jerk_orientation[contact_index:, 1] - impedance_orientation[contact_index:, 1], 'g',
             label='from simulation')
    # plt.plot(time, min_jerk_position[contact_index:, 0] - impedance_position[contact_index:, 0], 'g',
    #          label='from simulation')
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.grid()

    plt.figure()
    plt.plot(t, sol[:, 1], 'b', label='vel from ODE')
    plt.plot(time, min_jerk_orientation_dot[contact_index:, 1] - impedance_angle_velocity[contact_index:, 1], 'g',
             label='from simulation')
    # plt.plot(time, min_jerk_velocity[contact_index:, 0] - impedance_velocity[contact_index:, 0], 'g',
    #          label='from simulation')
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.grid()
    plt.show()

def pend(y, t, F, time, K, C, M):
    x, x_dot = y
    f_interp = interp1d(time, F, axis=0, fill_value="extrapolate")
    f = f_interp(t)
    dydt = [x_dot, -K / M * x - C / M * x_dot + f / M]
    return dydt


#
# def F(t):


if __name__ == "__main__":
    simulation_total_time = 3  # sec
    control_freq = 10
    horizon = control_freq*simulation_total_time

    control_param = dict(type='JOINT_TORQUE', input_max=1, input_min=-1,
                         output_max=[0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
                         output_min=[-0.05, -0.05, -0.05, -0.5, -0.5, -0.5], kp=700, damping_ratio=np.sqrt(2),
                         impedance_mode='fixed', kp_limits=[0, 100000], damping_ratio_limits=[0, 10],
                         position_limits=None,
                         orientation_limits=None, uncouple_pos_ori=True, control_delta=True, interpolation=None,
                         ramp_ratio=0.2,
                         control_dim=6,
                         plotting=True,
                         collect_data=True,
                         simulation_total_time=simulation_total_time)

    env = GymWrapper(
        suite.make(
            "PegInHole",
            # "Lift",
            robots="UR5e",  # use UR5e robot
            # user_init_qpos=[-0.470, -1.735, 2.480, -2.275, -1.590, -1.991],# EC - set initial position of the joints
            user_init_qpos=[-0.09525232, - 0.83036537, 1.21126885, - 1.95171192, - 1.57079775, - 1.6660471],
            # EC - set initial position of the joints
            use_camera_obs=False,  # do not use pixel observations
            has_offscreen_renderer=False,  # not needed since not using pixel obs
            # has_renderer=True,  ##True  # make sure we can render to the screen
            has_renderer=False,  ##True  # make sure we can render to the screen
            reward_shaping=True,  # use dense rewards
            ignore_done=False,
            horizon=horizon,
            control_freq=control_freq,
            controller_configs=control_param,
            Peg_density=1,

        )
    )
    env.reset()

    done = False
    action = np.random.randn(env.robots[0].action_dim)  # sample random action - it doesn't matter

    while not done:
        obs, reward, done, info = env.step(action)  # take action in the environment
        # env.render()  # render on display

    # EC -  get all path information
    info = env.get_path_info()

    # Wrap this environment in a visualization wrapper
    env = VisualizationWrapper(env, indicator_configs=None)

    Git = 1
    # plot graphs
    # plotter(info)
