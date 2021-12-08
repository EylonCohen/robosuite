import numpy as np
from numpy import copy
from scipy.linalg import expm

import robosuite.utils.transform_utils as trans
from robosuite.utils.numba import jit_decorator


@jit_decorator
def nullspace_torques(mass_matrix, nullspace_matrix, initial_joint, joint_pos, joint_vel, joint_kp=10):
    """
    For a robot with redundant DOF(s), a nullspace exists which is orthogonal to the remainder of the controllable
    subspace of the robot's joints. Therefore, an additional secondary objective that does not impact the original
    controller objective may attempt to be maintained using these nullspace torques.

    This utility function specifically calculates nullspace torques that attempt to maintain a given robot joint
    positions @initial_joint with zero velocity using proportinal gain @joint_kp

    :Note: @mass_matrix, @nullspace_matrix, @joint_pos, and @joint_vel should reflect the robot's state at the current
    timestep

    Args:
        mass_matrix (np.array): 2d array representing the mass matrix of the robot
        nullspace_matrix (np.array): 2d array representing the nullspace matrix of the robot
        initial_joint (np.array): Joint configuration to be used for calculating nullspace torques
        joint_pos (np.array): Current joint positions
        joint_vel (np.array): Current joint velocities
        joint_kp (float): Proportional control gain when calculating nullspace torques

    Returns:
          np.array: nullspace torques
    """

    # kv calculated below corresponds to critical damping
    joint_kv = np.sqrt(joint_kp) * 2

    # calculate desired torques based on gains and error
    pose_torques = np.dot(mass_matrix, (joint_kp * (
            initial_joint - joint_pos) - joint_kv * joint_vel))

    # map desired torques to null subspace within joint torque actuator space
    nullspace_torques = np.dot(nullspace_matrix.transpose(), pose_torques)
    return nullspace_torques


@jit_decorator
def opspace_matrices(mass_matrix, J_full, J_pos, J_ori):
    """
    Calculates the relevant matrices used in the operational space control algorithm

    Args:
        mass_matrix (np.array): 2d array representing the mass matrix of the robot
        J_full (np.array): 2d array representing the full Jacobian matrix of the robot
        J_pos (np.array): 2d array representing the position components of the Jacobian matrix of the robot
        J_ori (np.array): 2d array representing the orientation components of the Jacobian matrix of the robot

    Returns:
        4-tuple:

            - (np.array): full lambda matrix (as 2d array)
            - (np.array): position components of lambda matrix (as 2d array)
            - (np.array): orientation components of lambda matrix (as 2d array)
            - (np.array): nullspace matrix (as 2d array)
    """
    mass_matrix_inv = np.linalg.inv(mass_matrix)

    # J M^-1 J^T
    lambda_full_inv = np.dot(
        np.dot(J_full, mass_matrix_inv),
        J_full.transpose())

    # Jx M^-1 Jx^T
    lambda_pos_inv = np.dot(
        np.dot(J_pos, mass_matrix_inv),
        J_pos.transpose())

    # Jr M^-1 Jr^T
    lambda_ori_inv = np.dot(
        np.dot(J_ori, mass_matrix_inv),
        J_ori.transpose())

    # take the inverses, but zero out small singular values for stability
    lambda_full = np.linalg.pinv(lambda_full_inv)
    lambda_pos = np.linalg.pinv(lambda_pos_inv)
    lambda_ori = np.linalg.pinv(lambda_ori_inv)

    # nullspace
    Jbar = np.dot(mass_matrix_inv, J_full.transpose()).dot(lambda_full)
    nullspace_matrix = np.eye(J_full.shape[-1], J_full.shape[-1]) - np.dot(Jbar, J_full)

    return lambda_full, lambda_pos, lambda_ori, nullspace_matrix


@jit_decorator
def orientation_error(desired, current):
    """
    This function calculates a 3-dimensional orientation error vector for use in the
    impedance controller. It does this by computing the delta rotation between the
    inputs and converting that rotation to exponential coordinates (axis-angle
    representation, where the 3d vector is axis * angle).
    See https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation for more information.
    Optimized function to determine orientation error from matrices

    Args:
        desired (np.array): 2d array representing target orientation matrix
        current (np.array): 2d array representing current orientation matrix

    Returns:
        np.array: 2d array representing orientation error as a matrix
    """
    rc1 = current[0:3, 0]
    rc2 = current[0:3, 1]
    rc3 = current[0:3, 2]
    rd1 = desired[0:3, 0]
    rd2 = desired[0:3, 1]
    rd3 = desired[0:3, 2]

    error = 0.5 * (np.cross(rc1, rd1) + np.cross(rc2, rd2) + np.cross(rc3, rd3))

    return error


def set_goal_position(delta,
                      current_position,
                      position_limit=None,
                      set_pos=None):
    """
    Calculates and returns the desired goal position, clipping the result accordingly to @position_limits.
    @delta and @current_position must be specified if a relative goal is requested, else @set_pos must be
    specified to define a global goal position

    Args:
        delta (np.array): Desired relative change in position
        current_position (np.array): Current position
        position_limit (None or np.array): 2d array defining the (min, max) limits of permissible position goal commands
        set_pos (None or np.array): If set, will ignore @delta and set the goal position to this value

    Returns:
        np.array: calculated goal position in absolute coordinates

    Raises:
        ValueError: [Invalid position_limit shape]
    """
    n = len(current_position)
    if set_pos is not None:
        goal_position = set_pos
    else:
        goal_position = current_position + delta

    if position_limit is not None:
        if position_limit.shape != (2, n):
            raise ValueError("Position limit should be shaped (2,{}) "
                             "but is instead: {}".format(n, position_limit.shape))

        # Clip goal position
        goal_position = np.clip(goal_position, position_limit[0], position_limit[1])

    return goal_position


def set_goal_orientation(delta,
                         current_orientation,
                         orientation_limit=None,
                         set_ori=None):
    """
    Calculates and returns the desired goal orientation, clipping the result accordingly to @orientation_limits.
    @delta and @current_orientation must be specified if a relative goal is requested, else @set_ori must be
    an orientation matrix specified to define a global orientation

    Args:
        delta (np.array): Desired relative change in orientation, in axis-angle form [ax, ay, az]
        current_orientation (np.array): Current orientation, in rotation matrix form
        orientation_limit (None or np.array): 2d array defining the (min, max) limits of permissible orientation goal commands
        set_ori (None or np.array): If set, will ignore @delta and set the goal orientation to this value

    Returns:
        np.array: calculated goal orientation in absolute coordinates

    Raises:
        ValueError: [Invalid orientation_limit shape]
    """
    # directly set orientation
    if set_ori is not None:
        goal_orientation = set_ori

    # otherwise use delta to set goal orientation
    else:
        # convert axis-angle value to rotation matrix
        quat_error = trans.axisangle2quat(delta)
        rotation_mat_error = trans.quat2mat(quat_error)
        goal_orientation = np.dot(rotation_mat_error, current_orientation)

    # check for orientation limits
    if np.array(orientation_limit).any():
        if orientation_limit.shape != (2, 3):
            raise ValueError("Orientation limit should be shaped (2,3) "
                             "but is instead: {}".format(orientation_limit.shape))

        # Convert to euler angles for clipping
        euler = trans.mat2euler(goal_orientation)

        # Clip euler angles according to specified limits
        limited = False
        for idx in range(3):
            if orientation_limit[0][idx] < orientation_limit[1][idx]:  # Normal angle sector meaning
                if orientation_limit[0][idx] < euler[idx] < orientation_limit[1][idx]:
                    continue
                else:
                    limited = True
                    dist_to_lower = euler[idx] - orientation_limit[0][idx]
                    if dist_to_lower > np.pi:
                        dist_to_lower -= 2 * np.pi
                    elif dist_to_lower < -np.pi:
                        dist_to_lower += 2 * np.pi

                    dist_to_higher = euler[idx] - orientation_limit[1][idx]
                    if dist_to_lower > np.pi:
                        dist_to_higher -= 2 * np.pi
                    elif dist_to_lower < -np.pi:
                        dist_to_higher += 2 * np.pi

                    if dist_to_lower < dist_to_higher:
                        euler[idx] = orientation_limit[0][idx]
                    else:
                        euler[idx] = orientation_limit[1][idx]
            else:  # Inverted angle sector meaning
                if (orientation_limit[0][idx] < euler[idx]
                        or euler[idx] < orientation_limit[1][idx]):
                    continue
                else:
                    limited = True
                    dist_to_lower = euler[idx] - orientation_limit[0][idx]
                    if dist_to_lower > np.pi:
                        dist_to_lower -= 2 * np.pi
                    elif dist_to_lower < -np.pi:
                        dist_to_lower += 2 * np.pi

                    dist_to_higher = euler[idx] - orientation_limit[1][idx]
                    if dist_to_lower > np.pi:
                        dist_to_higher -= 2 * np.pi
                    elif dist_to_lower < -np.pi:
                        dist_to_higher += 2 * np.pi

                    if dist_to_lower < dist_to_higher:
                        euler[idx] = orientation_limit[0][idx]
                    else:
                        euler[idx] = orientation_limit[1][idx]
        if limited:
            goal_orientation = trans.euler2mat(np.array([euler[0], euler[1], euler[2]]))
    return goal_orientation


def build_imp_matrices_circular_peg(action):
    # implementation of the structure of K for circular peg
    # original case
    # K1, K2, K3, K4, K5, K6 = action[0], action[1], action[2], action[3], action[4], action[5]
    # M1, M2, M3, M4, M5, M6 = action[6], action[7], action[8], action[9], action[10], action[11]
    # K_imp_inv = np.array([[K1, 0, 0, 0, -K5, 0],
    #                       [0, K1, 0, K5, 0, 0],
    #                       [0, 0, K2, 0, 0, 0],
    #                       [0, -K6, 0, K3, 0, 0],
    #                       [K6, 0, 0, 0, K3, 0],
    #                       [0, 0, 0, 0, 0, K4]])

    K1, K2, K3, K4, K5, K6, K7, K8, K9, K10 = action[0], action[1], action[2], action[3], \
                                              action[4], action[5], action[6], action[7], action[8], action[9]
    M1, M2, M3, M4, M5, M6 = action[10], action[11], action[12], action[13], action[14], action[15]

    K_imp_inv = np.array([[K1, 0, 0, 0, K8, 0],
                          [0, K2, 0, K7, 0, 0],
                          [0, 0, K3, 0, 0, 0],
                          [0, K9, 0, K4, 0, 0],
                          [K10, 0, 0, 0, K5, 0],
                          [0, 0, 0, 0, 0, K6]])

    if np.linalg.det(K_imp_inv) == 0:  # k_inv is singular
        # this is for making sure A has positive eig and the inverse of K_inv would not be calculated
        A = np.identity(2)
        return 0, 0, 0, A, 0, 0

    K_imp = np.linalg.inv(K_imp_inv)
    M_imp = np.array([[M1, 0, 0, 0, M5, 0],
                      [0, M1, 0, M5, 0, 0],
                      [0, 0, M2, 0, 0, 0],
                      [0, M6, 0, M3, 0, 0],
                      [M6, 0, 0, 0, M3, 0],
                      [0, 0, 0, 0, 0, M4]])

    zeta = 0.707
    C_multiplier = 1
    # Wn = sqrt(k/m) , C = 2 * zeta * Wn
    C_imp = copy(K_imp)  # just for initialization
    for idx, x in np.ndenumerate(K_imp):
        if M_imp[idx] != 0:
            Wn = np.sqrt(np.abs(x) / M_imp[idx])
            C_imp[idx] = C_multiplier * 2 * zeta * np.sign(x) * Wn
        else:
            C_imp[idx] = 0
        M_imp[idx] = np.sign(x) * M_imp[idx]  # make M values sign the same as K values

    # K1, K2, K3, K4, K5, K6 = action[0], action[1], action[2], action[3], action[4], action[5]
    # M1, M2, M3, M4, M5, M6 = action[6], action[7], action[8], action[9], action[10], action[11]
    # C1, C2, C3, C4, C5, C6 = action[12], action[13], action[14], action[15], action[16], action[17]
    #
    # # Weight mat param
    # W1, W2, W4, W5, W7, W8, W9, \
    # W10 = action[18], action[19], action[20], action[21], \
    #       action[22], action[23], action[24], action[25]
    # W3, W6 = 1.0, 1.0
    # W = np.array([[W1, 0, 0, 0, W8, 0],
    #               [0, W2, 0, W7, 0, 0],
    #               [0, 0, W3, 0, 0, 0],
    #               [0, W9, 0, W4, 0, 0],
    #               [W10, 0, 0, 0, W5, 0],
    #               [0, 0, 0, 0, 0, W6]])
    # if np.linalg.det(W) == 0:  # W_inv is singular
    #     # this is for making sure A has positive eig and the inverse of K_inv would not be calculated
    #     A = np.identity(2)
    #     return 0, 0, 0, A, 0, 0
    #
    # W_inv = np.linalg.inv(W)
    # K_imp = W_inv @ np.array([[K1, 0, 0, 0, 0, 0],
    #                           [0, K2, 0, 0, 0, 0],
    #                           [0, 0, K3, 0, 0, 0],
    #                           [0, 0, 0, K4, 0, 0],
    #                           [0, 0, 0, 0, K5, 0],
    #                           [0, 0, 0, 0, 0, K6]])
    # M_imp = W_inv @ np.array([[M1, 0, 0, 0, 0, 0],
    #                           [0, M2, 0, 0, 0, 0],
    #                           [0, 0, M3, 0, 0, 0],
    #                           [0, 0, 0, M4, 0, 0],
    #                           [0, 0, 0, 0, M5, 0],
    #                           [0, 0, 0, 0, 0, M6]])
    # C_imp = W_inv @ np.array([[C1, 0, 0, 0, 0, 0],
    #                           [0, C2, 0, 0, 0, 0],
    #                           [0, 0, C3, 0, 0, 0],
    #                           [0, 0, 0, C4, 0, 0],
    #                           [0, 0, 0, 0, C5, 0],
    #                           [0, 0, 0, 0, 0, C6]])

    M_inv = np.linalg.pinv(M_imp)
    A_1 = np.concatenate((np.zeros([6, 6], dtype=int), np.identity(6)), axis=1)
    A_2 = np.concatenate((np.dot(-M_inv, K_imp), np.dot(-M_inv, C_imp)), axis=1)
    A = np.concatenate((A_1, A_2), axis=0)

    B_1 = np.zeros([6, 6], dtype=int)
    B_2 = M_inv
    B = np.concatenate((B_1, B_2), axis=0)

    # discrete state space A, B matrices interaction_forces
    Delta_T = 0.002
    A_d = expm(A * Delta_T)
    B_d = np.dot(np.dot(np.linalg.pinv(A), (A_d - np.identity(A_d.shape[0]))), B)
    return K_imp, C_imp, M_imp, A, A_d, B_d


def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


def is_stable_system(x):
    eig = np.linalg.eigvals(x)
    real_eig = np.real(eig)
    is_stable_system_bool = np.all(real_eig < 0)
    return is_stable_system_bool
