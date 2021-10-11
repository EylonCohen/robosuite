import abc
from collections.abc import Iterable
from copy import deepcopy

import numpy as np
import mujoco_py
from scipy.linalg import expm

import robosuite.utils.macros as macros
import robosuite.utils.angle_transformation as at
from robosuite.utils.control_utils import *
import robosuite.utils.transform_utils as T
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.interpolate import interp1d


class Controller(object, metaclass=abc.ABCMeta):
    """
    General controller interface.

    Requires reference to mujoco sim object, eef_name of specific robot, relevant joint_indexes to that robot, and
    whether an initial_joint is used for nullspace torques or not

    Args:
        sim (MjSim): Simulator instance this controller will pull robot state updates from

        eef_name (str): Name of controlled robot arm's end effector (from robot XML)

        joint_indexes (dict): Each key contains sim reference indexes to relevant robot joint information, namely:

            :`'joints'`: list of indexes to relevant robot joints
            :`'qpos'`: list of indexes to relevant robot joint positions
            :`'qvel'`: list of indexes to relevant robot joint velocities

        actuator_range (2-tuple of array of float): 2-Tuple (low, high) representing the robot joint actuator range
    """

    def __init__(self,
                 sim,
                 eef_name,
                 joint_indexes,
                 actuator_range,
                 plotting,
                 collect_data,
                 simulation_total_time,
                 ):

        # Actuator range
        self.actuator_min = actuator_range[0]
        self.actuator_max = actuator_range[1]

        # Attributes for scaling / clipping inputs to outputs
        self.action_scale = None
        self.action_input_transform = None
        self.action_output_transform = None

        # Private property attributes
        self.control_dim = None
        self.output_min = None
        self.output_max = None
        self.input_min = None
        self.input_max = None

        # mujoco simulator state
        self.sim = sim
        self.model_timestep = macros.SIMULATION_TIMESTEP
        self.eef_name = eef_name
        self.joint_index = joint_indexes["joints"]
        self.qpos_index = joint_indexes["qpos"]
        self.qvel_index = joint_indexes["qvel"]

        # robot states
        self.ee_pos = None
        self.ee_ori_mat = None
        self.ee_pos_vel = None
        self.ee_ori_vel = None
        self.joint_pos = None
        self.joint_vel = None

        # dynamics and kinematics
        self.J_pos = None
        self.J_ori = None
        self.J_full = None
        self.mass_matrix = None

        self.interaction_forces = None
        self.interaction_forces_vec = []
        self.PD_force_command = []
        self.desired_frame_FT_vec = []
        self.desired_frame_imp_position_vec = []
        self.desired_frame_imp_ori_vec = []
        self.desired_frame_imp_vel_vec = []
        self.desired_frame_imp_ang_vel_vec = []

        # Joint dimension
        self.joint_dim = len(joint_indexes["joints"])

        # Torques being outputted by the controller
        self.torques = None

        # Update flag to prevent redundant update calls
        self.new_update = True

        # Move forward one timestep to propagate updates before taking first update
        self.sim.forward()

        # Initialize controller by updating internal state and setting the initial joint, pos, and ori
        self.update()
        self.initial_joint = self.joint_pos
        self.initial_ee_pos = self.ee_pos
        self.initial_ee_ori_mat = self.ee_ori_mat

        self.simulation_total_time = simulation_total_time  # from main
        # EC - define the duration of each trajectory part by the
        self.trajectory_duration_ratio = 0.5

        # self.hole_middle_cylinder = np.array(self.sim.data.site_xpos[self.sim.model.site_name2id("hole_middle_cylinder")])
        # EC - when the box origin is at (0,0) the nominal_hole_middle_cylinder is at [0.2, 0.062, 0.8]
        self.nominal_hole_middle_cylinder = np.array([0.2, 0.062, 0.8])
        self.peg_edge = np.array(self.sim.data.site_xpos[self.sim.model.site_name2id("peg_site")])
        self.delta_z = 0.005  # make sure not hitting tha table
        self.z_diff = self.nominal_hole_middle_cylinder[2] - self.peg_edge[2] + self.delta_z

        # EC - minimum jerk specification first trajectory_part
        self.first_trajectory_part_initial_position = self.initial_ee_pos
        self.first_trajectory_part_final_position = [self.nominal_hole_middle_cylinder[0],
                                                     self.nominal_hole_middle_cylinder[1],
                                                     self.first_trajectory_part_initial_position[2]
                                                     + self.z_diff + 7 * self.delta_z]
        self.first_trajectory_part_initial_orientation = self.initial_ee_ori_mat
        self.first_trajectory_part_final_orientation = np.array([[1, 0, 0],
                                                                 [0, -1, 0],
                                                                 [0, 0, -1]])  # peg vertical (pointing down)

        self.first_trajectory_part_euler_initial_orientation = R.from_matrix(
            self.first_trajectory_part_initial_orientation).as_euler('xyz', degrees=False)
        self.first_trajectory_part_euler_final_orientation = R.from_matrix(
            self.first_trajectory_part_final_orientation).as_euler('xyz', degrees=False)
        indexes_for_correction = np.abs(
            self.first_trajectory_part_euler_final_orientation - self.first_trajectory_part_euler_initial_orientation) > np.pi
        correction = np.sign(self.first_trajectory_part_euler_final_orientation) * (2 * np.pi) * indexes_for_correction
        self.first_trajectory_part_euler_final_orientation = self.first_trajectory_part_euler_final_orientation - correction
        self.first_trajectory_part_final_time = self.simulation_total_time * self.trajectory_duration_ratio

        # EC - minimum jerk specification second trajectory_part
        self.initiate_trajectory = True
        self.second_trajectory_part_initial_position = None
        self.second_trajectory_part_final_position = None
        self.second_trajectory_part_initial_orientation = None
        self.second_trajectory_part_final_orientation = np.array([[1, 0, 0],
                                                                  [0, -1, 0],
                                                                  [0, 0, -1]])  # peg vertical (pointing down)
        self.second_trajectory_part_euler_initial_orientation = None
        self.second_trajectory_part_euler_final_orientation = None
        self.second_trajectory_part_euler_final_orientation = None
        self.second_trajectory_part_final_time = self.simulation_total_time

        # define a reference distance for deciding if the robot is not stable
        self.nominal_stability_distance = None

        # EC - Run further definition and class variables
        self._specify_constants()

        # EC - this is the vector for the impedance equations
        self.X_m = np.zeros((12, 1))
        self.is_contact = False  # becomes true when the peg hits the table
        self.contact_time = 0.0
        self.first_contact = True
        self.Delta_T = 0.002
        self.f_0 = np.array([0, 0, 0, 0, 0, 0])

        # EC - this is the default values
        self.K = 5000
        self.M = 5
        Wn = np.sqrt(self.K / self.M)
        zeta = 0.707
        # zeta = 1
        self.C = 2 * self.M * zeta * Wn
        # C = 0

        self.K_imp = self.K * np.array([[1, 0, 0, 0, 0, 0],
                                        [0, 1, 0, 0, 0, 0],
                                        [0, 0, 1, 0, 0, 0],
                                        [0, 0, 0, 1, 0, 0],
                                        [0, 0, 0, 0, 1, 0],
                                        [0, 0, 0, 0, 0, 1]])
        self.C_imp = self.C * np.array([[1, 0, 0, 0, 0, 0],
                                        [0, 1, 0, 0, 0, 0],
                                        [0, 0, 1, 0, 0, 0],
                                        [0, 0, 0, 1, 0, 0],
                                        [0, 0, 0, 0, 1, 0],
                                        [0, 0, 0, 0, 0, 1]])
        self.M_imp = self.M * np.array([[1, 0, 0, 0, 0, 0],
                                        [0, 1, 0, 0, 0, 0],
                                        [0, 0, 1, 0, 0, 0],
                                        [0, 0, 0, 1, 0, 0],
                                        [0, 0, 0, 0, 1, 0],
                                        [0, 0, 0, 0, 0, 1]])

        # define if you want to plot some data
        self.collect_data = collect_data
        self.plotting = plotting

    def impedance_computations(self):
        # EC - compute next impedance Xm(n+1) and Vm(n+1) in world base frame.
        # state space formulation
        # X=[xm;thm;xm_d;thm_d] U=[F_int;M_int]
        M_inv = np.linalg.pinv(self.M_imp)
        A_1 = np.concatenate((np.zeros([6, 6], dtype=int), np.identity(6)), axis=1)
        A_2 = np.concatenate((np.dot(-M_inv, self.K_imp), np.dot(-M_inv, self.C_imp)), axis=1)
        A = np.concatenate((A_1, A_2), axis=0)

        B_1 = np.zeros([6, 6], dtype=int)
        B_2 = M_inv
        B = np.concatenate((B_1, B_2), axis=0)

        # discrete state space A, B matrices interaction_forces
        A_d = expm(A * self.Delta_T)
        B_d = np.dot(np.dot(np.linalg.pinv(A), (A_d - np.identity(A_d.shape[0]))), B)

        # convert the forces and torques to the desired frame
        Rotation_world_to_desired = R.from_euler("xyz", self.min_jerk_orientation, degrees=False).as_matrix()
        Rotation_desired_to_world = Rotation_world_to_desired.T
        F_d = Rotation_desired_to_world @ self.interaction_forces[:3]
        M_d = Rotation_desired_to_world @ self.interaction_forces[3:6]
        f_0 = np.concatenate((Rotation_desired_to_world @ self.f_0[:3],
                              Rotation_desired_to_world @ self.f_0[3:6]), axis=0)
        U = (f_0 + np.concatenate((F_d, M_d), axis=0)).reshape(6, 1)

        # only for graphs!
        if self.collect_data:
            self.desired_frame_FT_vec.append(np.array(U))
            self.desired_frame_imp_position_vec.append(np.array((self.X_m[:3]).reshape(3, )))
            self.desired_frame_imp_ori_vec.append(np.array((self.X_m[3:6]).reshape(3, )))
            self.desired_frame_imp_vel_vec.append(np.array((self.X_m[6:9]).reshape(3, )))
            self.desired_frame_imp_ang_vel_vec.append(np.array((self.X_m[9:12]).reshape(3, )))

        # discrete state solution X(k+1)=Ad*X(k)+Bd*U(k)
        X_m_next = np.dot(A_d, self.X_m.reshape(12, 1)) + np.dot(B_d, U)

        self.X_m = deepcopy(X_m_next)

        return

    def _min_jerk(self):
        """
        EC
        Compute the value of position velocity and acceleration in a minimum jerk trajectory

        """
        if self.time < self.first_trajectory_part_final_time:  # first trajectory part
            X_init, Y_init, Z_init = self.first_trajectory_part_initial_position
            X_final, Y_final, Z_final = self.first_trajectory_part_final_position
            euler_initial_orientation = self.first_trajectory_part_euler_initial_orientation
            euler_final_orientation = self.first_trajectory_part_euler_final_orientation
            tfinal = self.first_trajectory_part_final_time
            t = self.time
        else:  # second trajectory part
            if self.initiate_trajectory:
                self.initiate_trajectory = False
                self.nominal_stability_distance = np.linalg.norm(self.peg_edge -
                                        np.array(self.sim.data.site_xpos[
                                                     self.sim.model.site_name2id("hole_middle_cylinder")]))
                self.z_diff = self.nominal_hole_middle_cylinder[2] - self.peg_edge[2] + self.delta_z
                self.second_trajectory_part_initial_position = self.ee_pos
                self.second_trajectory_part_final_position = [self.nominal_hole_middle_cylinder[0],
                                                              self.nominal_hole_middle_cylinder[1],
                                                              self.second_trajectory_part_initial_position[
                                                                  2] + self.z_diff]
                self.second_trajectory_part_initial_orientation = self.ee_ori_mat
                self.second_trajectory_part_euler_initial_orientation = R.from_matrix(
                    self.second_trajectory_part_initial_orientation).as_euler('xyz', degrees=False)
                self.second_trajectory_part_euler_final_orientation = R.from_matrix(
                    self.second_trajectory_part_final_orientation).as_euler('xyz', degrees=False)
                indexes_for_correction = np.abs(
                    self.second_trajectory_part_euler_final_orientation - self.second_trajectory_part_euler_initial_orientation) > np.pi
                correction = np.sign(self.second_trajectory_part_euler_final_orientation) * (
                            2 * np.pi) * indexes_for_correction
                self.second_trajectory_part_euler_final_orientation = self.second_trajectory_part_euler_final_orientation - correction

            X_init, Y_init, Z_init = self.second_trajectory_part_initial_position
            X_final, Y_final, Z_final = self.second_trajectory_part_final_position
            euler_initial_orientation = self.second_trajectory_part_euler_initial_orientation
            euler_final_orientation = self.second_trajectory_part_euler_final_orientation
            tfinal = self.simulation_total_time - self.first_trajectory_part_final_time
            t = self.time - self.first_trajectory_part_final_time

        x_traj = (X_final - X_init) / (tfinal ** 3) * (
                6 * (t ** 5) / (tfinal ** 2) - 15 * (t ** 4) / tfinal + 10 * (t ** 3)) + X_init
        y_traj = (Y_final - Y_init) / (tfinal ** 3) * (
                6 * (t ** 5) / (tfinal ** 2) - 15 * (t ** 4) / tfinal + 10 * (t ** 3)) + Y_init
        z_traj = (Z_final - Z_init) / (tfinal ** 3) * (
                6 * (t ** 5) / (tfinal ** 2) - 15 * (t ** 4) / tfinal + 10 * (t ** 3)) + Z_init
        self.min_jerk_position = np.array([x_traj, y_traj, z_traj])

        # velocities
        vx = (X_final - X_init) / (tfinal ** 3) * (
                30 * (t ** 4) / (tfinal ** 2) - 60 * (t ** 3) / tfinal + 30 * (t ** 2))
        vy = (Y_final - Y_init) / (tfinal ** 3) * (
                30 * (t ** 4) / (tfinal ** 2) - 60 * (t ** 3) / tfinal + 30 * (t ** 2))
        vz = (Z_final - Z_init) / (tfinal ** 3) * (
                30 * (t ** 4) / (tfinal ** 2) - 60 * (t ** 3) / tfinal + 30 * (t ** 2))
        self.min_jerk_velocity = np.array([vx, vy, vz])

        # acceleration
        ax = (X_final - X_init) / (tfinal ** 3) * (
                120 * (t ** 3) / (tfinal ** 2) - 180 * (t ** 2) / tfinal + 60 * t)
        ay = (Y_final - Y_init) / (tfinal ** 3) * (
                120 * (t ** 3) / (tfinal ** 2) - 180 * (t ** 2) / tfinal + 60 * t)
        az = (Z_final - Z_init) / (tfinal ** 3) * (
                120 * (t ** 3) / (tfinal ** 2) - 180 * (t ** 2) / tfinal + 60 * t)
        self.min_jerk_acceleration = np.array([ax, ay, az])

        # euler xyz representation
        alfa = (euler_final_orientation[0] - euler_initial_orientation[0]) / (tfinal ** 3) * (
                6 * (t ** 5) / (tfinal ** 2) - 15 * (t ** 4) / tfinal + 10 * (t ** 3)) + \
               euler_initial_orientation[0]
        beta = (euler_final_orientation[1] - euler_initial_orientation[1]) / (tfinal ** 3) * (
                6 * (t ** 5) / (tfinal ** 2) - 15 * (t ** 4) / tfinal + 10 * (t ** 3)) + \
               euler_initial_orientation[1]
        gamma = (euler_final_orientation[2] - euler_initial_orientation[2]) / (tfinal ** 3) * (
                6 * (t ** 5) / (tfinal ** 2) - 15 * (t ** 4) / tfinal + 10 * (t ** 3)) + \
                euler_initial_orientation[2]

        alfa_dot = (euler_final_orientation[0] - euler_initial_orientation[0]) / (tfinal ** 3) * (
                30 * (t ** 4) / (tfinal ** 2) - 60 * (t ** 3) / tfinal + 30 * (t ** 2))
        beta_dot = (euler_final_orientation[1] - euler_initial_orientation[1]) / (tfinal ** 3) * (
                30 * (t ** 4) / (tfinal ** 2) - 60 * (t ** 3) / tfinal + 30 * (t ** 2))
        gamma_dot = (euler_final_orientation[2] - euler_initial_orientation[2]) / (tfinal ** 3) * (
                30 * (t ** 4) / (tfinal ** 2) - 60 * (t ** 3) / tfinal + 30 * (t ** 2))

        self.min_jerk_orientation = np.array([alfa, beta, gamma])
        self.min_jerk_orientation_dot = np.array([alfa_dot, beta_dot, gamma_dot])
        R_world_to_body = R.from_euler('xyz', self.min_jerk_orientation, degrees=False).as_matrix()
        # w = T*V  -- the angular velocity
        self.min_jerk_ang_vel = R_world_to_body @ (T.T_mat(self.min_jerk_orientation) @
                                                   self.min_jerk_orientation_dot.T)

        return

    def _specify_constants(self):
        """
        EC
        Assign constants in class variables

        """

        self.min_jerk_position = None
        self.min_jerk_velocity = None
        self.min_jerk_acceleration = None
        self.min_jerk_orientation = None
        self.min_jerk_orientation_dot = None
        self.min_jerk_ang_vel = None
        self.min_jerk_ang_acc = None

        self.min_jerk_position_vec = []
        self.min_jerk_velocity_vec = []
        self.min_jerk_acceleration_vec = []
        self.min_jerk_orientation_vec = []
        self.min_jerk_orientation_dot_vec = []
        self.min_jerk_angle_velocity_vec = []

        self.tfinal = 2  # this is for the minimum jerk
        self.time = 0.0
        self.time_vec = []
        self.real_position = None
        self.real_velocity = None
        self.real_orientation = None
        self.real_angle_velocity = None

        self.real_position_vec = []
        self.real_velocity_vec = []
        self.real_orientation_vec = []
        self.real_angle_velocity_vec = []

        self.impedance_orientation = []

        self.impedance_position_vec = []
        self.impedance_velocity_vec = []
        self.impedance_acceleration_vec = []
        self.impedance_orientation_vec = []
        self.impedance_angle_velocity_vec = []

    @abc.abstractmethod
    def run_controller(self):
        """
        Abstract method that should be implemented in all subclass controllers, and should convert a given action
        into torques (pre gravity compensation) to be executed on the robot.
        Additionally, resets the self.new_update flag so that the next self.update call will occur
        """
        self.new_update = True

    def scale_action(self, action):
        """
        Clips @action to be within self.input_min and self.input_max, and then re-scale the values to be within
        the range self.output_min and self.output_max

        Args:
            action (Iterable): Actions to scale

        Returns:
            np.array: Re-scaled action
        """

        if self.action_scale is None:
            self.action_scale = abs(self.output_max - self.output_min) / abs(self.input_max - self.input_min)
            self.action_output_transform = (self.output_max + self.output_min) / 2.0
            self.action_input_transform = (self.input_max + self.input_min) / 2.0
        action = np.clip(action, self.input_min, self.input_max)
        transformed_action = (action - self.action_input_transform) * self.action_scale + self.action_output_transform

        return transformed_action

    def update(self, force=False):
        """
        Updates the state of the robot arm, including end effector pose / orientation / velocity, joint pos/vel,
        jacobian, and mass matrix. By default, since this is a non-negligible computation, multiple redundant calls
        will be ignored via the self.new_update attribute flag. However, if the @force flag is set, the update will
        occur regardless of that state of self.new_update. This base class method of @run_controller resets the
        self.new_update flag

        Args:
            force (bool): Whether to force an update to occur or not
        """

        # Only run update if self.new_update or force flag is set
        # if self.new_update or force:
        self.sim.forward()

        self.time = self.sim.data.time
        self.peg_edge = np.array(self.sim.data.site_xpos[self.sim.model.site_name2id("peg_site")])
        self.ee_pos = np.array(self.sim.data.site_xpos[self.sim.model.site_name2id(self.eef_name)])
        self.ee_ori_mat = np.array(self.sim.data.site_xmat[self.sim.model.site_name2id(self.eef_name)].reshape([3, 3]))
        self.ee_pos_vel = np.array(self.sim.data.site_xvelp[self.sim.model.site_name2id(self.eef_name)])
        self.ee_ori_vel = np.array(self.sim.data.site_xvelr[self.sim.model.site_name2id(self.eef_name)])

        self.joint_pos = np.array(self.sim.data.qpos[self.qpos_index])
        self.joint_vel = np.array(self.sim.data.qvel[self.qvel_index])

        self.J_pos = np.array(self.sim.data.get_site_jacp(self.eef_name).reshape((3, -1))[:, self.qvel_index])
        self.J_ori = np.array(self.sim.data.get_site_jacr(self.eef_name).reshape((3, -1))[:, self.qvel_index])
        self.J_full = np.array(np.vstack([self.J_pos, self.J_ori]))

        mass_matrix = np.ndarray(shape=(len(self.sim.data.qvel) ** 2,), dtype=np.float64, order='C')
        mujoco_py.cymj._mj_fullM(self.sim.model, mass_matrix, self.sim.data.qM)
        mass_matrix = np.reshape(mass_matrix, (len(self.sim.data.qvel), len(self.sim.data.qvel)))
        self.mass_matrix = mass_matrix[self.qvel_index, :][:, self.qvel_index]

        # EC - force readings
        # the forces needs to be transform to the world base frame
        # the minus sign is because the measured forces are the forces that the robot apply on the environment
        forces_world = np.dot(self.ee_ori_mat, -self.sim.data.sensordata[:3])
        torques_world = np.dot(self.ee_ori_mat, -self.sim.data.sensordata[3:6])
        self.interaction_forces = np.concatenate((forces_world, torques_world), axis=0)

        # Clear self.new_update
        self.new_update = False

    def update_base_pose(self, base_pos, base_ori):
        """
        Optional function to implement in subclass controllers that will take in @base_pos and @base_ori and update
        internal configuration to account for changes in the respective states. Useful for controllers e.g. IK, which
        is based on pybullet and requires knowledge of simulator state deviations between pybullet and mujoco

        Args:
            base_pos (3-tuple): x,y,z position of robot base in mujoco world coordinates
            base_ori (4-tuple): x,y,z,w orientation or robot base in mujoco world coordinates
        """
        pass

    def update_initial_joints(self, initial_joints):
        """
        Updates the internal attribute self.initial_joints. This is useful for updating changes in controller-specific
        behavior, such as with OSC where self.initial_joints is used for determine nullspace actions

        This function can also be extended by subclassed controllers for additional controller-specific updates

        Args:
            initial_joints (Iterable): Array of joint position values to update the initial joints
        """
        self.initial_joint = np.array(initial_joints)
        self.update(force=True)
        self.initial_ee_pos = self.ee_pos
        self.initial_ee_ori_mat = self.ee_ori_mat

    def clip_torques(self, torques):
        """
        Clips the torques to be within the actuator limits

        Args:
            torques (Iterable): Torques to clip

        Returns:
            np.array: Clipped torques
        """
        return np.clip(torques, self.actuator_min, self.actuator_max)

    def reset_goal(self):
        """
        Resets the goal -- usually by setting to the goal to all zeros, but in some cases may be different (e.g.: OSC)
        """
        raise NotImplementedError

    @staticmethod
    def nums2array(nums, dim):
        """
        Convert input @nums into numpy array of length @dim. If @nums is a single number, broadcasts it to the
        corresponding dimension size @dim before converting into a numpy array

        Args:
            nums (numeric or Iterable): Either single value or array of numbers
            dim (int): Size of array to broadcast input to env.sim.data.actuator_force

        Returns:
            np.array: Array filled with values specified in @nums
        """
        # First run sanity check to make sure no strings are being inputted
        if isinstance(nums, str):
            raise TypeError("Error: Only numeric inputs are supported for this function, nums2array!")

        # Check if input is an Iterable, if so, we simply convert the input to np.array and return
        # Else, input is a single value, so we map to a numpy array of correct size and return
        return np.array(nums) if isinstance(nums, Iterable) else np.ones(dim) * nums

    @property
    def torque_compensation(self):
        """
        Gravity compensation for this robot arm

        Returns:
            np.array: torques
        """
        return self.sim.data.qfrc_bias[self.qvel_index]

    @property
    def actuator_limits(self):
        """
        Torque limits for this controller

        Returns:
            2-tuple:

                - (np.array) minimum actuator torques
                - (np.array) maximum actuator torques
        """
        return self.actuator_min, self.actuator_max

    @property
    def control_limits(self):
        """
        Limits over this controller's action space, which defaults to input min/max

        Returns:
            2-tuple:

                - (np.array) minimum action values
                - (np.array) maximum action values
        """
        return self.input_min, self.input_max

    @property
    def name(self):
        """
        Name of this controller

        Returns:
            str: controller name
        """
        raise NotImplementedError

    # EC
    def add_path_parameter(self):

        self.time_vec.append(self.time)

        self.min_jerk_position_vec.append(self.min_jerk_position)
        self.min_jerk_velocity_vec.append(self.min_jerk_velocity)
        self.min_jerk_acceleration_vec.append(self.min_jerk_acceleration)
        # self.min_jerk_orientation_vec.append(self.min_jerk_orientation)
        self.min_jerk_orientation_vec.append(R.from_euler("xyz", self.min_jerk_orientation, degrees=False).as_rotvec())
        self.min_jerk_orientation_dot_vec.append(self.min_jerk_orientation_dot)
        self.min_jerk_angle_velocity_vec.append(self.min_jerk_ang_vel)

        self.real_position_vec.append(self.real_position)
        self.real_velocity_vec.append(self.real_velocity)
        self.real_orientation_vec.append(self.real_orientation)
        self.real_angle_velocity_vec.append(self.real_angle_velocity)
        self.interaction_forces_vec.append(np.array(self.interaction_forces))

    def plotter(self):

        time = np.array(self.time_vec)
        min_jerk_position = np.array(self.min_jerk_position_vec)
        min_jerk_velocity = np.array(self.min_jerk_velocity_vec)
        min_jerk_acceleration = np.array(self.min_jerk_acceleration_vec)
        min_jerk_orientation = np.array(self.min_jerk_orientation_vec)
        min_jerk_angle_velocity = np.array(self.min_jerk_angle_velocity_vec)

        impedance_position = np.array(self.impedance_position_vec)
        impedance_velocity = np.array(self.impedance_velocity_vec)
        impedance_orientation = np.array(self.impedance_orientation_vec)
        impedance_angle_velocity = np.array(self.impedance_angle_velocity_vec)

        real_position = np.array(self.real_position_vec)
        real_velocity = np.array(self.real_velocity_vec)
        real_orientation = np.array(self.real_orientation_vec)
        real_angle_velocity = np.array(self.real_angle_velocity_vec)
        interaction_forces = np.array(self.interaction_forces_vec)
        PD_force_command = np.array(self.PD_force_command)

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
        ax1.plot(time, impedance_velocity[:, 0], 'b--', label="$\dot X$ impedance")
        ax1.plot(time, real_velocity[:, 0], 'r--', label=" $\dot X$ real")
        ax1.legend()
        ax1.set_title("$\dot X$ [m/s]")

        ax2 = plt.subplot(312)
        ax2.plot(time, min_jerk_velocity[:, 1], 'g', label=" $\dot Y$ minimum jerk")
        ax2.plot(time, impedance_velocity[:, 1], 'b--', label="$\dot X$ impedance")
        ax2.plot(time, real_velocity[:, 1], 'r--', label=" $\dot Y$ real")
        ax2.legend()
        ax2.set_title("$\dot Y$ [m/s]")

        ax3 = plt.subplot(313)
        ax3.plot(time, min_jerk_velocity[:, 2], 'g', label=" $\dot Z$ minimum jerk")
        ax3.plot(time, impedance_velocity[:, 2], 'b--', label="$\dot X$ impedance")
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
        ax1.plot(time, impedance_angle_velocity[:, 0], 'b--', label="orientation 1st element - impedance")
        ax1.plot(time, real_angle_velocity[:, 0], 'r--', label="Wx  - real")
        ax1.legend()
        ax1.set_title("Wx")

        ax2 = plt.subplot(312)
        ax2.plot(time, min_jerk_angle_velocity[:, 1], 'g', label="Wy - minimum jerk")
        ax2.plot(time, impedance_angle_velocity[:, 1], 'b--', label="orientation 2nd element - impedance")
        ax2.plot(time, real_angle_velocity[:, 1], 'r--', label="Wy - real")
        ax2.legend()
        ax2.set_title("Wy")

        ax3 = plt.subplot(313)
        ax3.plot(time, min_jerk_angle_velocity[:, 2], 'g', label="Wz - minimum jerk")
        ax3.plot(time, impedance_angle_velocity[:, 2], 'b--', label="orientation 3rd element - impedance")
        ax3.plot(time, real_angle_velocity[:, 2], 'r--', label="Wz - real")
        ax3.legend()
        ax3.set_title("Wz")

        plt.tight_layout()
        # -------------------------------------------------------------
        plt.figure()

        ax1 = plt.subplot(311)
        ax1.plot(time, interaction_forces[:, 0], 'r--', label="from sensor")
        ax1.plot(time, PD_force_command[:, 0], 'g--', label="from PD")
        ax1.legend()
        ax1.set_title("Fx [N]")

        ax2 = plt.subplot(312)
        ax2.plot(time, interaction_forces[:, 1], 'r--', label="from sensor")
        ax2.plot(time, PD_force_command[:, 1], 'g--', label="from PD")
        ax2.legend()
        ax2.set_title("Fy [N]")

        ax3 = plt.subplot(313)
        ax3.plot(time, interaction_forces[:, 1], 'r--', label="from sensor")
        ax3.plot(time, PD_force_command[:, 1], 'g--', label="from PD")
        ax3.legend()
        ax3.set_title("Fz [N]")

        plt.tight_layout()
        # -------------------------------------------------------------
        plt.figure()

        ax1 = plt.subplot(311)
        ax1.plot(time, interaction_forces[:, 3], 'r--', label="from sensor")
        ax1.plot(time, PD_force_command[:, 3], 'g--', label="from PD")
        ax1.legend()
        ax1.set_title("Mx [Nm]")

        ax2 = plt.subplot(312)
        ax2.plot(time, interaction_forces[:, 4], 'r--', label="from sensor")
        ax2.plot(time, PD_force_command[:, 4], 'g--', label="from PD")
        ax2.legend()
        ax2.set_title("My [Nm]")

        ax3 = plt.subplot(313)
        ax3.plot(time, interaction_forces[:, 5], 'r--', label="from sensor")
        ax3.plot(time, PD_force_command[:, 5], 'g--', label="from PD")
        ax3.legend()
        ax3.set_title("Mz [Nm]")

        plt.tight_layout()

        # plt.show()

        # contact_time = self.contact_time
        # contact_index = np.where(np.array(time) == contact_time)
        # contact_index = int(contact_index[0])
        # time = time[contact_index:]
        # t = np.array(time)
        # K = self.K
        # M = self.M
        # C = self.C
        # FT = np.array(self.desired_frame_FT_vec)
        # F = FT[:, 1]
        # # F = FT[:, 3]
        # y0 = [0, 0]
        # pos = np.array(self.desired_frame_imp_position_vec)
        # ori = np.array(self.desired_frame_imp_ori_vec)
        # vel = np.array(self.desired_frame_imp_vel_vec)
        # ang_vel = np.array(self.desired_frame_imp_ang_vel_vec)
        #
        # sol = odeint(self.pend, y0, t, args=(F, time, K, C, M))
        # plt.figure()
        # plt.plot(t, sol[:, 0], 'b', label='pos from ODE')
        # plt.plot(time, pos[:, 1], 'g',
        #          label='from simulation')
        # # plt.plot(time, ori[:, 0], 'g',
        # #          label='from simulation')
        # # plt.plot(time, min_jerk_position[contact_index:, 0] - impedance_position[contact_index:, 0], 'g',
        # #          label='from simulation')
        # plt.legend(loc='best')
        # plt.xlabel('t')
        # plt.grid()
        #
        # plt.figure()
        # plt.plot(t, sol[:, 1], 'b', label='vel from ODE')
        # plt.plot(time, vel[:, 1],
        #          'g',
        #          label='from simulation')
        # # plt.plot(time, ang_vel[:, 0],
        # #          'g',
        # #          label='from simulation')
        # plt.legend(loc='best')
        # plt.xlabel('t')
        # plt.grid()
        plt.show()
        return

    def pend(self, y, t, F, time, K, C, M):
        x, x_dot = y
        f_interp = interp1d(time, F, axis=0, fill_value="extrapolate")
        f = f_interp(t)
        dydt = [x_dot, -K / M * x - C / M * x_dot + f / M]
        return dydt

    # EC
    def is_robot_stable(self):
        is_robot_stable = True
        if self.is_contact:
            current_distance = np.linalg.norm(self.peg_edge - np.array(self.sim.data.site_xpos[
                                                     self.sim.model.site_name2id("hole_middle_cylinder")]))
            if self.nominal_stability_distance < current_distance:
                is_robot_stable = False
        return is_robot_stable
