from copy import deepcopy

from robosuite.controllers.base_controller import Controller
import numpy as np
import robosuite.utils.angle_transformation as at
from robosuite.utils.control_utils import opspace_matrices
from scipy.spatial.transform import Rotation as R
import robosuite.utils.transform_utils as T
from robosuite.utils.control_utils import build_imp_matrices_circular_peg


class JointTorqueController(Controller):
    """
    Controller for controlling the robot arm's joint torques. As the actuators at the mujoco sim level are already
    torque actuators, this "controller" usually simply "passes through" desired torques, though it also includes the
    typical input / output scaling and clipping, as well as interpolator features seen in other controllers classes
    as well

    NOTE: Control input actions assumed to be taken as absolute joint torques. A given action to this
    controller is assumed to be of the form: (torq_j0, torq_j1, ... , torq_jn-1) for an n-joint robot

    Args:
        sim (MjSim): Simulator instance this controller will pull robot state updates from

        eef_name (str): Name of controlled robot arm's end effector (from robot XML)

        joint_indexes (dict): Each key contains sim reference indexes to relevant robot joint information, namely:

            :`'joints'`: list of indexes to relevant robot joints
            :`'qpos'`: list of indexes to relevant robot joint positions
            :`'qvel'`: list of indexes to relevant robot joint velocities

        actuator_range (2-tuple of array of float): 2-Tuple (low, high) representing the robot joint actuator range

        input_max (float or list of float): Maximum above which an inputted action will be clipped. Can be either be
            a scalar (same value for all action dimensions), or a list (specific values for each dimension). If the
            latter, dimension should be the same as the control dimension for this controller

        input_min (float or list of float): Minimum below which an inputted action will be clipped. Can be either be
            a scalar (same value for all action dimensions), or a list (specific values for each dimension). If the
            latter, dimension should be the same as the control dimension for this controller

        output_max (float or list of float): Maximum which defines upper end of scaling range when scaling an input
            action. Can be either be a scalar (same value for all action dimensions), or a list (specific values for
            each dimension). If the latter, dimension should be the same as the control dimension for this controller

        output_min (float or list of float): Minimum which defines upper end of scaling range when scaling an input
            action. Can be either be a scalar (same value for all action dimensions), or a list (specific values for
            each dimension). If the latter, dimension should be the same as the control dimension for this controller

        policy_freq (int): Frequency at which actions from the robot policy are fed into this controller

        torque_limits (2-list of float or 2-list of list of floats): Limits (N-m) below and above which the magnitude
            of a calculated goal joint torque will be clipped. Can be either be a 2-list (same min/max value for all
            joint dims), or a 2-list of list (specific min/max values for each dim)
            If not specified, will automatically set the limits to the actuator limits for this robot arm

        interpolator (Interpolator): Interpolator object to be used for interpolating from the current joint torques to
            the goal joint torques during each timestep between inputted actions

        **kwargs: Does nothing; placeholder to "sink" any additional arguments so that instantiating this controller
            via an argument dict that has additional extraneous arguments won't raise an error
    """

    def __init__(self,
                 sim,
                 eef_name,
                 joint_indexes,
                 actuator_range,
                 input_max=1,
                 input_min=-1,
                 output_max=0.05,
                 output_min=-0.05,
                 control_dim=108,
                 policy_freq=None,
                 torque_limits=None,
                 interpolator=None,
                 plotting=False,
                 collect_data=False,
                 simulation_total_time=None,
                 box_shift=None,
                 box_ori=None,
                 **kwargs,  # does nothing; used so no error raised when dict is passed with extra terms used previously

                 ):

        super().__init__(
            sim,
            eef_name,
            joint_indexes,
            actuator_range,
            plotting,
            collect_data,
            simulation_total_time,
            box_shift,
            box_ori,
        )

        # Control dimension
        # self.control_dim = len(joint_indexes["joints"])
        self.control_dim = control_dim
        # input and output max and min (allow for either explicit lists or single numbers)
        self.input_max = self.nums2array(input_max, self.control_dim)
        self.input_min = self.nums2array(input_min, self.control_dim)
        self.output_max = self.nums2array(output_max, self.control_dim)
        self.output_min = self.nums2array(output_min, self.control_dim)

        # limits (if not specified, set them to actuator limits by default)
        # self.torque_limits = np.array(torque_limits) if torque_limits is not None else self.actuator_limits
        self.torque_limits = self.actuator_limits

        # control frequency
        self.control_freq = policy_freq

        # interpolator
        self.interpolator = interpolator

        # initialize torques
        self.goal_torque = None  # Goal torque desired, pre-compensation
        self.current_torque = np.zeros(self.control_dim)  # Current torques being outputted, pre-compensation
        self.torques = None  # Torques returned every time run_controller is called

    def set_goal(self, action):
        """
        Sets goal based on input @torques.

        Args:
            torques (Iterable): Desired joint torques

        Raises:
            AssertionError: [Invalid action dimension size]
        """
        # Update state
        self.update()
        # EC - this is not a good if statement! need to be changed!
        if not self.plotting:
            self.K_imp, self.C_imp, self.M_imp, A, self.A_d, self.B_d = build_imp_matrices_circular_peg(action)
        else:
            # 3 mm
            # action = np.array([4.35259205e+03, 9.15022671e-03, 8.55004503e+03, 3.56983938e+01,
            #                    4.83567059e+03, 6.18553957e+03, 2.61789874e+00, 9.05644681e+00,
            #                    1.77795127e+01, 5.75398765e+01, 8.54733279e+00, 1.21962811e+00])
            # 2mm
            # action = np.array([2.63713898e+03, 9.11008932e-03, 4.13452192e+03, 4.84758140e+01,
            #                    4.05917899e+03, 3.61409106e+03, 5.56474596e+01, 7.32159378e+01,
            #                    6.96215856e+00, 3.04045775e+01, 5.19098970e+01, 7.59970523e+00])
            # 2.5 mm with 25 param
            action = np.array([3.40711857e+03, 4.19765954e+03, 8.83809649e-03, 4.08465502e+03,
                               1.89018108e+03, 6.14752723e+01, 6.43077143e+03, -6.28038101e+03,
                               -3.56963539e+03, 9.56465797e+03, 6.01699806e+00, 5.28973427e+01,
                               2.38419869e+01, 3.60602664e+01, 9.00807238e+01, 1.42865942e+00])
            self.K_imp, self.C_imp, self.M_imp, A, self.A_d, self.B_d = build_imp_matrices_circular_peg(action)

        self.goal_torque = np.zeros(self.control_dim)  # just for sending something. it doesn't matter

    def run_controller(self):
        """
        Calculates the torques required to reach the desired setpoint

        Returns:
             np.array: Command torques
        """
        # Make sure goal has been set
        if self.goal_torque is None:
            self.set_goal(np.zeros(self.control_dim))

        # Update state
        self.update()
        # EC - calculate minimum jerk path
        if self.time <= self.second_trajectory_part_final_time:
            self._min_jerk()

        # check the if the any axis force is greater then some value
        if any(abs(self.interaction_forces) > 10) and self.first_trajectory_part_final_time <= self.time:
            self.is_contact = True

        if self.is_contact:
            if self.first_contact:
                self.first_contact = False
                self.contact_time = deepcopy(self.time)
                self.X_m = deepcopy(np.concatenate((np.zeros(3, ), np.zeros(3, ), np.zeros(3, ), np.zeros(3, ))))
                self.peg_edge_contact_position = np.copy(self.peg_edge)
                self.peg_and_hole_contact_distance = np.linalg.norm(self.peg_edge_contact_position
                                                                    - self.nominal_hole_middle_cylinder)
                # print("contact")
                # print(self.contact_time)
            self.cumulative_cost()
            self.impedance_computations()
            Rotation_world_to_desired = R.from_euler("xyz", self.min_jerk_orientation, degrees=False).as_matrix()
            compliance_position_relative_to_desired = (self.X_m[:3]).reshape(3, )
            compliance_velocity_relative_to_desired = (self.X_m[6:9]).reshape(3, )
            compliance_rotVec_relative_to_desired = (self.X_m[3:6]).reshape(3, )
            compliance_ang_velocity_relative_to_desired = (self.X_m[9:12]).reshape(3, )

            compliance_position = Rotation_world_to_desired @ compliance_position_relative_to_desired + \
                                  self.min_jerk_position

            Rotation_desired_to_compliance = R.from_rotvec(compliance_rotVec_relative_to_desired).as_matrix()
            Rotation_world_to_compliance = Rotation_world_to_desired @ Rotation_desired_to_compliance

            compliance_velocity = Rotation_world_to_desired @ compliance_velocity_relative_to_desired \
                                  + self.min_jerk_velocity

            compliance_ang_velocity = Rotation_world_to_desired @ compliance_ang_velocity_relative_to_desired \
                                      + self.min_jerk_ang_vel

            self.PD_control(compliance_position, Rotation_world_to_compliance,
                            compliance_velocity, compliance_ang_velocity)
            # compute the values of self.current_torque based on the impedance parameters
        else:
            self.PD_control(self.min_jerk_position, R.from_euler("xyz", self.min_jerk_orientation, degrees=False)
                            .as_matrix(), self.min_jerk_velocity, self.min_jerk_ang_vel)
            # compute the values of self.current_torque based on the minimum jerk trajectory

        # Add gravity compensation
        self.torques = self.current_torque + self.torque_compensation
        # Always run superclass call for any cleanups at the end
        super().run_controller()

        # Return final torques
        return self.torques

    def reset_goal(self):
        """
        Resets joint torque goal to be all zeros (pre-compensation)
        """
        self.goal_torque = np.zeros(self.control_dim)

        # Reset interpolator if required
        if self.interpolator is not None:
            self.interpolator.set_goal(self.goal_torque)

    @property
    def name(self):
        return 'JOINT_TORQUE'

    def PD_control(self, desired_position, desired_orientation, desired_velocity, desired_angle_velocity):
        # EC - compute the error between desired values and real values
        # desired_orientation needs to be a rotation matrix!
        position_error = desired_position - self.ee_pos

        orientation_error = at.Rotation_Matrix_To_Vector(self.ee_ori_mat, desired_orientation)

        velocity_error = desired_velocity - self.ee_pos_vel
        rotational_velocity_error = (desired_angle_velocity - self.ee_ori_vel)

        # error = np.concatenate((position_error, orientation_error), axis=0)
        # error_dot = np.concatenate((velocity_error, rotational_velocity_error), axis=0)

        # desired_acceleration = np.concatenate((self.desired_acceleration, desired_angle_acceleration), axis=0)

        # Compute nullspace matrix (I - Jbar * J) and lambda matrices ((J * M^-1 * J^T)^-1)
        lambda_full, lambda_pos, lambda_ori, nullspace_matrix = opspace_matrices(self.mass_matrix,
                                                                                 self.J_full,
                                                                                 self.J_pos,
                                                                                 self.J_ori)

        if self.is_contact:  # contact control
            z_pos = 450 * np.ones(1)
            Kp_pos = np.append(700 * np.ones(2), z_pos)
            # Kp_pos = 700 * np.ones(3)
            Kp_ori = 2 * 700 * np.ones(3)
            Kp = np.append(Kp_pos, Kp_ori)
            # z_pos = 450 * np.ones(1)
            # Kp_pos = np.append(4500 * np.ones(2), z_pos)
            # # Kp_pos = 700 * np.ones(3)
            # Kp_ori = 2 * 4500 * np.ones(3)
            # Kp = np.append(Kp_pos, Kp_ori)

            Kd_pos = 0.707 * 2 * np.sqrt(Kp_pos)
            Kd_ori = 2 * 0.75 * 0.707 * 2 * np.sqrt(Kp_ori)
            Kd = np.append(Kd_pos, Kd_ori)

            # error = np.concatenate((position_error, orientation_error), axis=0)
            # error_dot = np.concatenate((velocity_error, rotational_velocity_error), axis=0)
            # wrench1 = Kp * error + Kd * error_dot

            error_box_frame = np.concatenate(
                (self.box_ori_mat.T @ position_error, self.box_ori_mat.T @ orientation_error), axis=0)
            error_dot_box_frame = np.concatenate(
                (self.box_ori_mat.T @ velocity_error, self.box_ori_mat.T @ rotational_velocity_error), axis=0)

            # wrench = Kp * error + Kd * error_dot
            wrench_box_frame = Kp * error_box_frame + Kd * error_dot_box_frame
            wrench = np.concatenate((self.box_ori_mat @ wrench_box_frame[:3], self.box_ori_mat @ wrench_box_frame[3:6]),
                                    axis=0)
        else:  # free space control
            # important! - this method is only suitable for free space movement.
            # if using it during contact it gives bad results!
            Kp_pos = 1 * 4500 * np.ones(3)
            Kp_ori = 2 * 4500 * np.ones(3)
            Kp = np.append(Kp_pos, Kp_ori)

            Kd_pos = 0.707 * 2 * np.sqrt(Kp_pos)
            Kd_ori = 2 * 0.75 * 0.707 * 2 * np.sqrt(Kp_ori)
            Kd = np.append(Kd_pos, Kd_ori)

            error = np.concatenate((position_error, orientation_error), axis=0)
            error_dot = np.concatenate((velocity_error, rotational_velocity_error), axis=0)

            wrench = Kp * error + Kd * error_dot

            wrench = np.dot(lambda_full, wrench)

        torques = np.dot(self.J_full.T,
                         wrench)  # - (np.dot(self.J_full.T, self.sim.data.sensordata)) * self.is_contact

        # make the robot work under torque limitations
        self.goal_torque = np.clip(torques, self.torque_limits[0], self.torque_limits[1])
        self.goal_torque = torques

        # EC - take measurements for graphs
        if self.collect_data:
            self.real_position = self.ee_pos
            self.real_velocity = self.ee_pos_vel
            if self.time == 0:
                self.real_orientation = R.from_matrix(self.ee_ori_mat).as_rotvec()
            elif np.dot(R.from_matrix(self.ee_ori_mat).as_rotvec(), self.real_orientation) > 0:
                self.real_orientation = R.from_matrix(self.ee_ori_mat).as_rotvec()
            elif np.dot(R.from_matrix(self.ee_ori_mat).as_rotvec(), self.real_orientation) <= 0:
                self.real_orientation = -1 * R.from_matrix(self.ee_ori_mat).as_rotvec()
            self.real_angle_velocity = self.ee_ori_vel

            self.impedance_position_vec.append(desired_position)
            self.impedance_velocity_vec.append(desired_velocity)
            # for the orientation part, make sure the rotation vectors are at the same direction as the previous step
            # the graph will show the rotation vector between the world frame and XX frame writen in the world frame
            if self.time == 0:
                self.impedance_orientation = R.from_matrix(desired_orientation).as_rotvec()
            elif np.dot(R.from_matrix(desired_orientation).as_rotvec(), self.impedance_orientation) > 0:
                self.impedance_orientation = R.from_matrix(desired_orientation).as_rotvec()
            elif np.dot(R.from_matrix(desired_orientation).as_rotvec(), self.impedance_orientation) < 0:
                self.impedance_orientation = -1 * R.from_matrix(desired_orientation).as_rotvec()
            self.impedance_orientation_vec.append(self.impedance_orientation)
            self.impedance_angle_velocity_vec.append(desired_angle_velocity)

            self.PD_force_command.append(wrench)
            self.add_path_parameter()
            #
            if self.time >= self.simulation_total_time - 2 * self.Delta_T and self.plotting:
                # print(np.array(self.sim.data.qpos[self.qpos_index]))
                self.plotter()

        # Only linear interpolator is currently supported
        if self.interpolator is not None:
            # Linear case
            if self.interpolator.order == 1:
                self.current_torque = self.interpolator.get_interpolated_goal()
            else:
                # Nonlinear case not currently supported
                pass
        else:
            self.current_torque = np.array(self.goal_torque)

        return

    def cumulative_cost(self):
        distance_cost = (np.linalg.norm(self.peg_edge - self.nominal_hole_middle_cylinder)) / \
                        self.peg_and_hole_contact_distance
        # z_world_vec_in_world_frame = np.array([0, 0, 1])
        # z_minus_ee_vec_in_world_frame = self.ee_ori_mat @ np.array([0, 0, -1])
        # # calculate the angle between the peg -z axis and z world in world frame
        # angle_cost = np.arcsin(np.linalg.norm(np.cross(z_world_vec_in_world_frame, z_minus_ee_vec_in_world_frame)))
        # self.total_cost += distance_cost + angle_cost
        self.total_cost += distance_cost
