"""
Minimum jerk trajectory for 6DOF robot
Written by Daniel Stankowski and Elad Newman
"""
import numpy as np
from robosuite.utils import angle_transformation as at


class PathPlan(object):

    def __init__(self, position_init, pose_desired, total_time, plan_orientation_in):
        #
        self.position_init = position_init[:3]
        self.position_des = pose_desired[:3]

        self.orientation_init = position_init[3:]
        self.orientation_des = pose_desired[3:]

        self.tfinal = total_time
        self.plan_orientation_in = plan_orientation_in

        if not (self.plan_orientation_in == 'axis' or self.plan_orientation_in == 'rot_vec'):
            raise ValueError(
                f'self.plan_orientation_in should get the values "axis" or "rot_vec". instead '
                f'self.plan_orientation_in was set to "{self.plan_orientation_in}"')

    def trajectory_planning(self, t):
        X_init = self.position_init[0]
        Y_init = self.position_init[1]
        Z_init = self.position_init[2]

        X_final = self.position_des[0]
        Y_final = self.position_des[1]
        Z_final = self.position_des[2]

        # position
        x_traj = (X_final - X_init) / (self.tfinal ** 3) * (
                6 * (t ** 5) / (self.tfinal ** 2) - 15 * (t ** 4) / self.tfinal + 10 * (t ** 3)) + X_init
        y_traj = (Y_final - Y_init) / (self.tfinal ** 3) * (
                6 * (t ** 5) / (self.tfinal ** 2) - 15 * (t ** 4) / self.tfinal + 10 * (t ** 3)) + Y_init
        z_traj = (Z_final - Z_init) / (self.tfinal ** 3) * (
                6 * (t ** 5) / (self.tfinal ** 2) - 15 * (t ** 4) / self.tfinal + 10 * (t ** 3)) + Z_init
        position = np.array([x_traj, y_traj, z_traj])

        # velocities
        vx = (X_final - X_init) / (self.tfinal ** 3) * (
                30 * (t ** 4) / (self.tfinal ** 2) - 60 * (t ** 3) / self.tfinal + 30 * (t ** 2))
        vy = (Y_final - Y_init) / (self.tfinal ** 3) * (
                30 * (t ** 4) / (self.tfinal ** 2) - 60 * (t ** 3) / self.tfinal + 30 * (t ** 2))
        vz = (Z_final - Z_init) / (self.tfinal ** 3) * (
                30 * (t ** 4) / (self.tfinal ** 2) - 60 * (t ** 3) / self.tfinal + 30 * (t ** 2))
        velocity = np.array([vx, vy, vz])

        # acceleration
        ax = (X_final - X_init) / (self.tfinal ** 3) * (
                120 * (t ** 3) / (self.tfinal ** 2) - 180 * (t ** 2) / self.tfinal + 60 * t)
        ay = (Y_final - Y_init) / (self.tfinal ** 3) * (
                120 * (t ** 3) / (self.tfinal ** 2) - 180 * (t ** 2) / self.tfinal + 60 * t)
        az = (Z_final - Z_init) / (self.tfinal ** 3) * (
                120 * (t ** 3) / (self.tfinal ** 2) - 180 * (t ** 2) / self.tfinal + 60 * t)
        acceleration = np.array([ax, ay, az])

        # = = = = = = = = = = = Orientation = = = = = = = = = = = = = = =
        if self.plan_orientation_in == 'axis':
            Rx_init = self.orientation_init[0]
            Ry_init = self.orientation_init[1]
            Rz_init = self.orientation_init[2]
            Rx_final = self.orientation_des[0]
            Ry_final = self.orientation_des[1]
            Rz_final = self.orientation_des[2]

            # Orientation
            Rx_traj = (Rx_final - Rx_init) / (self.tfinal ** 3) * (
                    6 * (t ** 5) / (self.tfinal ** 2) - 15 * (t ** 4) / self.tfinal + 10 * (t ** 3)) + Rx_init
            Ry_traj = (Ry_final - Ry_init) / (self.tfinal ** 3) * (
                    6 * (t ** 5) / (self.tfinal ** 2) - 15 * (t ** 4) / self.tfinal + 10 * (t ** 3)) + Ry_init
            Rz_traj = (Rz_final - Rz_init) / (self.tfinal ** 3) * (
                    6 * (t ** 5) / (self.tfinal ** 2) - 15 * (t ** 4) / self.tfinal + 10 * (t ** 3)) + Rz_init
            orientation = np.array([Rx_traj, Ry_traj, Rz_traj])

            # Angular Velocities
            dRx = (Rx_final - Rx_init) / (self.tfinal ** 3) * (
                    30 * (t ** 4) / (self.tfinal ** 2) - 60 * (t ** 3) / self.tfinal + 30 * (t ** 2))
            dRy = (Ry_final - Ry_init) / (self.tfinal ** 3) * (
                    30 * (t ** 4) / (self.tfinal ** 2) - 60 * (t ** 3) / self.tfinal + 30 * (t ** 2))
            dRz = (Rz_final - Rz_init) / (self.tfinal ** 3) * (
                    30 * (t ** 4) / (self.tfinal ** 2) - 60 * (t ** 3) / self.tfinal + 30 * (t ** 2))
            ang_vel = np.array([dRx, dRy, dRz])

        elif self.plan_orientation_in == 'rot_vec':
            #   -----rotation (based on rotation vector from initial to desired) -----------------
            Vrot = at.RotationVector(self.orientation_init, self.orientation_des)  # creates rotation vector
            # In case of lack of rotation:
            upper_bound = 1e-6
            if np.linalg.norm(Vrot) < upper_bound:
                # magnitude = 0.0
                magnitude_traj = 0.0
                magnitude_vel_traj = 0.0
                direction = np.array([0.0, 0.0, 0.0])
            else:
                magnitude, direction = at.Axis2Vector(Vrot)
                #   we want to decrease the magnitude of the rotation from some initial value to 0
                magnitude_traj = (0 - magnitude) / (self.tfinal ** 3) * (
                            6 * (t ** 5) / (self.tfinal ** 2) - 15 * (t ** 4) / self.tfinal + 10 * (t ** 3)) + magnitude
                magnitude_vel_traj = (0 - magnitude) / (self.tfinal ** 3) * (
                            30 * (t ** 4) / (self.tfinal ** 2) - 60 * (t ** 3) / self.tfinal + 30 * (t ** 2))

            orientation = magnitude_traj * direction
            ang_vel = magnitude_vel_traj * direction
        else:
            print(
                f'self.plan_orientation_in should get the values "axis" or "rot_vec". instead '
                f'self.plan_orientation_in was set to "{self.plan_orientation_in}"')
            raise ValueError(
                f'self.plan_orientation_in should get the values "axis" or "rot_vec". instead '
                f'self.plan_orientation_in was set to "{self.plan_orientation_in}"')
        return [position, orientation, velocity, ang_vel, acceleration]
