import collections
from collections import OrderedDict
import numpy as np
import robosuite.utils.transform_utils as T
from scipy.spatial.transform import Rotation as R
from robosuite.utils.transform_utils import convert_quat
from robosuite.utils.mjcf_utils import CustomMaterial, array_to_string, new_site, find_elements

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv

from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject, PlateWithHoleObject, CylinderObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.observables import Observable, sensor


class PegInHole(SingleArmEnv):
    """
    This class corresponds to the lifting task for a single robot arm.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!

        env_configuration (str): Specifies how to position the robots within the environment (default is "default").
            For most single arm environments, this argument has no impact on the robot setup.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        table_full_size (3-tuple): x, y, and z dimensions of the table.

        table_friction (3-tuple): the three mujoco friction parameters for
            the table.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (cube) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        placement_initializer (ObjectPositionSampler): if provided, will
            be used to place objects on every reset, else a UniformRandomSampler
            is used by default.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

    Raises:
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(
            self,
            robots,
            env_configuration="default",
            user_init_qpos=None,
            controller_configs=None,
            gripper_types="default",
            initialization_noise="default",
            table_full_size=(0.8, 0.8, 0.05),
            table_friction=(1., 5e-3, 1e-4),
            table_height=0.8,
            use_camera_obs=True,
            use_object_obs=True,
            reward_scale=1.0,
            reward_shaping=False,
            placement_initializer=None,
            has_renderer=False,
            has_offscreen_renderer=True,
            render_camera="frontview",
            render_collision_mesh=False,
            render_visual_mesh=True,
            render_gpu_device_id=-1,
            control_freq=20,
            horizon=1000,
            ignore_done=False,
            hard_reset=True,
            camera_names="agentview",
            camera_heights=256,
            camera_widths=256,
            camera_depths=False,
            num_via_point=0,
            dist_error=0.002,
            angle_error=0,
            tanh_value=2.0,
            r_reach_value=0.94,
            error_type='circle',
            control_spec=36,
            peg_radius=0.004,  # (0.00125, 0.00125)
            peg_length=0.12,  # it is actually half of the peg length
            Peg_density=1,
            radial_error=0.0,
            angular_error=0.0,
    ):
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_height = table_height
        self.table_offset = np.array((0, 0, self.table_height))

        # only for putting the box in a better pos
        self.box_shift = np.array((0.2, 0, 0))
        controller_configs['box_shift'] = self.box_shift
        # in euler notation "xyz"
        self.box_ori = np.array([np.deg2rad(0), 0.0, 0])
        # self.box_ori = np.array([0, 0.0, 0])
        controller_configs['box_ori'] = self.box_ori
        # Save peg specs
        self.peg_radius = peg_radius
        self.peg_length = peg_length
        self.Peg_density = Peg_density
        self.radial_error = radial_error
        self.angular_error = angular_error

        self.dist_error = dist_error
        self.angle_error = angle_error

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        # EC - define desired initial joint position
        self.user_init_qpos = user_init_qpos

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            user_init_qpos=self.user_init_qpos,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,

        )

    def reward(self, action=None):
        """
        Reward function for the task.

        Sparse un-normalized reward:

            - a discrete reward of 2.25 is provided if the cube is lifted

        Un-normalized summed components if using reward shaping:

            - Reaching: in [0, 1], to encourage the arm to reach the cube
            - Grasping: in {0, 0.25}, non-zero if arm is grasping the cube
            - Lifting: in {0, 1}, non-zero if arm has lifted the cube

        The sparse reward only consists of the lifting component.

        Note that the final reward is normalized and scaled by
        reward_scale / 2.25 as well so that the max score is equal to reward_scale

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        # if not self.robots[0].controller.is_robot_stable_bool:
        #     reward = 0
        #     print(reward)
        #     return reward
        #
        # # check euclidean distance between peg edge and the goal
        # peg_edge = np.array(self.sim.data.site_xpos[self.sim.model.site_name2id("peg_site")])
        # hole_middle_cylinder = np.array(
        #     self.sim.data.site_xpos[self.sim.model.site_name2id("hole_middle_cylinder")])
        # dist = np.linalg.norm(peg_edge - hole_middle_cylinder)
        # eps = 0.0001  # avoiding inf rewards - the maximum reward is 10000
        # dist_reward = 1 / (dist + eps)

        # reward = dist_reward  # + direction_reward
        # print(reward)

        # return reward

        # if not self.robots[0].controller.is_robot_stable_bool:
        #     cost = 20000
        #     print(cost)
        #     return cost

        cost = self.robots[0].controller.total_cost
        # print(cost)
        return cost

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly

        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])
        self.rotation = None

        # initialize objects of interest
        self.peg = CylinderObject(name='peg',
                                  size=[self.peg_radius, self.peg_length],
                                  density=self.Peg_density,
                                  duplicate_collision_geoms=True,
                                  rgba=[1, 0, 0, 1], joints=None)

        # load peg object (returns extracted object in XML form)
        peg_obj = self.peg.get_obj()
        # set pegs position relative to place where it is being placed
        peg_obj.set("pos", array_to_string((0, 0, 0.145)))  # 0.145 is the middle of the fingers
        peg_obj.append(new_site(name="peg_site", pos=(0, 0, self.peg_length), size=(0.0002,)))
        # append the object top the gripper (attach body to body)
        # main_eef = self.robots[0].robot_model.eef_name    # 'robot0_right_hand'
        main_eef = self.robots[0].gripper.bodies[1]  # 'gripper0_eef' body
        main_model = self.robots[
            0].robot_model  # <robosuite.models.robots.manipulators.ur5e_robot.UR5e at 0x7fd9ead87ca0>
        main_body = find_elements(root=main_model.worldbody, tags="body", attribs={"name": main_eef}, return_first=True)
        main_body.append(peg_obj)  # attach body to body

        if self.rotation is None:
            rot_angle = np.random.uniform(high=2 * np.pi, low=0)
        elif isinstance(self.rotation, collections.Iterable):
            rot_angle = np.random.uniform(
                high=max(self.rotation), low=min(self.rotation)
            )
        else:
            rot_angle = self.rotation

        # max_radial_shift = 0.0025
        # min_radial_shift = 0.0012
        # radial_shift = np.random.uniform(high=max_radial_shift, low=min_radial_shift)
        # shift_angel = np.random.uniform(high=2 * np.pi, low=0)
        # hole_pos_set = np.array([radial_shift * np.cos(shift_angel), radial_shift * np.sin(shift_angel),
        #                          self.table_height]) + self.box_shift
        hole_pos_set = np.array([0, 0.0025, self.table_height]) + self.box_shift

        # hole_pos_set = np.array([self.radial_error * np.cos(self.angular_error),
        #                          self.radial_error * np.sin(self.angular_error),
        #                          self.table_height]) + self.box_shift

        # for euler it is "xyz"
        hole_rot_set = np.copy(self.box_ori)

        hole_pos_str = ' '.join(map(str, hole_pos_set))
        hole_rot_str = ' '.join(map(str, hole_rot_set))

        self.hole = PlateWithHoleObject(name='hole')
        hole_obj = self.hole.get_obj()
        # hole_obj.set("axisangle", hole_rot_str)
        hole_obj.set("euler", hole_rot_str)
        hole_obj.set("pos", hole_pos_str)

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.hole
        )

        # Make sure to add relevant assets from peg and hole objects
        self.model.merge_assets(self.peg)
        # print(self.model.get_xml())

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.peg_body_id = self.sim.model.body_name2id(self.peg.root_body)

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled
        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            # Get robot prefix and define observables modality
            pf = self.robots[0].robot_model.naming_prefix
            modality = "object"

            # peg-related observables
            @sensor(modality=modality)
            def peg_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.peg_body_id])

            @sensor(modality=modality)
            def peg_quat(obs_cache):
                return T.convert_quat(np.array(self.sim.data.body_xquat[self.peg_body_id]), to="xyzw")

            @sensor(modality=modality)
            def gripper_to_peg_pos(obs_cache):
                return obs_cache[f"{pf}eef_pos"] - obs_cache["peg_pos"] if \
                    f"{pf}eef_pos" in obs_cache and "peg_pos" in obs_cache else np.zeros(3)

            sensors = [peg_pos, peg_quat, gripper_to_peg_pos]
            names = [s.__name__ for s in sensors]

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()
        # # Reset all object positions using initializer sampler if we're not directly loading from an xml
        # if not self.deterministic_reset:
        #
        #     # Sample from the placement initializer for all objects
        #     object_placements = self.placement_initializer.sample()
        #
        #     # Loop through all objects and reset their positions
        #     for obj_pos, obj_quat, obj in object_placements.values():
        #         self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the cube.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the cube
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.peg)

    def _check_success(self):
        """
        Check if cube has been lifted.

        Returns:
            bool: True if cube has been lifted
        """
        # cube_height = self.sim.data.body_xpos[self.cube_body_id][2]
        # table_height = self.model.mujoco_arena.table_offset[2]
        #
        # # cube is higher than the table top above a margin
        # return cube_height > table_height + 0.04
        return 1
