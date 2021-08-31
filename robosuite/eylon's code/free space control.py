"""
This demo script demonstrates the various functionalities of each controller available within robosuite.

For a given controller, runs through each dimension and executes a perturbation "test_value" from its
neutral (stationary) value for a certain amount of time "steps_per_action", and then returns to all neutral values
for time "steps_per_rest" before proceeding with the next action dim.

    E.g.: Given that the expected action space of the Pos / Ori (OSC_POSE) controller (without a gripper) is
    (dx, dy, dz, droll, dpitch, dyaw), the testing sequence of actions over time will be:

        ***START OF DEMO***
        ( dx,  0,  0,  0,  0,  0, grip)     <-- Translation in x-direction      for 'steps_per_action' steps
        (  0,  0,  0,  0,  0,  0, grip)     <-- No movement (pause)             for 'steps_per_rest' steps
        (  0, dy,  0,  0,  0,  0, grip)     <-- Translation in y-direction      for 'steps_per_action' steps
        (  0,  0,  0,  0,  0,  0, grip)     <-- No movement (pause)             for 'steps_per_rest' steps
        (  0,  0, dz,  0,  0,  0, grip)     <-- Translation in z-direction      for 'steps_per_action' steps
        (  0,  0,  0,  0,  0,  0, grip)     <-- No movement (pause)             for 'steps_per_rest' steps
        (  0,  0,  0, dr,  0,  0, grip)     <-- Rotation in roll (x) axis       for 'steps_per_action' steps
        (  0,  0,  0,  0,  0,  0, grip)     <-- No movement (pause)             for 'steps_per_rest' steps
        (  0,  0,  0,  0, dp,  0, grip)     <-- Rotation in pitch (y) axis      for 'steps_per_action' steps
        (  0,  0,  0,  0,  0,  0, grip)     <-- No movement (pause)             for 'steps_per_rest' steps
        (  0,  0,  0,  0,  0, dy, grip)     <-- Rotation in yaw (z) axis        for 'steps_per_action' steps
        (  0,  0,  0,  0,  0,  0, grip)     <-- No movement (pause)             for 'steps_per_rest' steps
        ***END OF DEMO***

    Thus the OSC_POSE controller should be expected to sequentially move linearly in the x direction first,
        then the y direction, then the z direction, and then begin sequentially rotating about its x-axis,
        then y-axis, then z-axis.

Please reference the documentation of Controllers in the Modules section for an overview of each controller.
Controllers are expected to behave in a generally controlled manner, according to their control space. The expected
sequential qualitative behavior during the test is described below for each controller:

* OSC_POSE: Gripper moves sequentially and linearly in x, y, z direction, then sequentially rotates in x-axis, y-axis,
            z-axis, relative to the global coordinate frame
* OSC_POSITION: Gripper moves sequentially and linearly in x, y, z direction, relative to the global coordinate frame
* IK_POSE: Gripper moves sequentially and linearly in x, y, z direction, then sequentially rotates in x-axis, y-axis,
            z-axis, relative to the local robot end effector frame
* JOINT_POSITION: Robot Joints move sequentially in a controlled fashion
* JOINT_VELOCITY: Robot Joints move sequentially in a controlled fashion
* JOINT_TORQUE: Unlike other controllers, joint torque controller is expected to act rather lethargic, as the
            "controller" is really just a wrapper for direct torque control of the mujoco actuators. Therefore, a
            "neutral" value of 0 torque will not guarantee a stable robot when it has non-zero velocity!

"""

import robosuite as suite
from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *
from robosuite.robots import Bimanual

if __name__ == "__main__":

    # Create dict to hold options that will be passed to env creation call
    options = {"env_name": "Lift", "robots": "UR5e"}

    # Choose environment and add it to options
    joint_dim = 6
    controller_name = "JOINT_TORQUE"
    # Load the desired controller
    options["controller_configs"] = suite.load_controller_config(default_controller=controller_name)

    # Define variables for each controller test
    action_dim = joint_dim
    num_test_steps = joint_dim
    test_value = 0.25

    # Define the number of timestamps to use per controller action as well as timestamps in between actions
    steps_per_action = 75
    steps_per_rest = 75

    # Help message to user
    print()
    print("Press \"H\" to show the viewer control panel.")

    # initialize the task
    env = suite.make(
        **options,
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        horizon=(steps_per_action + steps_per_rest) * num_test_steps,
        # control_freq=20,
        control_freq=50,  # EC
    )
    env.reset()
    env.viewer.set_camera(camera_id=0)

    # To accommodate for multi-arm settings (e.g.: Baxter), we need to make sure to fill any extra action space
    # Get total number of arms being controlled
    n = 0
    gripper_dim = 0
    for robot in env.robots:
        gripper_dim = robot.gripper["right"].dof if isinstance(robot, Bimanual) else robot.gripper.dof
        n += int(robot.action_dim / (action_dim + gripper_dim))

    # Define neutral value
    neutral = np.zeros(action_dim + gripper_dim)

    # Keep track of done variable to know when to break loop
    count = 0
    # Loop through controller space
    while count < num_test_steps:
        action = neutral.copy()
        # for i in range(steps_per_action):
        #     action[count] = test_value
        #     total_action = np.tile(action, n)
        #     env.step(total_action)
        #     env.render()
        # for i in range(steps_per_rest):
        #     total_action = np.tile(neutral, n)
        #     env.step(total_action)
        #     env.render()
        env.step(action)
        env.render()
        count += 1

    # Shut down this env before starting the next test
    env.close()
