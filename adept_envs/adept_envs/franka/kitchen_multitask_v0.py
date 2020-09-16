""" Kitchen environment for long horizon manipulation """
#!/usr/bin/python
#
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np
from adept_envs import robot_env
from adept_envs.utils.configurable import configurable
from gym import spaces
from dm_control.mujoco import engine

from .utils import reset_mocap2body_xpos, reset_mocap_welds
from .rotations import euler2quat, mat2euler, quat2euler, mat2quat

CAMERAS = {
    0: dict(distance=4.5, azimuth=-66, elevation=-65),
    1: dict(distance=2.2, lookat=[-0.2, .5, 2.], azimuth=70, elevation=-35), # as in https://relay-policy-learning.github.io/
    2: dict(distance=2.65, lookat=[0, 0, 2.], azimuth=90, elevation=-60), # similar to appendix D of https://arxiv.org/pdf/1910.11956.pdf
    3: dict(distance=2.5, lookat=[-0.2, .5, 2.], azimuth=90, elevation=-60), # 3-6 are first person views at different angles and distances
    4: dict(distance=2.5, lookat=[-0.2, .5, 2.], azimuth=90, elevation=-45), # problem w/ POV is that the knobs can be hidden by the hinge drawer and arm
    5: dict(distance=2.2, lookat=[-0.2, .5, 2.], azimuth=90, elevation=-45),
    6: dict(distance=2.2, lookat=[-0.2, .5, 2.], azimuth=90, elevation=-10),
}
@configurable(pickleable=True)
class KitchenV0(robot_env.RobotEnv):

    CALIBRATION_PATHS = {
        'default':
        os.path.join(os.path.dirname(__file__), 'robot/franka_config.xml')
    }
    # Converted to velocity actuation
    ROBOTS = {'robot': 'adept_envs.franka.robot.franka_robot:Robot_VelAct'}
    MODEL = os.path.join(
        os.path.dirname(__file__),
        'assets/franka_kitchen_jntpos_act_ab.xml')
    MODEL_TELEOP = os.path.join(
        os.path.dirname(__file__),
        'assets/franka_kitchen_teleop.xml')
    N_DOF_ROBOT = 9
    N_DOF_OBJECT = 21

    def __init__(self, robot_params={}, frame_skip=40, camera_id=1, use_mocap_ctrl=False, mocap_use_euler=False):
        # self.goal_concat = True
        # self.goal = np.zeros((30,))
        self.obs_dict = {}
        self.robot_noise_ratio = 0.1  # 10% as per robot_config specs
        self.camera_id = camera_id
        self.use_mocap_ctrl = use_mocap_ctrl
        self.mocap_use_euler = mocap_use_euler

        self.pos_range = 0.05 # limit maximum change in position
        self.rot_range = 0.05

        super().__init__(
            self.MODEL_TELEOP if self.use_mocap_ctrl else self.MODEL,
            robot=self.make_robot(
                n_jnt=self.N_DOF_ROBOT,  #root+robot_jnts
                n_obj=self.N_DOF_OBJECT,
                **robot_params),
            frame_skip=frame_skip,
            camera_settings=CAMERAS[camera_id],
        )
        self.init_qpos = self.sim.model.key_qpos[0].copy()

        # For the microwave kettle slide hinge
        self.init_qpos = np.array([ 1.48388023e-01, -1.76848573e+00,  1.84390296e+00, -2.47685760e+00,
                                    2.60252026e-01,  7.12533105e-01,  1.59515394e+00,  4.79267505e-02,
                                    3.71350919e-02, -2.66279850e-04, -5.18043486e-05,  3.12877220e-05,
                                   -4.51199853e-05, -3.90842156e-06, -4.22629655e-05,  6.28065475e-05,
                                    4.04984708e-05,  4.62730939e-04, -2.26906415e-04, -4.65501369e-04,
                                   -6.44129196e-03, -1.77048263e-03,  1.08009684e-03, -2.69397440e-01,
                                    3.50383255e-01,  1.61944683e+00,  1.00618764e+00,  4.06395120e-03,
                                   -6.62095997e-03, -2.68278933e-04])

        self.init_qvel = self.sim.model.key_qvel[0].copy()

        action_dim = 8 if self.use_mocap_ctrl and self.mocap_use_euler else self.N_DOF_ROBOT
        self.act_mid = np.zeros(action_dim)
        self.act_amp = 2.0 * np.ones(action_dim)
        act_lower = -1*np.ones((action_dim,))
        act_upper =  1*np.ones((action_dim,))
        self.action_space = spaces.Box(act_lower, act_upper)

        obs_upper = 8. * np.ones(self.obs_dim)
        obs_lower = -obs_upper
        self.observation_space = spaces.Box(obs_lower, obs_upper)

    def _get_reward_n_score(self, obs_dict):
        raise NotImplementedError()

    def step(self, *args, **kwargs):
        if self.use_mocap_ctrl:
            self.step_mocap(*args, **kwargs)
        else:
            self.step_joints(*args, **kwargs)

        # observations
        obs = self._get_obs()

        #rewards
        reward_dict, score = self._get_reward_n_score(self.obs_dict)

        # termination
        done = False

        # finalize step
        env_info = {
            # 'time': self.obs_dict['t'],
            'obs_dict': self.obs_dict,
            # 'rewards': reward_dict,
            # 'score': score,
            # 'images': np.asarray(self.render(mode='rgb_array'))
        }
        # self.render()
        return obs, reward_dict['r_total'], done, env_info

    def step_mocap(self, a, b=None):
        a = np.clip(a, -1.0, 1.0)

        reset_mocap2body_xpos(self.sim)

        # split action [3-dim Cartesian coordinate, 3-dim euler angle OR 4-dim quarternion, 2-dim gripper joints]
        current_pos = self.sim.data.mocap_pos.copy()
        new_pos = current_pos + a[:3] * self.pos_range
        self.sim.data.mocap_pos[:] = new_pos.copy()

        if self.mocap_use_euler:
            current_quat = self.sim.data.mocap_quat.copy()
            current_rot = quat2euler(current_quat) 
            new_rot = current_rot + a[3:6] * self.rot_range
            new_quat = euler2quat(new_rot)[0]
            self.sim.data.mocap_quat[:] = new_quat.copy()

            gripper_a = a[6:8]
        else:
            current_quat = self.sim.data.mocap_quat.copy()
            new_quat = current_quat + a[3:7] # need some other way to limit this since identity quat is [1, 0, 0 ,0]
            self.sim.data.mocap_quat[:] = new_quat.copy()

            gripper_a = a[7:9]

        # copy gripper joints into empty action and run
        ja = np.zeros(9, dtype=np.float)

        # # position ctrl for gripper
        # ja[:2] = gripper_a
        # if not self.initializing:
        #     # act_mid and act_map are uniform so it's fine if gripper jnts are the first indices
        #     ja = self.act_mid + ja * self.act_amp  # mean center and scale

        # open/close saved if zero action for gripper
        ja[:2] = self.sim.data.qpos[7:9] + gripper_a # qpos indices from env.sim.model.actuator_trnid

        self.robot.step( # this will call mujoco_env.do_simulation, which should take the first model.nu == 2 indices of ja
            self, ja, step_duration=self.skip * self.model.opt.timestep, enforce_limits=False)

    def step_joints(self, a, b=None):
        a = np.clip(a, -1.0, 1.0)

        if not self.initializing:
            a = self.act_mid + a * self.act_amp  # mean center and scale
        # else:
        #     self.goal = self._get_task_goal()  # update goal if init

        self.robot.step(
            self, a, step_duration=self.skip * self.model.opt.timestep)

    def _get_obs(self):
        t, qp, qv, obj_qp, obj_qv = self.robot.get_obs(
            self, robot_noise_ratio=self.robot_noise_ratio)

        self.obs_dict = {}
        self.obs_dict['t'] = t
        self.obs_dict['qp'] = qp
        self.obs_dict['qv'] = qv
        self.obs_dict['obj_qp'] = obj_qp
        self.obs_dict['obj_qv'] = obj_qv
        # self.obs_dict['goal'] = self.goal
        # if self.goal_concat:
        #     return np.concatenate([self.obs_dict['qp'], self.obs_dict['obj_qp'], self.obs_dict['goal']])
        return np.concatenate([self.obs_dict['qp'], self.obs_dict['obj_qp']])

    def get_obs_mocap(self): # should in in global coordinates
        i = self.sim.model.site_name2id('end_effector')
        pos = self.sim.data.site_xpos[i, ...]
        xmat = self.sim.data.site_xmat[i, ...]
        xmat = xmat.reshape(3, 3)
        if self.mocap_use_euler:
            rot = mat2euler(xmat)
        else:
            rot = mat2quat(xmat)

        # i = self.sim.model.body_name2id('panda0_link7')
        # pos = self.sim.data.body_xpos[i, ...]
        # if self.mocap_use_euler:
        #     xmat = self.sim.data.body_xmat[i, ...]
        #     xmat = xmat.reshape(3, 3)
        #     rot = mat2euler(xmat)
        # else:
        #     rot = self.sim.data.xquat[i, ...]
        return np.concatenate([pos, rot])

    def reset_model(self):
        reset_pos = self.init_qpos[:].copy()
        reset_vel = self.init_qvel[:].copy()
        self.robot.reset(self, reset_pos, reset_vel)

        reset_mocap_welds(self.sim)
        self.sim.forward()
        for _ in range(10):
            self.sim.step()

        # self.goal = self._get_task_goal()  #sample a new goal on reset
        return self._get_obs()

    # def evaluate_success(self, paths):
    #     # score
    #     mean_score_per_rollout = np.zeros(shape=len(paths))
    #     for idx, path in enumerate(paths):
    #         mean_score_per_rollout[idx] = np.mean(path['env_infos']['score'])
    #     mean_score = np.mean(mean_score_per_rollout)

    #     # success percentage
    #     num_success = 0
    #     num_paths = len(paths)
    #     for path in paths:
    #         num_success += bool(path['env_infos']['rewards']['bonus'][-1])
    #     success_percentage = num_success * 100.0 / num_paths

    #     # fuse results
    #     return np.sign(mean_score) * (
    #         1e6 * round(success_percentage, 2) + abs(mean_score))

    def close_env(self):
        self.robot.close()

    # def set_goal(self, goal):
    #     self.goal = goal

    # def _get_task_goal(self):
    #     return self.goal

    # # Only include goal
    # @property
    # def goal_space(self):
    #     len_obs = self.observation_space.low.shape[0]
    #     env_lim = np.abs(self.observation_space.low[0])
    #     return spaces.Box(low=-env_lim, high=env_lim, shape=(len_obs//2,))

    def convert_to_active_observation(self, observation):
        return observation

class KitchenTaskRelaxV1(KitchenV0):
    """Kitchen environment with proper camera and goal setup"""

    def __init__(self, **kwargs):
        super(KitchenTaskRelaxV1, self).__init__(**kwargs)

    def _get_reward_n_score(self, obs_dict):
        reward_dict = {}
        reward_dict['true_reward'] = 0.
        reward_dict['bonus'] = 0.
        reward_dict['r_total'] = 0.
        score = 0.
        return reward_dict, score

    def render(self, mode='human', camera_id=None, height=1920, width=2560):
        if mode =='rgb_array':
            if camera_id is None: camera_id = self.camera_id
            camera = engine.MovableCamera(self.sim, height=height, width=width)
            camera.set_pose(**CAMERAS[camera_id])
            img = camera.render()
            return img
        else:
            super(KitchenTaskRelaxV1, self).render()
