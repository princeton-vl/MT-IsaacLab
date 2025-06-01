# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab_tasks.manager_based.multitask.reach.reach_env_cfg import ReachEnvCfg
from isaaclab.envs import ManagerBasedMTRLEnvCfg, TaskConfigs

##
# Pre-defined configs
##
from isaaclab_assets import FRANKA_PANDA_CFG  # isort: skip

@configclass
class FrankaReachEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        # switch robot to franka
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # override rewards
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["panda_hand"]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["panda_hand"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["panda_hand"]

        # override actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        )
        # override command generator body
        # end-effector is along z-direction
        self.commands.ee_pose.body_name = "panda_hand"
        self.commands.ee_pose.ranges.pitch = (math.pi, math.pi)

##
# Environment configuration
##
@configclass
class MTReachEnvCfg_Homogeneous(ManagerBasedMTRLEnvCfg):
    """Configuration for the reach end-effector pose tracking environment."""

    franka1: TaskConfigs = FrankaReachEnvCfg()
    franka2: TaskConfigs = FrankaReachEnvCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.num_multi_task_envs = 2
        self.task_spacing = 5.0
        self.num_envs_per_task = 128
        self.envs_spacing = 2.5
        self.append_task_id = False
        self.concatenate_step_results = True

        self.sim.render_interval = self.decimation
        self.episode_length_s = 12.0
        self.viewer.eye = (3.5, 3.5, 3.5)
        # simulation settings
        self.sim.dt = 1.0 / 60.0