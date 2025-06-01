# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, MultiTaskRLEnvConfig, TaskConfigs
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
import isaacsim.core.utils.prims as prim_utils
from isaaclab_assets import FRANKA_PANDA_CFG, UR10_CFG  # isort: skip
import math

from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import (
    CommandsCfg,
    ActionsCfg,
    ObservationsCfg,
    EventCfg,
    RewardsCfg,
    TerminationsCfg,
    CurriculumCfg,
)
import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp

##
# Scene definition
##


# load different robot hands in each of the task.
# the observation space and action space needs to be adjusted accrodingly.
# here we use a common observation space and action space for all the robots.

# common elements, such as ground, which are shared across all tasks, should be defined starting with 
# `world_` prefix to initialize them only once. 

@configclass
class ReachSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with a robotic arm."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.size = (250, 150)  # large enough for all environments
        self.world_ground = AssetBaseCfg(
            prim_path="/World/defaultGroundPlane",
            spawn=ground_cfg,
            init_state=AssetBaseCfg.InitialStateCfg(pos=(10.0, 0.0, 0.0)),
            collision_group=-1,
        )

        # note (mt-isaac): ENV_REGEX_NS is different for each task, so the assets is automatically spawned
        #                  without name collisions between tasks.
        self.table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(0.55, 0.0, 0.0), rot=(0.70711, 0.0, 0.0, 0.70711)
            ),
        )

        # lights
        self.light = AssetBaseCfg(
            # prim_path="/World/light",
            prim_path="{ENV_REGEX_NS}/Light",
            spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
        )

        # robots
        self.robot: ArticulationCfg = MISSING

##
# Environment configuration
##
@configclass
class ReachEnvCfg(TaskConfigs):
    episode_length_s: float = 12.0
    scene: ReachSceneCfg = ReachSceneCfg()
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    

# copied from source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/reach/config/ur_10/joint_pos_env_cfg.py
@configclass
class UR10ReachEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        # switch robot to ur10
        self.scene.robot = UR10_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # override events
        self.events.reset_robot_joints.params["position_range"] = (0.75, 1.25)
        # override rewards
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["ee_link"]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["ee_link"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["ee_link"]
        # override actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True
        )
        # override command generator body
        # end-effector is along x-direction
        self.commands.ee_pose.body_name = "ee_link"
        self.commands.ee_pose.ranges.pitch = (math.pi / 2, math.pi / 2)


# copied from source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/reach/config/franka/joint_pos_env_cfg.py
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


@configclass
class MTReachEnvCfg(MultiTaskRLEnvConfig):
    """Configuration for the reach end-effector pose tracking environment."""

    franka: TaskConfigs = FrankaReachEnvCfg()
    ur10: TaskConfigs = UR10ReachEnvCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.num_multi_task_envs = 2
        self.task_spacing = 5.0
        self.num_envs_per_task = 128
        self.envs_spacing = 2.5
        self.append_task_id = True
        self.concatenate_step_results = False
        
        self.sim.render_interval = self.decimation
        self.episode_length_s = 12.0
        self.viewer.eye = (3.5, 3.5, 3.5)
        # simulation settings
        self.sim.dt = 1.0 / 60.0

