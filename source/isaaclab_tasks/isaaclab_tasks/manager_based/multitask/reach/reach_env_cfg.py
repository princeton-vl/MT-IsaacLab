# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, Author: Meenal Parakh
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Reference: source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/reach

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import TaskConfigs
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import (
    ActionsCfg,
    CommandsCfg,
    CurriculumCfg,
    EventCfg,
    ObservationsCfg,
    RewardsCfg,
    TerminationsCfg,
)

##
# Scene definition
##


@configclass
class ReachSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with a robotic arm."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.size = (250, 150)  # large enough for all

        # note (mt-isaac): Define common scene elements (e.g., ground) with a `world_` prefix to ensure they are
        # initialized only once across all tasks.
        self.world_ground = AssetBaseCfg(
            prim_path="/World/defaultGroundPlane",
            spawn=ground_cfg,
            init_state=AssetBaseCfg.InitialStateCfg(pos=(10.0, 0.0, -1.05)),
            collision_group=-1,
        )

        # note (mt-isaac): ENV_REGEX_NS is different for each task, so the assets is automatically spawned
        # without name collisions between tasks.
        self.table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.55, 0.0, 0.0), rot=(0.70711, 0.0, 0.0, 0.70711)),
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
