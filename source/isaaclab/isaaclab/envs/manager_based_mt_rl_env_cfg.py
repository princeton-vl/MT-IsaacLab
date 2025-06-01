
from .ui import ManagerBasedRLEnvWindow
from dataclasses import MISSING

from isaaclab.sim import SimulationCfg
from .common import ViewerCfg
from isaaclab.devices.openxr import XrCfg

from isaaclab.utils.configclass import configclass
from isaaclab.managers import RecorderManagerBaseCfg as DefaultEmptyRecorderManagerCfg
from isaaclab.scene import InteractiveSceneCfg
from .manager_based_env_cfg import DefaultEventManagerCfg

def wrap_info(info, env_name):
    info["env_name"] = env_name
    return info

# common settings
@configclass
class MultiTaskRLEnvConfig:
    
    # simulation settings
    viewer: ViewerCfg = ViewerCfg()
    """Viewer configuration. Default is ViewerCfg()."""

    sim: SimulationCfg = SimulationCfg()
    """Physics simulation configuration. Default is SimulationCfg()."""
    
    decimation: int = MISSING
    """Number of control action updates @ sim dt per policy dt.

    For instance, if the simulation dt is 0.01s and the policy dt is 0.1s, then the decimation is 10.
    This means that the control action is updated every 10 simulation steps.
    """
    # ui settings
    ui_window_class_type: type | None = ManagerBasedRLEnvWindow


    rerender_on_reset: bool = False
    """Whether a render step is performed again after at least one environment has been reset.
    Defaults to False, which means no render step will be performed after reset.

    * When this is False, data collected from sensors after performing reset will be stale and will not reflect the
      latest states in simulation caused by the reset.
    * When this is True, an extra render step will be performed to update the sensor data
      to reflect the latest states from the reset. This comes at a cost of performance as an additional render
      step will be performed after each time an environment is reset.

    """

    wait_for_textures: bool = True
    """True to wait for assets to be loaded completely, False otherwise. Defaults to True."""

    xr: XrCfg | None = None
    """Configuration for viewing and interacting with the environment through an XR device."""


    # Camera configuration
    camera_eye_position = (0.0, 0.5, 1.0)
    camera_target_position = (0.0, 0.0, 0.5)
    
    # Number of different multi-task environments
    num_multi_task_envs = 1
    task_spacing = 10.0

    # Environment configuration
    num_envs_per_task = 2
    envs_spacing = 5.0
    
    # Observation 
    append_task_id: bool = False
    """If True, appends the task ID to the observation space.
       Otherwise, the task ID can also be explicitly provided through the observations in each environment.
    """
    
    concatenate_step_results: bool = True  
    """If True, concatenates the observations, rewards, terminated, and timeouts from all environments.
       Otherwise, `step` returns a dict of observations, and others concatenated on a new axis.
    """
    
    
    @property
    def base_dataclass_fields(self) -> dict:
        """Return the basic data fields in common with ManagerBasedEnvCfg."""
        return {
            "viewer": self.viewer,
            "sim": self.sim,
            "decimation": self.decimation,
            "ui_window_class_type": self.ui_window_class_type,
            "rerender_on_reset": self.rerender_on_reset,
            "wait_for_textures": self.wait_for_textures,
            "xr": self.xr,
        }
    
    
@configclass
class TaskConfigs:
    # environment settings
    episode_length_s: float = MISSING

    scene: InteractiveSceneCfg = MISSING
    """Scene settings.

    Please refer to the :class:`isaaclab.scene.InteractiveSceneCfg` class for more details.
    """

    recorders: object = DefaultEmptyRecorderManagerCfg()
    """Recorder settings. Defaults to recording nothing.

    Please refer to the :class:`isaaclab.managers.RecorderManager` class for more details.
    """

    observations: object = MISSING
    """Observation space settings.

    Please refer to the :class:`isaaclab.managers.ObservationManager` class for more details.
    """

    actions: object = MISSING
    """Action space settings.

    Please refer to the :class:`isaaclab.managers.ActionManager` class for more details.
    """

    events: object = DefaultEventManagerCfg()
    """Event settings. Defaults to the basic configuration that resets the scene to its default state.

    Please refer to the :class:`isaaclab.managers.EventManager` class for more details.
    """

    # environment settings
    rewards: object = MISSING
    """Reward settings.

    Please refer to the :class:`isaaclab.managers.RewardManager` class for more details.
    """

    terminations: object = MISSING
    """Termination settings.

    Please refer to the :class:`isaaclab.managers.TerminationManager` class for more details.
    """

    curriculum: object | None = None
    """Curriculum settings. Defaults to None, in which case no curriculum is applied.

    Please refer to the :class:`isaaclab.managers.CurriculumManager` class for more details.
    """

    commands: object | None = None
    """Command settings. Defaults to None, in which case no commands are generated.

    Please refer to the :class:`isaaclab.managers.CommandManager` class for more details.
    """
