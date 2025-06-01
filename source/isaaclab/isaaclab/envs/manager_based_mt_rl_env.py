import gymnasium as gym
from .manager_based_rl_env import ManagerBasedRLEnv
import numpy as np
import torch

from isaaclab.sim import SimulationContext
from isaaclab.sim import SimulationCfg

from isaacsim.core.cloner import Cloner
from isaaclab.utils.timer import Timer
from .utils.mtrl import (
    get_environment_position_offsets,
    wrap_observation_space,
    concatenate_observations,
    wrap_info,
)
from .manager_based_rl_env_cfg import ManagerBasedRLEnvCfg
from .manager_based_mt_rl_env_cfg import TaskConfigs, MultiTaskRLEnvConfig
from typing import Dict, Union, List, Sequence, Any
from .ui import ViewportCameraController
import isaacsim.core.utils.torch as torch_utils


class ManagerBasedMTRLEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        cfg: MultiTaskRLEnvConfig,
        is_vector_env: bool = True,
        device="cuda:0",
        render_mode=None,
    ):
        self.is_vector_env = is_vector_env

        self.cfg = cfg

        self.envs: Dict[str, ManagerBasedRLEnv] = {}

        sim_cfg = SimulationCfg()
        sim_cfg.device = device

        self.sim: SimulationContext = SimulationContext(sim_cfg)
        self._sim_step_counter = 0  
        
        self.sim.set_camera_view(cfg.viewer.eye, cfg.viewer.lookat)
        self.task_configs = self.extract_tasks()

        env_position_offsets = get_environment_position_offsets(
            num_clones_per_env=cfg.num_envs_per_task,
            num_environments=len(self.task_configs),
            clone_spacing=cfg.envs_spacing,
            environment_spacing= cfg.task_spacing,
        )
        
        env_prim_paths = []

        for task_idx, (task_name, task_cfg) in enumerate(self.task_configs.items()):
            
            rl_env_cfg = ManagerBasedRLEnvCfg(**(self.cfg.base_dataclass_fields()), **(task_cfg.__dict__))
            
            # filter out the common scene elements, such as ground, etc, which belong to
            if task_idx > 0:
                world_elements = [attr for attr in rl_env_cfg.scene.__dict__.keys() if attr.startswith("world_")]
                for attr in world_elements:
                    delattr(rl_env_cfg.scene, attr)

            
            rl_env_cfg.scene.env_prefix = task_name
            rl_env_cfg.scene.pos_offset = env_position_offsets[task_idx].tolist()
            rl_env_cfg.scene.num_envs = cfg.num_envs_per_task
            rl_env_cfg.scene.env_spacing = cfg.envs_spacing
            
            # note (mt-isaac): collision filtering is handled outside the loop
            rl_env_cfg.scene.filter_collisions = False
            self.envs[task_name] = ManagerBasedRLEnv(
                rl_env_cfg, sim=self.sim, render_mode=render_mode
            )

            env_prim_paths.extend(self.envs[task_name].scene.env_prim_paths)


        example_env = list(self.envs.values())[0]
        self.example_env = example_env  

        # remove global collisions between the tasks and environments within tasks
        cloner = Cloner()
        cloner.filter_collisions(
            physicsscene_path=example_env.scene.physics_scene_path,
            collision_root_path="/World/collisions_tasks",
            prim_paths=env_prim_paths,
            global_paths=example_env.scene.global_prim_paths,
        )

        # set up camera viewport controller
        # viewport is not available in other rendering modes so the function will throw a warning
        # FIXME: This needs to be fixed in the future when we unify the UI functionalities even for
        # non-rendering modes.
        if self.sim.render_mode >= self.sim.RenderMode.PARTIAL_RENDERING:
            self.viewport_camera_controller = ViewportCameraController(self, self.cfg.viewer)
        else:
            self.viewport_camera_controller = None


        # start the sim
        print(
            "[INFO]: Starting the simulation. This may take a few seconds. Please wait..."
        )
        with Timer("[INFO]: Time taken for simulation start"):
            self.sim.reset()
            
        for task_name, env in self.envs.items():
            # add timeline event to load managers
            print(f"[INFO]: Loading environment {task_name}...")
            env.scene.update(dt=env.physics_dt)
            env.load_managers()
            print(f"[INFO]: Environment {env.scene.env_ns} loaded.")

        # ui_window_class_type = ManagerBasedRLEnvWindow
        # FIXME: currently provides visualization only for the first environment. 
        # Potentially requires a multi-task version of BaseEnvWindow class.
        if self.sim.has_gui() and self.cfg.ui_window_class_type is not None:
            example_env.setup_manager_visualizers()
            example_env._window = self.cfg.ui_window_class_type(
                example_env, window_name="IsaacLab"
            )

        self.set_observation_action_spaces()
        self.render_mode = render_mode

    def extract_tasks(self):
        task_configs = {}
        for task_name, task_cfg in self.cfg.__dict__.items():
            if isinstance(task_cfg, TaskConfigs):
                task_configs[task_name] = task_cfg
        return task_configs

    def set_observation_action_spaces(self):
        
        # two possibilities exist:
        # 1. the envs allow for a common observation and action space.
        # 2. the envs have different observation and action spaces.
        
        # note (mt-isaac): Approach 1 is efficient for batch processing and downstream processing. 
        # note (mt-isaac): we provide multi-task env for both cases, but downstream processing 
        # (with RL algorithms) is only supported for (1). Hence, we do not need to handle (2) here.

        
        # observation space here accounts for the number of paralle environments per task
        self.observation_space = self.example_env.observation_space  
        self.action_space = self.example_env.action_space
                       
        if self.cfg.append_task_id:
            self.observation_space = wrap_observation_space(
                self.observation_space,
                addon_space=gym.spaces.Dict(
                    spaces={
                        "task_id": gym.spaces.Box(
                            low=0,
                            high=len(self.envs) - 1,
                            shape=(self.cfg.num_envs_per_task, 1),
                            dtype=np.int32,
                        )
                    }
                ),
            )

    @property
    def num_envs(self) -> int:
        return self.cfg.num_envs_per_task

    @property
    def device(self):
        return self.sim.cfg.device

    @property
    def physics_dt(self) -> float:
        """The physics time-step (in s).

        This is the lowest time-decimation at which the simulation is happening.
        """
        return self.sim.cfg.dt

    @property
    def step_dt(self) -> float:
        """The environment stepping time-step (in s).

        This is the time-step at which the environment steps forward.
        """
        return self.sim.cfg.dt * self.cfg.decimation

    @property
    def max_episode_length_s(self):
        return max([env.max_episode_length for env in self.envs.values()])

    @property
    def max_episode_length(self):
        return max([env.max_episode_length for env in self.envs.values()])

    def step(self, action: Union[torch.Tensor, List[torch.Tensor]]):  # -> VecEnvStepReturn:
        """Step all the task environments with the given action."""

        if not isinstance(action, list):
            action_dim = sum(self.example_env.action_manager.action_term_dim)
            action = action.view(self.cfg.num_multi_task_envs, self.cfg.num_envs_per_task, action_dim)

        # assign actions to the environments
        for task_idx, (task_name, task_env) in enumerate(self.envs.items()):
            task_env.action_manager.process_action(action[task_idx])

        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # apply the 
        for _ in range(self.cfg.decimation):

            self._sim_step_counter += 1

            # set actions into buffers
            for env in self.envs.values():
                env.action_manager.apply_action()
                env.scene.write_data_to_sim()

            # step all the environments together
            self.sim.step(render=False)

            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()

            for env in self.envs.values():
                env.scene.update(dt=self.physics_dt)

        # post-step:
        
        is_reset = False        
        for task_idx, (task_name, task_env) in enumerate(self.envs.items()):
            task_env.curriculum_manager.compute(env_ids=None)

            task_env.episode_length_buf += 1
            task_env.common_step_counter += 1  # total step (common for all envs)
            # -- check terminations
            task_env.reset_buf = task_env.termination_manager.compute()
            task_env.reset_terminated = task_env.termination_manager.terminated
            task_env.reset_time_outs = task_env.termination_manager.time_outs
            # -- reward computation
            task_env.reward_buf = task_env.reward_manager.compute(dt=self.step_dt)
            # -- reset envs that terminated/timed-out and log the episode information
            
            reset_env_ids = task_env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
            if len(reset_env_ids) > 0:
                task_env._reset_idx(reset_env_ids)
                # -- update command
                task_env.scene.write_data_to_sim()
            
            is_reset = is_reset or (len(reset_env_ids) > 0)
                
            # -- update command
            task_env.command_manager.compute(dt=self.step_dt)
            # -- step interval events
            if "interval" in task_env.event_manager.available_modes:
                task_env.event_manager.apply(mode="interval", dt=self.step_dt)
            # -- compute observations
            # note: done after reset to get the correct observations for reset envs
            task_env.obs_buf = task_env.observation_manager.compute()
            
        if is_reset:
            self.sim.forward()
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()
                
        # concatenate the observations, rewards, resets and extras
        
        reward_buf = [task_env.reward_buf for task_env in self.envs.values()]
        reset_terminated = [
            task_env.reset_terminated for task_env in self.envs.values()
        ]
        reset_time_outs = [
            task_env.reset_time_outs for task_env in self.envs.values()
        ]
        extras = [
            wrap_info(task_env.extras, task_name)
            for task_name, task_env in self.envs.items()
        ]
        
        obs_buf = {task_name: task_env.obs_buf for task_name, task_env in self.envs.items()}    
        
        if self.cfg.concatenate_step_results:
            # assumes that all environments have the same observation space
            self.reward_buf = torch.cat(reward_buf, dim=0)
            self.reset_terminated = torch.cat(reset_terminated, dim=0)
            self.reset_time_outs = torch.cat(reset_time_outs, dim=0)
                    
            self.obs_buf = concatenate_observations(list(obs_buf.values()))
            
        else:
            self.reward_buf = torch.stack(reward_buf, dim=0)
            self.reset_terminated = torch.stack(reset_terminated, dim=0)
            self.reset_time_outs = torch.stack(reset_time_outs, dim=0)
            self.obs_buf = obs_buf
                        
        return (
            self.obs_buf,
            self.reward_buf,
            self.reset_terminated,
            self.reset_time_outs,
            extras,
        )
        

    def _reset_idx(self, task_name, env_ids: Sequence[int]):
        """Reset environments based on specified indices.

        Args:
            env_ids: List of environment ids which must be reset
        """
        task_env = self.envs[task_name]

        # update the curriculum for environments that need a reset
        task_env.curriculum_manager.compute(env_ids=env_ids)
        # reset the internal buffers of the scene elements
        task_env.scene.reset(env_ids)
        # apply events such as randomizations for environments that need a reset
        if "reset" in task_env.event_manager.available_modes:
            env_step_count = self._sim_step_counter // self.cfg.decimation
            task_env.event_manager.apply(mode="reset", env_ids=env_ids, global_env_step_count=env_step_count)

        # iterate over all managers and reset them
        # this returns a dictionary of information which is stored in the extras
        # note: This is order-sensitive! Certain things need be reset before others.
        task_env.extras["log"] = dict()
        # -- observation manager
        info = task_env.observation_manager.reset(env_ids)
        task_env.extras["log"].update(info)
        # -- action manager
        info = task_env.action_manager.reset(env_ids)
        task_env.extras["log"].update(info)
        # -- rewards manager
        info = task_env.reward_manager.reset(env_ids)
        task_env.extras["log"].update(info)
        # -- curriculum manager
        info = task_env.curriculum_manager.reset(env_ids)
        task_env.extras["log"].update(info)
        # -- command manager
        info = task_env.command_manager.reset(env_ids)
        task_env.extras["log"].update(info)
        # -- event manager
        info = task_env.event_manager.reset(env_ids)
        task_env.extras["log"].update(info)
        # -- termination manager
        info = task_env.termination_manager.reset(env_ids)
        task_env.extras["log"].update(info)
        # -- recorder manager
        # info = self.recorder_manager.reset(env_ids)
        # self.extras["log"].update(info)

        # reset the episode length buffer
        task_env.episode_length_buf[env_ids] = 0

    @staticmethod
    def seed(seed: int = -1) -> int:
        """Set the seed for the environment.

        Args:
            seed: The seed for random generator. Defaults to -1.

        Returns:
            The seed used for random generator.
        """
        # set seed for replicator
        try:
            import omni.replicator.core as rep

            rep.set_global_seed(seed)
        except ModuleNotFoundError:
            pass
        # set seed for torch and other libraries
        return torch_utils.set_seed(seed)
        
    def reset(
        self, seed: int | None = None, env_ids: list[Sequence[int]] | None = None, options: dict[str, Any] | None = None
    ):        
        """Resets the specified environments and returns observations.

        This function calls the :meth:`_reset_idx` function to reset the specified environments.
        However, certain operations, such as procedural terrain generation, that happened during initialization
        are not repeated.

        Args:
            seed: The seed to use for randomization. Defaults to None, in which case the seed is not set.
            env_ids: The environment ids to reset. Defaults to None, in which case all environments are reset.
            options: Additional information to specify how the environment is reset. Defaults to None.

                Note:
                    This argument is used for compatibility with Gymnasium environment definition.

        Returns:
            A tuple containing the observations and extras.
        """
        if env_ids is None:
            env_ids = [None] * len(self.envs)
            
        for idx in range(len(self.envs)):
            if env_ids[idx] is None:
                env_ids[idx] = torch.arange(self.cfg.num_envs_per_task, dtype=torch.int64, device=self.device)
        
        if seed is not None:
            self.seed(seed)
            
        for task_idx, (task_name, task_env) in enumerate(self.envs.items()):
            self._reset_idx(task_name=task_name, env_ids=env_ids[task_idx])
            task_env.scene.write_data_to_sim()
            
        self.sim.forward()
        
        self.obs_buf = {}
        for task_name, task_env in self.envs.items():   
            task_env.obs_buf = task_env.observation_manager.compute()
            self.obs_buf[task_name] = task_env.obs_buf
            
        if self.cfg.concatenate_step_results:
            # concatenate the observations
            self.obs_buf = concatenate_observations(list(self.obs_buf.values()))
            
        infos_list = [wrap_info(task_env.extras, task_name) for task_name, task_env in self.envs.items()]
        
        return self.obs_buf, infos_list

    def close(self):
        for env in self.envs.values():
            env.close()

    def render(self, recompute=False):        
        # run a rendering step of the simulator
        # if we have rtx sensors, we do not need to render again sin
        if not self.sim.has_rtx_sensors() and not recompute:
            self.sim.render()
        # decide the rendering mode
        if self.render_mode == "human" or self.render_mode is None:
            return None
        elif self.render_mode == "rgb_array":
            # check that if any render could have happened
            if self.sim.render_mode.value < self.sim.RenderMode.PARTIAL_RENDERING.value:
                raise RuntimeError(
                    f"Cannot render '{self.render_mode}' when the simulation render mode is"
                    f" '{self.sim.render_mode.name}'. Please set the simulation render mode to:"
                    f"'{self.sim.RenderMode.PARTIAL_RENDERING.name}' or '{self.sim.RenderMode.FULL_RENDERING.name}'."
                    " If running headless, make sure --enable_cameras is set."
                )
            # create the annotator if it does not exist
            if not hasattr(self, "_rgb_annotator"):
                import omni.replicator.core as rep

                # create render product
                self._render_product = rep.create.render_product(
                    self.cfg.viewer.cam_prim_path, self.cfg.viewer.resolution
                )
                # create rgb annotator -- used to read data from the render product
                self._rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
                self._rgb_annotator.attach([self._render_product])
            # obtain the rgb data
            rgb_data = self._rgb_annotator.get_data()
            # convert to numpy array
            rgb_data = np.frombuffer(rgb_data, dtype=np.uint8).reshape(*rgb_data.shape)
            # return the rgb data
            # note: initially the renerer is warming up and returns empty data
            if rgb_data.size == 0:
                print("[WARN] Empty RGB data returned.")
                return np.zeros((self.cfg.viewer.resolution[1], self.cfg.viewer.resolution[0], 3), dtype=np.uint8)
            else:
                return rgb_data[:, :, :3]
        else:
            raise NotImplementedError(
                f"Render mode '{self.render_mode}' is not supported. Please use: {self.metadata['render_modes']}."
            )        