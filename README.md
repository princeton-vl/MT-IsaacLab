# Multi-Task Setup in Isaac Lab

This repository overlays minimal changes to the [Isaac Lab](https://isaac-sim.github.io/IsaacLab) framework to support multi-task environments, and training multi-task RL policies.

Please refer to the [Isaac Lab documentation](https://isaac-sim.github.io/IsaacLab) for more information on the framework, its features, and how to get started.


## Installation
The installation process is identical to that of the original Isaac Lab repository, except you should clone this repository instead of the official one.

## Multi-Task Environments
We extend the manager-based environments in Isaac Lab to support multi-task setups. This repository introduces a dedicated multi-task environment class called `ManagerBasedMTRLEnv`, following the `ManagerBasedRLEnv` class, along with simple usage examples.

**Key differences in the multi-task environment:**

- **Configuration Structure:**
   - Instead of a single `ManagerBasedRLEnvCfg`, the multi-task environment uses two configuration types:
      - `ManagerBasedMTRLEnvCfg`: Contains parameters shared across all tasks.
      - `TaskConfigs`: Holds task-specific settings.

- **TaskConfigs Details:**
   - The `TaskConfigs` class includes fields such as:
      - `episode_length_s`
      - `scene`
      - `observations`
      - `actions`
      - `commands`
      - `rewards`
      - `terminations`
      - `events`
      - `curriculum`

- **ManagerBasedMTRLEnvCfg Details:**
   - All the remaining parameters from `ManagerBasedRLEnvCfg` are defined in `ManagerBasedMTRLEnvCfg` and are shared across all tasks.
   - Additional fields include:
      - `num_multi_task_envs`: Number of tasks.
      - `task_spacing`: Spacing between tasks.
      - `num_envs_per_task`: Number of environments for each task. Overrides the `num_envs` in `InteractiveSceneCfg`.
      - `envs_spacing`: Spacing between environments within each task. Overrides the `env_spacing` in `InteractiveSceneCfg`
      - `append_task_id`: Whether to append the task ID (0 to `num_tasks - 1`) to the observations.
      - `concatenate_step_results`: Whether to concatenate step results across tasks. Possible only for homogeneous tasks, that allow for a common observation and action space. Other fields, such as `rewards`, `terminations` etc are also concatenated if set to `True`.

- **Adding Tasks:**
   - To add new tasks, simply include their corresponding `TaskConfigs` in the `ManagerBasedMTRLEnvCfg`.

- **Example Config:**
   - See [`source/isaaclab_tasks/isaaclab_tasks/manager_based/multitask/reach`](source/isaaclab_tasks/isaaclab_tasks/manager_based/multitask/reach) for example multi-task environments. The following config is for `mt_franka_ur10`, a heterogeneous multi-task environment with two tasks: `franka` and `ur10`.

      ```python
      @configclass
      class MTReachEnvCfg_Heterogeneous(ManagerBasedMTRLEnvCfg):
         """Configuration for the reach end-effector pose tracking environment."""

         franka: TaskConfigs = FrankaReachEnvCfg()
         ur10: TaskConfigs = UR10ReachEnvCfg()

         def __post_init__(self):
            """Post initialization."""

            # general settings
            self.decimation = 2
            self.num_multi_task_envs = 2
            self.task_spacing = 5.0
            self.num_envs_per_task = 64
            self.envs_spacing = 2.5
            self.append_task_id = True
            self.concatenate_step_results = False

            # viewer settings
            self.viewer.eye = (3.5, 3.5, 3.5)

            # simulation settings
            self.sim.render_interval = self.decimation
            self.sim.dt = 1.0 / 60.0
      ```

   - **Note:** The task config attribute names (such as `franka1`, `franka2`, `ur10`) will be treated as task names, and are used at various places:
      - In the prim paths of environments, for example `/World/<task_name>_envs/envs_*`
      - In the `envs` field of `ManagerBasedMTRLEnv`, which is a dictionary mapping task names to their respective environments.
      - When `concatenate_step_results` is `False`, the observations are dictionaries mapping task names to their respective observations.

- **Example Scripts:**
   - **Homogeneous Multi-Task Environment:**
     See [`scripts/tutorials/03_envs/run_mt_rl_env_v1.py`](scripts/tutorials/03_envs/run_mt_rl_env_v1.py) for an example using two homogeneous tasks, `franka1` and `franka2`. These tasks are functionally identical, but any tasks that share the same observation and action spaces can be used.
   - **Heterogeneous Multi-Task Environment:**
     See [`scripts/tutorials/03_envs/run_mt_rl_env_v2.py`](scripts/tutorials/03_envs/run_mt_rl_env_v2.py) for an example using heterogeneous tasks, `franka` and `ur10`, which have different observation and action spaces.
   - **Multi-Task RL Training for Homogeneous Environment:**
     ```shell
     ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Reach-MTFranka-v0
     ```

## License
This repository builds on top of the [Isaac Lab](https://isaac-sim.github.io/IsaacLab) framework, which is released under the [BSD-3 License](LICENSE). The `isaaclab_mimic` extension and its corresponding standalone scripts are released under [Apache 2.0](LICENSE-mimic). The license files of its dependencies and assets are present in the [`docs/licenses`](docs/licenses) directory.
