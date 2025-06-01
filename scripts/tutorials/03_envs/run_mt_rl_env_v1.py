# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to run the RL environment for the multi-task RL env.

.. code-block:: bash

    ./isaaclab.sh -p scripts/tutorials/03_envs/run_mt_rl_env.py --num_envs 32

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on running the cartpole RL environment.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

from isaaclab.envs import ManagerBasedMTRLEnv

from isaaclab_tasks.manager_based.multitask.reach.reach_env_cfg import MTReachEnvCfg_Homogeneous as MTReachEnvCfg


def main():
    """Main function."""
    # create environment configuration
    env_cfg = MTReachEnvCfg()
    env_cfg.num_envs_per_task = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    # setup RL environment
    env = ManagerBasedMTRLEnv(cfg=env_cfg)

    # simulate physics
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 300 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")
            # sample random actions
            actions = []
            for task_env in env.envs.values():
                joint_efforts = torch.randn_like(task_env.action_manager.action)
                actions.append(joint_efforts)

            actions = torch.cat(actions, dim=0)  # concatenate actions for all tasks
            # step the environment
            obs, rew, terminated, truncated, info = env.step(actions)
            # print current observations

            if count % 100 == 0:
                print(f"[INFO]: Current observations (step {count}):")
                obs_MND = obs["policy"].view(env.cfg.num_multi_task_envs, env.cfg.num_envs_per_task, -1)
                for idx in range(env.cfg.num_multi_task_envs):
                    task_name = info[idx]["task_name"]
                    task_obs = obs_MND[idx]
                    print(f"\t[Env {task_name}]: {task_obs[0][0].item()}")

            # update counter
            count += 1

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
