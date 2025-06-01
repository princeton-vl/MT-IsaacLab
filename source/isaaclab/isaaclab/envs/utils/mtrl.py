# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, Author: Meenal Parakh
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym
import numpy as np
import torch
from gymnasium.error import CustomSpaceError


def compute_grid_center_offset(
    square_dimensions: tuple[float, float],
    grid_shape: tuple[int, int],
    spacing: tuple[float, float],
) -> np.ndarray:
    """
    Compute the offset required to center a grid of squares at the origin.

    Args:
        square_dimensions (Tuple[float, float]): Width and height of each square.
        grid_shape (Tuple[int, int]): Number of rows and columns in the grid.
        spacing (Tuple[float, float]): Horizontal and vertical spacing between squares.

    Returns:
        np.ndarray: Offset (x, y) to center the grid at the origin.
    """
    width, height = square_dimensions
    rows, cols = grid_shape
    dx, dy = spacing

    total_width = (cols - 1) * (width + dx)
    total_height = (rows - 1) * (height + dy)

    offset_x = -total_width / 2.0
    offset_y = -total_height / 2.0

    return np.array([offset_x, offset_y])


def generate_centered_grid_positions(
    square_dimensions: tuple[float, float],
    total_squares: int,
    spacing: tuple[float, float],
) -> list[np.ndarray]:
    """
    Generate positions for placing squares in a grid centered at the origin.

    Args:
        square_dimensions (Tuple[float, float]): Width and height of each square.
        total_squares (int): Total number of squares to arrange.
        spacing (Tuple[float, float]): Horizontal and vertical spacing between squares.

    Returns:
        List[np.ndarray]: List of 3D center positions (x, y, 0) for each square.
    """
    rows = int(np.ceil(np.sqrt(total_squares)))
    cols = int(np.ceil(total_squares / rows))
    grid_shape = (rows, cols)

    offset = compute_grid_center_offset(square_dimensions, grid_shape, spacing)

    centers: list[np.ndarray] = []

    for row in range(rows):
        for col in range(cols):
            if len(centers) >= total_squares:
                break
            x = col * (square_dimensions[0] + spacing[0]) + offset[0]
            y = row * (square_dimensions[1] + spacing[1]) + offset[1]
            centers.append(np.array([x, y, 0]))

    return centers


def get_environment_position_offsets(
    num_clones_per_env: int, num_environments: int, clone_spacing: float, environment_spacing: float = 5.0
) -> list[np.ndarray]:
    """
    Compute center positions for placing multiple environment instances in a grid.

    Each environment is a grid of clones. This function first computes the overall
    size of one such clone grid and then arranges the environments with proper spacing
    such that they are centered at the origin.

    Args:
        num_clones_per_env (int): Number of clones in a single environment.
        num_environments (int): Total number of environment instances to arrange.
        clone_spacing (float): Distance between adjacent clones in a grid.
        environment_spacing (float): Distance between adjacent environments. Defaults to 5.0.

    Returns:
        List[np.ndarray]: A list of 3D position offsets for each environment center.
    """
    clones_per_row = int(np.sqrt(num_clones_per_env))
    num_rows = np.ceil(num_clones_per_env / clones_per_row)
    num_cols = np.ceil(num_clones_per_env / num_rows)

    # Compute span (width and height) of a single clone grid
    grid_width = clone_spacing * (num_cols - 1)
    grid_height = clone_spacing * (num_rows - 1)

    environment_positions = generate_centered_grid_positions(
        square_dimensions=(grid_width, grid_height),
        total_squares=num_environments,
        spacing=(environment_spacing, environment_spacing),
    )

    return environment_positions


class ObservationDict(dict):
    def float(self):
        for k, v in self.items():
            self[k] = v.float()
        return self

    def copy_(self, other):
        for k in self.keys():
            self[k].copy_(other[k])
        return self

    @property
    def shape(self):
        return


def wrap_observation_space(observation_space, addon_space):
    """Wrap the observation space with an additional space.

    Args:
        observation_space: The original observation space.
        addon_space: The additional space to be added.

    Returns:
        A new observation space that combines the original and the additional space.
    """
    # If observation_space is a Dict and contains a "policy" key which is also a Dict, merge addon_space into it
    if isinstance(observation_space, gym.spaces.Dict):
        if "policy" in observation_space.spaces and isinstance(observation_space.spaces["policy"], gym.spaces.Dict):
            # Merge addon_space into the nested "policy" dict
            merged_policy_spaces = dict(observation_space.spaces["policy"].spaces)
            for k, v in addon_space.spaces.items():
                if k in merged_policy_spaces:
                    raise CustomSpaceError(
                        f"Key '{k}' already exists in the observation space. "
                        "Please ensure that the keys in the addon space are unique."
                    )
                merged_policy_spaces[k] = v
            # Rebuild the observation space with the merged policy dict
            new_spaces = dict(observation_space.spaces)
            new_spaces["policy"] = gym.spaces.Dict(merged_policy_spaces)
            return gym.spaces.Dict(new_spaces)
        else:
            # Merge addon_space at the top level
            merged_spaces = dict(observation_space.spaces)
            for k, v in addon_space.spaces.items():
                if k in merged_spaces:
                    raise CustomSpaceError(
                        f"Key '{k}' already exists in the observation space. "
                        "Please ensure that the keys in the addon space are unique."
                    )
                merged_spaces[k] = v
            return gym.spaces.Dict(merged_spaces)
    else:
        # Wrap the original space and addon_space into a new Dict under "policy"
        return gym.spaces.Dict(spaces={"observation": observation_space, **addon_space.spaces})


def concatenate_observations(obs_list: list):
    # If all observations are dictionaries, concatenate them key by key.
    # Handle recursive dictionaries.
    if not obs_list:
        return obs_list
    if isinstance(obs_list[0], dict):
        result = ObservationDict()
        keys = obs_list[0].keys()
        for k in keys:
            values = [obs[k] for obs in obs_list]
            result[k] = concatenate_observations(values)
        return result
    elif isinstance(obs_list[0], torch.Tensor):
        return torch.cat(obs_list, dim=0)
    elif isinstance(obs_list[0], np.ndarray):
        return np.concatenate(obs_list, axis=0)
    else:
        # fallback: return as list
        return obs_list


def wrap_info(info, env_name):
    info["task_name"] = env_name
    return info
