import torch
import numpy as np
from forward_locomotion.go1_gym.utils.math_utils import quat_apply_yaw, wrap_to_pi, get_scale_shift
from isaacgym.torch_utils import *

class EurekaReward():
    def __init__(self, env):
        self.env = env

    def load_env(self, env):
        self.env = env

    def compute_reward(self):
        env = self.env  # Do not skip this line. Afterwards, use env.{parameter_name} to access parameters of the environment.
    
        # Define reward components
        desired_velocity = 2.0
        desired_base_height = 0.34
    
        # Calculate velocity error
        forward_velocity = env.root_states[:, 7]
        velocity_error = torch.abs(forward_velocity - desired_velocity)
    
        # Calculate base height error
        base_height = env.root_states[:, 2]  # Z position is at index 2
        base_height_error = torch.abs(base_height - desired_base_height)
    
        # Calculate orientation error
        # Assuming that being "perpendicular to gravity" means the robot's up vector (in its local frame) should align with the global z-axis
        projected_gravity_norm = torch.norm(env.projected_gravity, dim=1)
        orientation_error = 1.0 - projected_gravity_norm
    
        # Calculate DOF position limit penalty
        dof_pos_error = torch.zeros_like(env.dof_pos)
        for i in range(env.num_dof):
            lower_limit = env.dof_pos_limits[i, 0]
            upper_limit = env.dof_pos_limits[i, 1]
            dof_pos_error[:, i] = torch.maximum(env.dof_pos[:, i] - upper_limit, lower_limit - env.dof_pos[:, i])
        dof_pos_error = torch.clamp(dof_pos_error, min=0.0)
        dof_pos_penalty = dof_pos_error.sum(dim=1)
    
        # Calculate action rate penalty to encourage minimal action rate for steadiness
        action_rate_penalty = torch.sum(torch.abs(env.actions - env.last_actions), dim=1)
    
        # Weights for each component of the reward
        weights = {
            "velocity_error": -10.0,
            "base_height_error": -1.0,
            "orientation_error": -10.0,
            "dof_pos_penalty": -0.1,
            "action_rate_penalty": -0.01,
        }
    
        # Compute the total reward
        total_reward = (
            weights["velocity_error"] * velocity_error +
            weights["base_height_error"] * base_height_error +
            weights["orientation_error"] * orientation_error +
            weights["dof_pos_penalty"] * dof_pos_penalty +
            weights["action_rate_penalty"] * action_rate_penalty
        )
    
        # Bundle reward components for potential debugging
        reward_components = {
            "velocity_error": velocity_error,
            "base_height_error": base_height_error,
            "orientation_error": orientation_error,
            "dof_pos_penalty": dof_pos_penalty,
            "action_rate_penalty": action_rate_penalty,
        }
    
        return total_reward, reward_components
    
    # Success criteria as forward velocity
    def compute_success(self):
        target_velocity = 2.0
        lin_vel_error = torch.square(target_velocity - self.env.root_states[:, 7])
        return torch.exp(-lin_vel_error / 0.25)

