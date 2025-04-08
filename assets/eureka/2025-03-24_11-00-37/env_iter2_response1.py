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
        env = self.env
    
        # Fine-tuned weights
        velocity_weight = 1.0  # Balanced to emphasize reaching and maintaining desired velocity
        z_pos_weight = 0.5  # Adjusted based on variability and importance in task
        orientation_weight = 0.5  # Adjusted to ensure balance with other components
        eof_limit_penalty_weight = 0.05  # Reduced severity to not overshadow other components
        smoothness_weight = 0.025  # Lowered to reduce over-penalization of necessary exploration
      
        # Converting orientation goal to tensor format for calculations
        desired_orientation = torch.tensor([0., 0., 0., 1.], device=env.device).unsqueeze(0)
    
        # Calculating component rewards and penalties
        velocity_error = torch.abs(env.root_states[:, 7] - 2.0)
        z_pos_error = torch.abs(env.root_states[:, 2] - 0.34)
        orientation_error = 1 - torch.abs(torch.sum(env.base_quat * desired_orientation, dim=1))
        
        dof_limit_penalty = torch.sum(torch.relu(torch.abs(env.dof_pos - env.default_dof_pos) - (env.dof_pos_limits[:, 1] - env.dof_pos_limits[:, 0]).unsqueeze(0) / 2), axis=1)
    
        action_smoothness = torch.mean(torch.abs(env.actions - env.last_actions), dim=1)
    
        # Reward calculations with updated formulas for normalization and balance
        velocity_reward = torch.exp(-velocity_weight * velocity_error)
        z_pos_reward = torch.exp(-z_pos_weight * z_pos_error)
        orientation_reward = torch.exp(-orientation_weight * orientation_error)
        
        dof_limit_penalty = -eof_limit_penalty_weight * dof_limit_penalty
        action_smoothness_penalty = -smoothness_weight * action_smoothness
    
        total_reward = velocity_reward + z_pos_reward + orientation_reward + dof_limit_penalty + action_smoothness_penalty
    
        reward_components = {
            'velocity': velocity_reward,
            'z_pos': z_pos_reward,
            'orientation': orientation_reward,
            'dof_limit_penalty': dof_limit_penalty,
            'action_smoothness': action_smoothness_penalty,
        }
    
        return total_reward, reward_components
    
    # Success criteria as forward velocity
    def compute_success(self):
        target_velocity = 2.0
        lin_vel_error = torch.square(target_velocity - self.env.root_states[:, 7])
        return torch.exp(-lin_vel_error / 0.25)

