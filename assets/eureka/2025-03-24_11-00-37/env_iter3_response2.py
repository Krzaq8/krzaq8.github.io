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
        
        # Constants for desired conditions
        desired_velocity_x = 2.0
        desired_z_pos = 0.34
    
        # Reward weights
        velocity_weight = 0.5
        z_pos_weight = 0.3
        orientation_weight = 0.1
        dof_limit_penalty_weight = 0.05
        action_smoothness_weight = 0.05
        
        # Compute errors
        velocity_error = torch.abs(env.root_states[:, 7] - desired_velocity_x)
        z_pos_error = torch.abs(env.root_states[:, 2] - desired_z_pos)
        orientation_error = 1.0 - torch.abs(torch.sum(env.base_quat * torch.tensor([0., 0., 0., 1.], device=env.device), dim=1))
        dof_limit_penalty = torch.sum(torch.relu(torch.abs(env.dof_pos - env.default_dof_pos) - (env.dof_pos_limits[:, 1] - env.dof_pos_limits[:, 0]).unsqueeze(0)/2), dim=1)
        action_smoothness = torch.mean(torch.abs(env.actions - env.last_actions), dim=1)
        
        # Scale and combine individual rewards
        velocity_reward = velocity_weight * (1 - torch.exp(-velocity_error))
        z_pos_reward = z_pos_weight * (1 - torch.exp(-z_pos_error))
        orientation_reward = orientation_weight * (1 - orientation_error)
        dof_limit_penalty_reward = -dof_limit_penalty_weight * dof_limit_penalty
        action_smoothness_reward = -action_smoothness_weight * action_smoothness
        
        # Total reward
        total_reward = velocity_reward + z_pos_reward + orientation_reward + dof_limit_penalty_reward + action_smoothness_reward
    
        # Breakdown for debugging/analysis
        reward_components = {
            'velocity': velocity_reward,
            'z_pos': z_pos_reward,
            'orientation': orientation_reward,
            'dof_limit_penalty': dof_limit_penalty_reward,
            'action_smoothness': action_smoothness_reward
        }
    
        return total_reward, reward_components
    
    # Success criteria as forward velocity
    def compute_success(self):
        target_velocity = 2.0
        lin_vel_error = torch.square(target_velocity - self.env.root_states[:, 7])
        return torch.exp(-lin_vel_error / 0.25)

