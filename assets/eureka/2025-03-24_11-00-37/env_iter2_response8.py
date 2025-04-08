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
    
        # Configurable weights for reward components
        weight = {
            'velocity': 1.0,
            'z_pos': 0.1,
            'orientation': 0.05,
            'dof_limit_penalty': -0.01,
            'smoothness': -0.001
        }
    
        # Target values
        desired_velocity_x = 2.0
        desired_z_pos = 0.34
    
        # Component calculations
        velocity_error = torch.abs(env.root_states[:, 7] - desired_velocity_x)
        z_pos_error = torch.abs(env.root_states[:, 2] - desired_z_pos)
        orientation_error = 1 - torch.abs(torch.sum(env.base_quat * torch.tensor([0., 0., 0., 1.], device=env.device), dim=1))
        
        dof_limit_penalty = torch.sum(torch.relu(torch.abs(env.dof_pos - env.default_dof_pos) - (env.dof_pos_limits[:, 1] - env.dof_pos_limits[:, 0])/2), axis=1)
        
        action_smoothness = torch.mean(torch.abs(env.actions - env.last_actions), dim=1)
    
        # Reward components rescaled and weighted
        reward_velocity = -velocity_error * weight['velocity']
        reward_z_pos = -z_pos_error * weight['z_pos']
        reward_orientation = -orientation_error * weight['orientation']
        penalty_dof_limit = dof_limit_penalty * weight['dof_limit_penalty']
        penalty_smoothness = action_smoothness * weight['smoothness']
    
        # Final reward calculation
        total_reward = reward_velocity + \
                       reward_z_pos + \
                       reward_orientation + \
                       penalty_dof_limit + \
                       penalty_smoothness
    
        reward_components = {
            'velocity': reward_velocity,
            'z_pos': reward_z_pos,
            'orientation': reward_orientation,
            'dof_limit_penalty': penalty_dof_limit,
            'action_smoothness': penalty_smoothness
        }
    
        return total_reward, reward_components
    
    # Success criteria as forward velocity
    def compute_success(self):
        target_velocity = 2.0
        lin_vel_error = torch.square(target_velocity - self.env.root_states[:, 7])
        return torch.exp(-lin_vel_error / 0.25)

