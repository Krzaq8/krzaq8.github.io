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
        
        # Adjusted weights
        velocity_weight = 2.0
        z_pos_weight = 1.5
        orientation_weight = 0.1
        dof_limit_penalty_weight = 0.05
        smoothness_weight = 0.05
    
        desired_velocity_x = 2.0  # m/s
        desired_z_pos = 0.34  # meters
        desired_orientation = torch.tensor([0., 0., 0., 1.], device=env.device)
    
        velocity_error = torch.abs(env.root_states[:, 7] - desired_velocity_x)
        z_pos_error = torch.abs(env.root_states[:, 2] - desired_z_pos)
        orientation_error = 1 - torch.abs(torch.sum(env.base_quat * desired_orientation, dim=1))
    
        # Scaled DOF limit penalty for exceeding half the soft limit range
        dof_limit_penalty = torch.max(torch.zeros_like(env.dof_pos), torch.abs(env.dof_pos - env.default_dof_pos.unsqueeze(0)) - (env.dof_pos_limits[:, 1] - env.dof_pos_limits[:, 0]).unsqueeze(0)/4).sum(axis=1)
        
        # Smoother action change penalty
        action_smoothness = torch.mean(torch.abs(env.actions - env.last_actions), dim=1)
    
        # Compose total reward with adjusted scaling
        reward = (velocity_weight * torch.exp(-velocity_error)) + \
                 (z_pos_weight * torch.exp(-z_pos_error)) + \
                 (orientation_weight * (1 - orientation_error)) - \
                 (dof_limit_penalty_weight * dof_limit_penalty) - \
                 (smoothness_weight * action_smoothness)
    
        reward_components = {
            'velocity': velocity_weight * torch.exp(-velocity_error),
            'z_pos': z_pos_weight * torch.exp(-z_pos_error),
            'orientation': orientation_weight * (1 - orientation_error),
            'dof_limit_penalty': -dof_limit_penalty_weight * dof_limit_penalty,
            'action_smoothness': -smoothness_weight * action_smoothness
        }
    
        return reward, reward_components
    
    # Success criteria as forward velocity
    def compute_success(self):
        target_velocity = 2.0
        lin_vel_error = torch.square(target_velocity - self.env.root_states[:, 7])
        return torch.exp(-lin_vel_error / 0.25)

