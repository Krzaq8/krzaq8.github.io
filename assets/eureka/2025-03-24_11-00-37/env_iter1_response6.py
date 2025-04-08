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
        
        # Adjust weightings
        velocity_weight = 2.0
        z_pos_weight = 1.0
        orientation_weight = 1.0
        dof_limit_penalty_weight = 0.05
        smoothness_weight = 0.05  # Reduced to lessen penalty
    
        # Desired orientation quaternion adjustment
        desired_orientation = torch.tensor([0., 0., 0., 1.], device=env.device)  
    
        # Velocity error with log scale
        velocity_error = torch.log(1 + torch.abs(env.root_states[:, 7] - 2.0))
        
        # Z position error remains the same
        z_pos_error = torch.abs(env.root_states[:, 2] - 0.34)  
        
        # Orientation error adjustment
        orientation_error = 1 - torch.abs(torch.sum(env.base_quat * desired_orientation, dim=1)) 
        
        # DOF limit penalty calculation adjustment
        dof_limit_penalty = torch.sum(torch.clamp(torch.abs(env.dof_pos - env.default_dof_pos) - (env.dof_pos_limits[:, 1] - env.dof_pos_limits[:, 0]).unsqueeze(0)/4, min=0), dim=1)
        
        # Smoothness penalty adjusted
        action_smoothness = torch.mean(torch.abs(env.actions - env.last_actions), dim=1)
    
        # Calculating total reward
        reward = velocity_weight * torch.exp(-velocity_error) + \
                 z_pos_weight * torch.exp(-z_pos_error) + \
                 orientation_weight * torch.exp(-orientation_error) - \
                 dof_limit_penalty_weight * dof_limit_penalty - \
                 smoothness_weight * action_smoothness
    
        reward_components = {
            'velocity': velocity_weight * torch.exp(-velocity_error),
            'z_pos': z_pos_weight * torch.exp(-z_pos_error),
            'orientation': orientation_weight * torch.exp(-orientation_error),
            'dof_limit_penalty': -dof_limit_penalty_weight * dof_limit_penalty,
            'action_smoothness': -smoothness_weight * action_smoothness
        }
    
        return reward, reward_components
    
    # Success criteria as forward velocity
    def compute_success(self):
        target_velocity = 2.0
        lin_vel_error = torch.square(target_velocity - self.env.root_states[:, 7])
        return torch.exp(-lin_vel_error / 0.25)

