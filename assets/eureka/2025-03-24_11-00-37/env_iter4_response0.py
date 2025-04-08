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
    
        # Adjusted weights for balance
        velocity_weight = 1.0
        z_pos_weight = 0.5
        orientation_weight = 0.2
        dof_limit_penalty_weight = 0.05  # Increased to give more focus on avoiding joint limits
        smoothness_weight = 0.02         # Reduced to mitigate overly harsh penalties
        
        # Objectives
        desired_velocity_x = 2.0
        desired_z_pos = 0.34
        desired_orientation = torch.tensor([0., 0., 0., 1.], device=env.device)
    
        # Calculate errors and penalties
        velocity_error = torch.abs(env.root_states[:, 7] - desired_velocity_x)
        z_pos_error = torch.abs(env.root_states[:, 2] - desired_z_pos)
        orientation_error = 1 - torch.abs(torch.sum(env.base_quat * desired_orientation, dim=1))
        
        dof_limit_penalty = env.dof_limit_penalty.sum(1)  # Assuming existing calculation logic
        action_smoothness = torch.mean(torch.abs(env.actions - env.last_actions), dim=1)
    
        # Define rewards
        velocity_score = velocity_weight * (1.0 - torch.tanh(velocity_error))
        z_pos_score = z_pos_weight * (1.0 - torch.tanh(z_pos_error))
        orientation_score = orientation_weight * (1.0 - torch.tanh(orientation_error))
        dof_limit_penalty_score = -dof_limit_penalty_weight * dof_limit_penalty
        smoothness_score = -smoothness_weight * action_smoothness
        
        # Composite total reward
        total_reward = velocity_score + z_pos_score + orientation_score + dof_limit_penalty_score + smoothness_score
    
        # Components dictionary for debugging and analysis
        reward_components = {
            'velocity': velocity_score,
            'z_pos': z_pos_score,
            'orientation': orientation_score,
            'dof_limit_penalty': dof_limit_penalty_score,
            'action_smoothness': smoothness_score
        }
    
        return total_reward, reward_components
    
    # Success criteria as forward velocity
    def compute_success(self):
        target_velocity = 2.0
        lin_vel_error = torch.square(target_velocity - self.env.root_states[:, 7])
        return torch.exp(-lin_vel_error / 0.25)

