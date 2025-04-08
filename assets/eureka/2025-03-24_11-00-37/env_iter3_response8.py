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
        
        # Adjusted weights for balancing:
        velocity_weight = 0.8
        z_pos_weight = 0.2
        orientation_weight = 0.1
        dof_limit_penalty_weight = 0.05
        smoothness_weight = 0.05  # Reduced to decrease its impact
        
        # Objectives
        desired_velocity_x = 2.0
        desired_z_pos = 0.34
        desired_orientation = torch.tensor([0., 0., 0., 1.], device=env.device)
    
        # Calculations
        velocity_error = (env.root_states[:, 7] - desired_velocity_x).abs()
        z_pos_error = (env.root_states[:, 2] - desired_z_pos).abs()
        orientation_error = 1 - torch.abs(torch.sum(env.base_quat * desired_orientation, dim=1))
        dof_limit_penalty = torch.sum(torch.relu(torch.abs(env.dof_pos - env.default_dof_pos) - 0.5 * (env.dof_pos_limits[:, 1] - env.dof_pos_limits[:, 0]).unsqueeze(0)), axis=1)
        action_smoothness = torch.mean(torch.abs(env.actions - env.last_actions), dim=1)
    
        # Reward computation with re-scaled values
        velocity_score = velocity_weight * (1 - torch.tanh(velocity_error))
        z_pos_score = z_pos_weight * (1 - torch.tanh(z_pos_error))
        orientation_score = orientation_weight * (1 - torch.tanh(orientation_error))
        dof_limit_penalty_score = -dof_limit_penalty_weight * dof_limit_penalty
        smoothness_score = -smoothness_weight * action_smoothness
    
        total_reward = velocity_score + z_pos_score + orientation_score + dof_limit_penalty_score + smoothness_score
    
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

