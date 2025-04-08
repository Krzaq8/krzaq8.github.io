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
        
        # Adjusted weights to fine-tune the balance among components
        velocity_weight = 1.2
        z_pos_weight = 0.1  # Less variance in z position suggests slight downscaling could be beneficial
        orientation_weight = 0.2  # Given its performance, a slight upscale might encourage better alignment
        dof_limit_penalty_weight = 0.015  # Minor increase to ensure DOF limits are respected more
        smoothness_weight = 0.005  # Significantly reduced to enable smoother action transitions
        
        # Desired objectives
        desired_velocity_x = 2.0
        desired_z_pos = 0.34
        desired_orientation = torch.tensor([0., 0., 0., 1.], device=env.device)  # Desired orientation quaternion
        
        # Calculations
        velocity_error = torch.abs(env.root_states[:, 7] - desired_velocity_x)
        z_pos_error = torch.abs(env.root_states[:, 2] - desired_z_pos)
        orientation_error = 1 - torch.abs(torch.sum(env.base_quat * desired_orientation, dim=1))  # Dot product for similarity
        
        dof_limit_penalty = torch.sum(torch.relu(torch.abs(env.dof_pos - env.default_dof_pos) - (env.dof_pos_limits[:, 1] - env.dof_pos_limits[:, 0]).unsqueeze(0)/2), dim=1)
        
        action_smoothness = torch.mean(torch.abs(env.actions - env.last_actions), dim=1)
    
        # Reward components recalibrated
        velocity_score = velocity_weight * (1 - torch.exp(-velocity_error * 0.5))  # Using exponential decay for focus on closer targets
        z_pos_score = z_pos_weight * (1 - torch.exp(-z_pos_error))  # Encouraging more precise positioning
        orientation_score = orientation_weight * (1 - orientation_error)  # Direct use of error
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

