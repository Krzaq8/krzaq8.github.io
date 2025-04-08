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
        
        # Adjusting weights based on the feedback analysis
        velocity_weight = 1.0  # Ensure it remains an essential component but not overly dominant
        z_pos_weight = 0.5  # Important for balance but should not overshadow velocity achievement
        orientation_weight = 0.5  # Important but adjusted to balance with other rewards
        dof_limit_penalty_weight = 0.2  # Increase its influence slightly for better motion quality
        smoothness_weight = 0.05  # Significantly reduce harshness
        
        desired_velocity_x = 2.0
        desired_z_pos = 0.34
        desired_orientation = torch.tensor([0., 0., 0., 1.], device=env.device)
    
        velocity_error = torch.abs(env.root_states[:, 7] - desired_velocity_x)
        z_pos_error = torch.abs(env.root_states[:, 2] - desired_z_pos)
        orientation_error = 1 - torch.abs(torch.sum(env.base_quat * desired_orientation, dim=1))
        
        dof_limit_penalty = torch.sum(torch.relu(torch.abs(env.dof_pos - env.default_dof_pos) - (env.dof_pos_limits[:, 1] - env.dof_pos_limits[:, 0]).unsqueeze(0)/2), axis=1)
        
        action_diff = env.actions - env.last_actions
        action_smoothness = torch.mean(torch.abs(action_diff), dim=1)
    
        # Reward calculations with adjusted scales and penalties
        reward_velocity = torch.exp(-0.5 * velocity_error)
        reward_z_pos = torch.exp(-10.0 * z_pos_error)
        reward_orientation = torch.exp(-10.0 * orientation_error)
        penalty_dof_limit = dof_limit_penalty_weight * dof_limit_penalty
        penalty_smoothness = smoothness_weight * action_smoothness
    
        total_reward = velocity_weight * reward_velocity + \
                       z_pos_weight * reward_z_pos + \
                       orientation_weight * reward_orientation - \
                       penalty_dof_limit - \
                       penalty_smoothness
    
        reward_components = {
            'velocity': reward_velocity,
            'z_pos': reward_z_pos,
            'orientation': reward_orientation,
            'dof_limit_penalty': -penalty_dof_limit,
            'action_smoothness': -penalty_smoothness
        }
    
        return total_reward, reward_components
    
    # Success criteria as forward velocity
    def compute_success(self):
        target_velocity = 2.0
        lin_vel_error = torch.square(target_velocity - self.env.root_states[:, 7])
        return torch.exp(-lin_vel_error / 0.25)

