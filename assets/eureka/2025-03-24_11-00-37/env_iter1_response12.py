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
        
        # Updated weights for different parts of the reward
        velocity_weight = 2.0  # Increase emphasis on reaching desired velocity.
        z_pos_weight = 0.3  # Decrease based on performance; it's learning well but don't want it overemphasized.
        orientation_weight = 0.1  # Decrease weight due to high scores and less room for improvement.
        dof_limit_penalty_weight = 0.05  # Decrease harshness since it's not a primary focus.
        smoothness_weight = 0.05  # Significantly reduce to encourage more varied actions.
        
        desired_velocity_x = 2.0
        desired_z_pos = 0.34
        desired_orientation = torch.tensor([0., 0., 0., 1.], device=env.device)
    
        velocity_error = torch.abs(env.root_states[:, 7] - desired_velocity_x)
        z_pos_error = torch.abs(env.root_states[:, 2] - desired_z_pos)
        orientation_error = 1 - torch.abs(torch.sum(env.base_quat * desired_orientation, dim=1))
        
        dof_limit_penalty = torch.sum(torch.relu(torch.abs(env.dof_pos - env.default_dof_pos) - (env.dof_pos_limits[:, 1] - env.dof_pos_limits[:, 0]).unsqueeze(0)/2), axis=1)
        
        action_smoothness = torch.mean(torch.abs(env.actions - env.last_actions), dim=1)
    
        # Normalizing components to a similar scale
        reward = (velocity_weight * torch.exp(-velocity_error)) + \
                 (z_pos_weight * torch.exp(-z_pos_error)) + \
                 (orientation_weight * torch.exp(-orientation_error)) - \
                 (dof_limit_penalty_weight * dof_limit_penalty) - \
                 (smoothness_weight * action_smoothness)
    
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

