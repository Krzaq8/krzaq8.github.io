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
    
        # Re-adjusted weights for different parts of the reward
        velocity_weight = 1.5  # Focus on reaching desired velocity but balanced against other objectives.
        z_pos_weight = 1.0  # Ensure Z position is accurately maintained.
        orientation_weight = 1.0  # Important for stability; keep this incentivized.
        dof_limit_penalty_weight = 0.5  # Increase slightly to encourage a wider range of movement within limits.
        smoothness_weight = 0.25  # Less penalty on action smoothness to encourage exploration of action space.
        
        desired_velocity_x = 2.0
        desired_z_pos = 0.34
        desired_orientation = torch.tensor([0., 0., 0., 1.], device=env.device)
    
        # Calculating errors
        velocity_error = torch.abs(env.root_states[:, 7] - desired_velocity_x)
        z_pos_error = torch.abs(env.root_states[:, 2] - desired_z_pos)
        orientation_error = 1 - torch.abs(torch.sum(env.base_quat * desired_orientation, dim=1))
        
        # Adjusting penalties and rewards with new scales or methods
        dof_limit_penalty = torch.mean(torch.square(torch.clamp(torch.abs(env.dof_pos - env.default_dof_pos) - (env.dof_pos_limits[:, 1] - env.dof_pos_limits[:, 0]).unsqueeze(0) / 2, min=0)), axis=1)
        
        action_smoothness = torch.mean(torch.square(env.actions - env.last_actions), dim=1)
    
        # Compute total reward with re-adjusted components
        reward = (velocity_weight * (1.0 - torch.tanh(velocity_error))) + \
                 (z_pos_weight * (1.0 - torch.tanh(z_pos_error))) + \
                 (orientation_weight * (1.0 - torch.tanh(orientation_error))) - \
                 (dof_limit_penalty_weight * dof_limit_penalty) - \
                 (smoothness_weight * action_smoothness)
    
        # Dictionary of each individual reward component for tracking and debugging
        reward_components = {
            'velocity': velocity_weight * (1.0 - torch.tanh(velocity_error)),
            'z_pos': z_pos_weight * (1.0 - torch.tanh(z_pos_error)),
            'orientation': orientation_weight * (1.0 - torch.tanh(orientation_error)),
            'dof_limit_penalty': -dof_limit_penalty_weight * dof_limit_penalty,
            'action_smoothness': -smoothness_weight * action_smoothness,
        }
    
        return reward, reward_components
    
    # Success criteria as forward velocity
    def compute_success(self):
        target_velocity = 2.0
        lin_vel_error = torch.square(target_velocity - self.env.root_states[:, 7])
        return torch.exp(-lin_vel_error / 0.25)

