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
    
        # Adjusting weights to rebalance based on the feedback
        velocity_weight = 0.5  # Adjusted for balance
        z_pos_weight = 0.2  # Further balancing
        orientation_weight = 0.1  # Fine-tuning based on learning progress
        dof_limit_penalty_weight = 0.1  # Adjusted to emphasize DOF limit adherence without dominating
        action_smoothness_weight = 0.05  # Slightly alleviate the penalty to encourage exploration
    
        desired_velocity_x = 2.0
        desired_z_pos = 0.34
        desired_orientation = torch.tensor([0., 0., 0., 1.], device=env.device).float()
    
        velocity_x = env.root_states[:, 7]
        z_pos = env.root_states[:, 2]
        orientation = env.base_quat
    
        # Calculate components
        velocity_error = torch.abs(velocity_x - desired_velocity_x)
        z_pos_error = torch.abs(z_pos - desired_z_pos)
        orientation_error = 1 - torch.abs(torch.sum(orientation * desired_orientation.unsqueeze(0), dim=-1))
    
        dof_limit_penalty = torch.mean(torch.clamp(torch.abs(env.dof_pos - env.default_dof_pos.unsqueeze(0)) - (env.dof_pos_limits[:, 1] - env.dof_pos_limits[:, 0])/2, min=0), dim=-1)
        action_smoothness = torch.mean(torch.abs(env.actions - env.last_actions), dim=1)
    
        # Reshaping reward components based on observed data and feedback
        velocity_reward = velocity_weight * (1 - torch.tanh(velocity_error))
        z_pos_reward = z_pos_weight * (1 - torch.tanh(z_pos_error))
        orientation_reward = orientation_weight * (1 - torch.tanh(orientation_error))
        dof_limit_penalty_reward = -dof_limit_penalty_weight * dof_limit_penalty
        action_smoothness_penalty = -action_smoothness_weight * action_smoothness
    
        # Total reward
        total_reward = velocity_reward + z_pos_reward + orientation_reward + dof_limit_penalty_reward + action_smoothness_penalty
    
        reward_components = {
            'velocity': velocity_reward,
            'z_pos': z_pos_reward,
            'orientation': orientation_reward,
            'dof_limit_penalty': dof_limit_penalty_reward,
            'action_smoothness': action_smoothness_penalty,
        }
    
        return total_reward, reward_components
    
    # Success criteria as forward velocity
    def compute_success(self):
        target_velocity = 2.0
        lin_vel_error = torch.square(target_velocity - self.env.root_states[:, 7])
        return torch.exp(-lin_vel_error / 0.25)

