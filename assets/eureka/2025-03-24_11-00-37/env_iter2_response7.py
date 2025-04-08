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
        
        # Weights
        velocity_weight = 1.0  # Balance with other rewards
        z_pos_weight = 0.5  # Fine as it’s a critical part but not too dominating
        orientation_weight = 0.2  # Reduced to not overshadow other components
        dof_limit_penalty_weight = 0.05  # Keep it minimal yet effective
        smoothness_weight = 0.01  # Significantly reduce to not overwhelm policy
        
        # Calculate desired components
        desired_velocity_x = 2.0
        desired_z_pos = 0.34
        
        # Velocity reward
        velocity_error = torch.abs(env.root_states[:, 7] - desired_velocity_x)
        velocity_reward = 1.0 - torch.tanh(velocity_error)  # Smoothed approach for velocity
        
        # Z Position reward
        z_pos_error = torch.abs(env.root_states[:, 2] - desired_z_pos)
        z_pos_reward = torch.exp(-z_pos_error)
        
        # Orientation reward - simpler calculation
        orientation_reward = torch.exp(-torch.norm(env.projected_gravity[:, 2:3], dim=1))  # Use z component of projected gravity
        
        # DOF Limit Penalty - dynamically scale based on current values
        dof_limit_penalty = torch.sum(torch.pow(torch.abs(env.dof_pos - env.default_dof_pos), 2), dim=1)  # Square to emphasize bigger deviations
        
        # Action Smoothness - less harsh, focusing on changes beyond a threshold
        action_smoothness_penalty = torch.mean(torch.pow(env.actions - env.last_actions, 2), dim=1)  # Use square to avoid negatives
        
        # Compute total reward with updated weights
        total_reward = (velocity_weight * velocity_reward) + \
                       (z_pos_weight * z_pos_reward) + \
                       (orientation_weight * orientation_reward) - \
                       (dof_limit_penalty_weight * dof_limit_penalty) - \
                       (smoothness_weight * action_smoothness_penalty)
        
        # Reward components dictionary for debugging
        reward_components = {
            'velocity_reward': velocity_weight * velocity_reward,
            'z_pos_reward': z_pos_weight * z_pos_reward,
            'orientation_reward': orientation_weight * orientation_reward,
            'dof_limit_penalty': -dof_limit_penalty_weight * dof_limit_penalty,
            'action_smoothness_penalty': -smoothness_weight * action_smoothness_penalty
        }
    
        return total_reward, reward_components
    
    # Success criteria as forward velocity
    def compute_success(self):
        target_velocity = 2.0
        lin_vel_error = torch.square(target_velocity - self.env.root_states[:, 7])
        return torch.exp(-lin_vel_error / 0.25)

