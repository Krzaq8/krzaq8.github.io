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
    
        # Adjusted weights for different parts of the reward
        velocity_weight = 1.0
        z_pos_weight = 0.5
        orientation_weight = 0.15  # Slightly decrease to balance with other components
        dof_limit_penalty_weight = 0.3  # Increasing impact
        smoothness_weight = 0.05  # Decreasing impact to encourage more exploration
    
        # Desired values
        desired_velocity_x = 2.0  # Target forward velocity in m/s
        desired_z_pos = 0.34  # Target torso height in meters
        desired_orientation = torch.tensor([0., 0., 0., 1.], device=env.device)  # Ideal orientation quaternion
    
        # Compute velocity reward
        actual_velocity_x = env.root_states[:, 7]
        velocity_error = torch.abs(actual_velocity_x - desired_velocity_x)
        velocity_reward = 1.0 - torch.tanh(velocity_error)
    
        # Compute z position reward
        actual_z_pos = env.root_states[:, 2]
        z_pos_error = torch.abs(actual_z_pos - desired_z_pos)
        z_pos_reward = 1.0 - torch.tanh(z_pos_error)
    
        # Compute orientation reward
        orientation_error = 1 - torch.abs(torch.sum(env.base_quat * desired_orientation, dim=1))
        orientation_reward = orientation_error  # Directly use the error as reward
    
        # Compute DOF limit penalty
        dof_limit_penalty = torch.sum(torch.relu(torch.abs(env.dof_pos - env.default_dof_pos) - (env.dof_pos_limits[:,1] - env.dof_pos_limits[:, 0]).unsqueeze(0) / 2), dim=1)
        dof_limit_penalty_reward = -dof_limit_penalty_weight * torch.tanh(dof_limit_penalty)
    
        # Compute action smoothness reward
        action_diff = torch.abs(env.actions - env.last_actions)
        action_smoothness_reward = -smoothness_weight * torch.mean(action_diff, dim=1)
    
        # Calculate the total reward
        total_reward = velocity_weight * velocity_reward + \
                       z_pos_weight * z_pos_reward + \
                       orientation_weight * orientation_reward + \
                       dof_limit_penalty_reward + \
                       action_smoothness_reward
    
        reward_components = {
            'velocity_reward': velocity_reward,
            'z_pos_reward': z_pos_reward,
            'orientation_reward': orientation_reward,
            'dof_limit_penalty_reward': dof_limit_penalty_reward,
            'action_smoothness_reward': action_smoothness_reward
        }
    
        return total_reward, reward_components
    
    # Success criteria as forward velocity
    def compute_success(self):
        target_velocity = 2.0
        lin_vel_error = torch.square(target_velocity - self.env.root_states[:, 7])
        return torch.exp(-lin_vel_error / 0.25)

