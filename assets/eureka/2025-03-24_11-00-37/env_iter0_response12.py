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
        env = self.env  # Do not skip this line. Afterwards, use env.{parameter_name} to access parameters of the environment.
        
        # Define weights for different parts of the reward
        velocity_weight = 1.0
        z_pos_weight = 0.5
        orientation_weight = 0.2
        dof_limit_penalty_weight = 0.1
        smoothness_weight = 0.2
    
        # Desired values
        desired_velocity_x = 2.0  # m/s
        desired_z_pos = 0.34  # meters
        desired_orientation = torch.tensor([0., 0., 0., 1.], device=env.device)  # WXYZ quaternion for no rotation
    
        # Calculate velocity error
        actual_velocity_x = env.root_states[:, 7]  # Global linear velocity x component
        velocity_error = torch.abs(actual_velocity_x - desired_velocity_x)
    
        # Calculate Z position error
        actual_z_pos = env.root_states[:, 2]  # Global Z position
        z_pos_error = torch.abs(actual_z_pos - desired_z_pos)
    
        # Calculate orientation error
        # Using quaternion dot product to check alignment, closer to 1 means better alignment
        orientation_error = 1 - torch.abs(torch.sum(env.base_quat * desired_orientation, dim=1))
    
        # DOF limit penalty
        dof_limit_penalty = torch.max(torch.zeros_like(env.dof_pos), torch.abs(env.dof_pos - env.default_dof_pos) - (env.dof_pos_limits[:, 1] - env.dof_pos_limits[:, 0]).unsqueeze(0)/2).sum(axis=1)
        
        # Action smoothness incentive
        action_smoothness = torch.mean(torch.abs(env.actions - env.last_actions), dim=1)
    
        # Calculate total reward
        reward = (velocity_weight / (1 + velocity_error)) + \
                 (z_pos_weight / (1 + z_pos_error)) + \
                 (orientation_weight * (1 - orientation_error)) - \
                 (dof_limit_penalty_weight * dof_limit_penalty) - \
                 (smoothness_weight * action_smoothness)
    
        # Construct detailed reward components for debugging/analysis
        reward_components = {
            'velocity': velocity_weight / (1 + velocity_error),
            'z_pos': z_pos_weight / (1 + z_pos_error),
            'orientation': orientation_weight * (1 - orientation_error),
            'dof_limit_penalty': -dof_limit_penalty_weight * dof_limit_penalty,
            'action_smoothness': -smoothness_weight * action_smoothness
        }
    
        return reward, reward_components
    
    # Success criteria as forward velocity
    def compute_success(self):
        target_velocity = 2.0
        lin_vel_error = torch.square(target_velocity - self.env.root_states[:, 7])
        return torch.exp(-lin_vel_error / 0.25)

