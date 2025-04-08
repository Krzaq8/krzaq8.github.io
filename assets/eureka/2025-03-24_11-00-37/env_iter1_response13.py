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
        velocity_weight = 2.0  # Increased weight for velocity to make it a priority
        z_pos_weight = 1.0
        orientation_weight = 1.5  # Increase the weight to prioritize orientation alignment
        dof_limit_penalty_weight = 0.05  # Decrease the penalty to avoid over penalization
        smoothness_weight = 0.1  # Decrease to encourage smoother actions with less penalty
    
        # Desired values
        desired_velocity_x = 2.0  # m/s
        desired_z_pos = 0.34  # meters
        desired_orientation = torch.tensor([0., 0., 0., 1.], device=env.device)  # WXYZ quaternion for no rotation
    
        # Velocity reward: Using exponential to prioritize reaching the desired speed
        velocity_error = torch.abs(env.root_states[:, 7] - desired_velocity_x)
        velocity_reward = torch.exp(-velocity_error)
    
        # Z position reward: Exponential decay to prioritize precise altitude maintenance
        z_pos_reward = torch.exp(-torch.abs(env.root_states[:, 2] - desired_z_pos))
    
        # Orientation reward: Dot product for orientation alignment, scaled exponentially
        orientation_error = 1 - torch.abs(torch.sum(env.base_quat * desired_orientation, dim=1))
        orientation_reward = torch.exp(5 * orientation_error) - 1  # Exponential scaling to increase sensitivity
    
        # DOF limit penalty: Reduced scale
        dof_limit_exceeded = torch.max(torch.zeros_like(env.dof_pos), torch.abs(env.dof_pos - env.default_dof_pos) - (env.dof_pos_limits[:, 1] - env.dof_pos_limits[:, 0]).unsqueeze(0)/2)
        dof_limit_penalty = dof_limit_exceeded.sum(axis=1)
    
        # Action smoothness reward: Sigmoid to penalize large deviations less harshly
        action_deltas = torch.abs(env.actions - env.last_actions)
        smoothness_penalty = torch.sigmoid(action_deltas).mean(dim=1)
    
        # Calculate total reward
        total_reward = velocity_weight * velocity_reward + \
                       z_pos_weight * z_pos_reward + \
                       orientation_weight * orientation_reward - \
                       dof_limit_penalty_weight * dof_limit_penalty - \
                       smoothness_weight * smoothness_penalty
    
        # Construct detailed reward components for debugging/analysis
        reward_components = {
            'velocity': velocity_reward,
            'z_pos': z_pos_reward,
            'orientation': orientation_reward,
            'dof_limit_penalty': -dof_limit_penalty,
            'action_smoothness': -smoothness_penalty
        }
    
        return total_reward, reward_components
    
    # Success criteria as forward velocity
    def compute_success(self):
        target_velocity = 2.0
        lin_vel_error = torch.square(target_velocity - self.env.root_states[:, 7])
        return torch.exp(-lin_vel_error / 0.25)

