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
        env = self.env  # Important: use env.{parameter_name} to access parameters.
        
        # Define reward components
        velocity_reward = 0
        height_reward = 0
        orientation_reward = 0
        smoothness_reward = 0
        dof_limit_penalty = 0
        
        # Velocity reward: agent should run forward at 2m/s. Using L2 norm to penalize deviation.
        forward_velocity = env.base_lin_vel[:, 0]  # Positive X direction velocity
        target_velocity = 2.0  # m/s
        velocity_error = forward_velocity - target_velocity
        velocity_reward = torch.exp(-torch.square(velocity_error))  # Max of 1 when error is 0
        
        # Height reward: keep the torso height around 0.34m.
        target_height = 0.34  # meters
        height_error = torch.abs(env.root_states[:, 2] - target_height)
        height_reward = torch.exp(-10.0 * height_error)  # Sharper penalty for deviation
        
        # Orientation reward: penalize deviation from being perpendicular to gravity
        # A perfect orientation would mean the robot's projected gravity is [0, 0, 1] in local frame
        target_orientation = torch.tensor([0., 0., 1.], device=env.device).repeat((env.num_envs, 1))
        orientation_error = torch.norm(env.projected_gravity - target_orientation, dim=1)
        orientation_reward = torch.exp(-10.0 * orientation_error)
        
        # Smoothness reward: minimize the change in action values to encourage smoother movements
        action_diff = torch.norm(env.actions - env.last_actions, dim=1)
        smoothness_reward = torch.exp(-0.1 * action_diff)
        
        # DOF Limit penalty: penalize when joints are near their limits to avoid hitting the limits
        pos_limits = env.dof_pos_limits.unsqueeze(0).repeat((env.num_envs, 1, 1))
        # Calculate how close each joint is to its limits. 0 means at the center, 1 means at the limit.
        proximity_to_limit = torch.abs((env.dof_pos - pos_limits[:, :, 0]) / (pos_limits[:, :, 1] - pos_limits[:, :, 0]) - 0.5) * 2
        dof_limit_penalty = torch.mean(torch.max(proximity_to_limit - 0.8, torch.tensor(0.0, device=env.device)), dim=1)  # Only penalize if proximity > 0.8
        
        # Calculating total reward
        total_reward = velocity_reward + height_reward + orientation_reward + smoothness_reward - dof_limit_penalty
        
        # Normalize total_reward to encourage all objectives equally
        total_reward = total_reward / 5.0
        
        # Return the total reward and a dictionary of individual reward components
        reward_components = {
            'velocity_reward': velocity_reward,
            'height_reward': height_reward,
            'orientation_reward': orientation_reward,
            'smoothness_reward': smoothness_reward,
            'dof_limit_penalty': dof_limit_penalty
        }
        
        return total_reward, reward_components
    
    # Success criteria as forward velocity
    def compute_success(self):
        target_velocity = 2.0
        lin_vel_error = torch.square(target_velocity - self.env.root_states[:, 7])
        return torch.exp(-lin_vel_error / 0.25)

