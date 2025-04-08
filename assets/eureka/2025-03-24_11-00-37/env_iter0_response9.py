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
        
        # Desired forward velocity
        target_velocity_x = 2.0
        # Desired torso height
        target_height_z = 0.34
        
        # Calculate the forward velocity error
        forward_velocity_error = abs(env.root_states[:, 7] - target_velocity_x)
        
        # Calculate the height error
        height_error = abs(env.root_states[:, 2] - target_height_z)
        
        # Calculate the orientation error
        # Assuming that when the robot is perfectly upright, the projected_gravity would be [0, 0, 1]
        target_projected_gravity = torch.tensor([0., 0., 1.], device=env.device).repeat((env.num_envs, 1))
        orientation_error = torch.norm(env.projected_gravity - target_projected_gravity, dim=1)
        
        # Calculate leg smoothness error
        # Take the difference between current velocity and last timestep's velocity, summed across all DOFs
        dof_velocity_change = torch.sum(torch.abs(env.dof_vel - env.last_dof_vel), dim=1)
        
        # Calculate the proximity to DOF limits
        dof_limit_penalty = 0.0
        for i in range(env.num_dof):
            lower_limit = env.dof_pos_limits[i, 0]
            upper_limit = env.dof_pos_limits[i, 1]
            
            # Calculate distances to the lower and upper limits and penalize positions close to the limits
            dist_to_lower = torch.clamp(lower_limit - env.dof_pos[:, i], min=0)
            dist_to_upper = torch.clamp(env.dof_pos[:, i] - upper_limit, min=0)
            
            # Accumulate penalties for all DOFs
            dof_limit_penalty += dist_to_lower + dist_to_upper
        
        # Define weights for each component of the reward
        weights = {
            'velocity': -10.0,  # Penalize the deviation from the desired velocity
            'height': -10.0,    # Penalize the deviation from the desired height
            'orientation': -5.0, # Penalize the orientation error
            'smoothness': -0.1, # Penalize sudden changes in the velocity of the DOFs to encourage smoothness
            'dof_limits': -0.01, # Penalize the proximity to DOF limits to encourage safe movements
        }
        
        # Compute the total weighted reward
        reward = (
            weights['velocity'] * forward_velocity_error +
            weights['height'] * height_error +
            weights['orientation'] * orientation_error +
            weights['smoothness'] * dof_velocity_change +
            weights['dof_limits'] * dof_limit_penalty
        )
        
        # Construct the dictionary for individual reward components
        reward_components = {
            'velocity_error': forward_velocity_error,
            'height_error': height_error,
            'orientation_error': orientation_error,
            'dof_velocity_change': dof_velocity_change,
            'dof_limit_penalty': dof_limit_penalty
        }
        
        return reward, reward_components
    
    # Success criteria as forward velocity
    def compute_success(self):
        target_velocity = 2.0
        lin_vel_error = torch.square(target_velocity - self.env.root_states[:, 7])
        return torch.exp(-lin_vel_error / 0.25)

