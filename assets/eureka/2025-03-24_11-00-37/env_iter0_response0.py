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
        env = self.env  # Do not skip this line.
    
        # Desired forward speed
        desired_speed = 2.0
        # Ideal height of the torso
        ideal_torso_height = 0.34
    
        # Calculate the difference in the desired speed and the actual speed in the x direction
        speed_diff = torch.abs(env.root_states[:, 7] - desired_speed)
        
        # Calculate the torso height deviation from the ideal position
        torso_height_deviation = torch.abs(env.root_states[:, 2] - ideal_torso_height)
        
        # Calculate orientation deviation from being perpendicular to gravity
        down_orientation = torch.tensor([0., 0., -1.], dtype=torch.float, device=env.device)
        orientation_deviation = torch.norm(env.projected_gravity - down_orientation.unsqueeze(0).expand_as(env.projected_gravity), dim=1)
        
        # Penalize deviations from desired DOF positions
        dof_deviation = torch.mean(torch.abs(env.dof_pos - env.default_dof_pos), dim=[1, 2])
        
        # Penalize high action rates
        action_diff = torch.mean(torch.abs(env.actions - env.last_actions), dim=1)
        
        # Normalize rewards components to be between 0 and 1, assuming reasonable maximums for each term
        speed_reward = torch.exp(-speed_diff)
        height_reward = torch.exp(-10 * torso_height_deviation)  # Assumes max deviation is around 0.1
        orientation_reward = torch.exp(-10 * orientation_deviation)  # Assumes max deviation is around 0.1 radians ~ 5.73 degrees
        smoothness_reward = 1.0 - torch.min(action_diff / 0.5, torch.tensor(1.0, device=env.device))  # Assumes max reasonable action_diff is 0.5
        dof_reward = torch.exp(-5 * dof_deviation)  # Assumes max deviation to be 0.2
        
        # Combine rewards, weighing components based on their importance
        total_reward = 0.3 * speed_reward + 0.2 * height_reward + 0.2 * orientation_reward + 0.2 * smoothness_reward + 0.1 * dof_reward
        
        # Encapsulate individual components for analysis
        reward_components = {
            'speed_reward': speed_reward, 
            'height_reward': height_reward, 
            'orientation_reward': orientation_reward, 
            'smoothness_reward': smoothness_reward, 
            'dof_reward': dof_reward
        }
        
        return total_reward, reward_components
    
    # Success criteria as forward velocity
    def compute_success(self):
        target_velocity = 2.0
        lin_vel_error = torch.square(target_velocity - self.env.root_states[:, 7])
        return torch.exp(-lin_vel_error / 0.25)

