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
    
        # Ideal forward speed
        ideal_speed = 2.0
        # Extract the linear velocity of the base
        lin_vel_x = env.root_states[:, 7]
    
        # Calculate the deviation from the ideal speed
        speed_deviation = torch.abs(lin_vel_x - ideal_speed)
    
        # Velocity reward - we want the robot to maintain a speed of 2.0 m/s
        velocity_reward = torch.exp(-speed_deviation)
    
        # Height from the ground stability - goal is to keep the z position of the torso at 0.34
        ideal_height = 0.34
        base_height = env.root_states[:, 2]  # z position is the third value in root_states
        height_deviation = torch.abs(base_height - ideal_height)
    
        # Height reward - higher reward for being closer to the ideal height
        height_reward = torch.exp(-height_deviation * 10)  # scaled to make it more sensitive
    
        # Orientation reward - ensure torso is perpendicular to gravity
        down_direction = torch.tensor([0., 0., -1.], device=env.device)
        orientation_error = env.projected_gravity - down_direction.unsqueeze(0).expand_as(env.projected_gravity)
        orientation_magnitude_error = torch.norm(orientation_error, dim=1)
    
        orientation_reward = torch.exp(-orientation_magnitude_error * 10)  # Make orientation highly rewarding
    
        # Smooth motion and avoiding DOF limits
        dof_pos_penalty = torch.zeros(env.num_envs, device=env.device)
        for i in range(env.num_dof):
            dof_min, dof_max = env.dof_pos_limits[i]
            # Squared penalty for DOF position near limits, allows for smoother motion within the middle of the range.
            limit_penalty = torch.where(env.dof_pos[:, i] < dof_min, (dof_min - env.dof_pos[:, i]) ** 2,
                                        torch.where(env.dof_pos[:, i] > dof_max, (env.dof_pos[:, i] - dof_max) ** 2,
                                                    torch.tensor(0., device=env.device)))
            dof_pos_penalty += limit_penalty
    
        dof_vel_penalty = torch.mean(torch.abs(env.dof_vel - env.last_dof_vel), dim=1)  # Reward for minimizing action (velocity) changes
    
        # Combine all rewards and penalties
        total_reward = velocity_reward + height_reward + orientation_reward - dof_pos_penalty - dof_vel_penalty
    
        env.extras['velocity_reward'] = velocity_reward
        env.extras['height_reward'] = height_reward
        env.extras['orientation_reward'] = orientation_reward
        env.extras['dof_pos_penalty'] = dof_pos_penalty
        env.extras['dof_vel_penalty'] = dof_vel_penalty
    
        return total_reward, env.extras
    
    # Success criteria as forward velocity
    def compute_success(self):
        target_velocity = 2.0
        lin_vel_error = torch.square(target_velocity - self.env.root_states[:, 7])
        return torch.exp(-lin_vel_error / 0.25)

