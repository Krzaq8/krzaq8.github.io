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
        
        # Velocity reward for moving forward at 2.0 m/s
        target_velocity = 2.0  # Target forward velocity in m/s
        current_velocity_x = env.root_states[:, 7]  # Linear velocity in the x direction
        velocity_reward = -torch.abs(current_velocity_x - target_velocity)
    
        # Position reward for keeping the torso near z = 0.34
        target_z_position = 0.34  # Target z position for the torso
        current_z_position = env.root_states[:, 2]  # Z position of the robot's base
        position_z_reward = -torch.abs(current_z_position - target_z_position)
    
        # Orientation reward to keep the torso perpendicular to gravity
        down_vector = torch.tensor([0.0, 0.0, 1.0], device=env.device).repeat((env.num_envs, 1))
        current_down = env.projected_gravity
        orientation_reward = -torch.norm(down_vector - current_down, dim=1)
    
        # DOF position rewards to encourage smooth leg movements and avoid reaching limits
        dof_pos_limits = env.dof_pos_limits
        dof_position_reward = -torch.sum(torch.max(torch.zeros_like(env.dof_pos), torch.abs(env.dof_pos - dof_pos_limits.mean(dim=1, keepdim=True)) - (dof_pos_limits[:, 1] - dof_pos_limits[:, 0]) / 4), dim=1)
        
        # Aggregate the individual rewards into a total reward
        total_reward = velocity_reward + position_z_reward + orientation_reward + dof_position_reward
    
        # Normalize rewards to have a similar scale
        total_reward /= 4.0
    
        # Create a dictionary for individual reward components
        reward_components = {
            "velocity_reward": velocity_reward.mean(),
            "position_z_reward": position_z_reward.mean(),
            "orientation_reward": orientation_reward.mean(),
            "dof_position_reward": dof_position_reward.mean(),
        }
    
        return total_reward, reward_components
    
    # Success criteria as forward velocity
    def compute_success(self):
        target_velocity = 2.0
        lin_vel_error = torch.square(target_velocity - self.env.root_states[:, 7])
        return torch.exp(-lin_vel_error / 0.25)

