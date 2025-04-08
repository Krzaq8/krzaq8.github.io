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
        env = self.env  # Do not skip this line
    
        # Constants
        target_vel_x = 2.0
        target_base_height = 0.34
        orientation_reward_weight = 0.1
        velocity_reward_weight = 1.0
        height_reward_weight = 0.1
        dof_limit_penalty_weight = 0.05
        action_diff_penalty_weight = 0.02
    
        # Compute velocity reward
        vel_x = env.root_states[:, 7]
        velocity_reward = -torch.abs(vel_x - target_vel_x)
        
        # Compute height reward
        base_height = env.root_states[:, 2]  # Z position is the third element
        height_reward = -torch.abs(base_height - target_base_height)
        
        # Orientation reward
        down_vector = torch.tensor([0., 0., -1.], device=env.device).repeat(env.num_envs, 1)
        current_down = env.projected_gravity / torch.norm(env.projected_gravity, dim=1, keepdim=True)
        orientation_reward = -torch.sum(down_vector * current_down, dim=1)
    
        # Compute DOF limit penalty
        dof_limit_penalty = torch.zeros(env.num_envs, device=env.device)
        for i in range(env.num_dof):
            dof_pos = env.dof_pos[:, i]
            lower_limits, upper_limits = env.dof_pos_limits[i, 0], env.dof_pos_limits[i, 1]
            over_lower_limit = torch.relu(lower_limits - dof_pos)
            over_upper_limit = torch.relu(dof_pos - upper_limits)
            dof_limit_penalty += over_lower_limit + over_upper_limit
    
        # Action rate penalty to encourage minimal actions change
        action_diff = torch.norm(env.actions - env.last_actions, dim=1)
        action_diff_penalty = action_diff
    
        # Combined reward with weights
        total_reward = velocity_reward_weight * velocity_reward + \
                       orientation_reward_weight * orientation_reward + \
                       height_reward_weight * height_reward + \
                       -dof_limit_penalty_weight * dof_limit_penalty + \
                       -action_diff_penalty_weight * action_diff_penalty
    
        return total_reward, {"velocity_reward": velocity_reward,
                              "height_reward": height_reward,
                              "orientation_reward": orientation_reward,
                              "dof_limit_penalty": dof_limit_penalty,
                              "action_diff_penalty": action_diff_penalty}
    
    # Success criteria as forward velocity
    def compute_success(self):
        target_velocity = 2.0
        lin_vel_error = torch.square(target_velocity - self.env.root_states[:, 7])
        return torch.exp(-lin_vel_error / 0.25)

