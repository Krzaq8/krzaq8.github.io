    def compute_reward(self):
        env = self.env  # Required environment access line
        
        # Calculate components of the reward function
        velocity_error = torch.square(env.forward_vec[:, 0] * 2.0 - env.root_states[:, 7])
        orientation_error = 1 - torch.abs(env.projected_gravity[:, 2])  # Closer to 1 when upright
        z_pos_error = torch.square(env.root_states[:, 2] - 0.34)
        action_diff = torch.mean(torch.abs(env.actions - env.last_actions), dim=1)
        dof_limit_penalties = torch.sum(torch.clamp(torch.abs(env.dof_pos - env.default_dof_pos) - (env.dof_pos_limits[:, 1] - env.dof_pos_limits[:, 0])/2, min=0), dim=1)
    
        # Reward weights, adjusted based on analysis
        weights = {
            'velocity': 1.0,
            'orientation': 0.5,
            'z_pos': 0.2,
            'smoothness': -0.05,  # Lessened impact to encourage exploration
            'dof_limit': -0.05  # Smoothed penalty for less harshness
        }
    
        # Compute individual component contributions and total reward
        rew_velocity = -weights['velocity'] * velocity_error
        rew_orientation = weights['orientation'] * (1 - orientation_error)
        rew_z_pos = -weights['z_pos'] * z_pos_error
        rew_smoothness = weights['smoothness'] * action_diff
        rew_dof_limit = weights['dof_limit'] * dof_limit_penalties
    
        total_reward = rew_velocity + rew_orientation + rew_z_pos + rew_smoothness + rew_dof_limit
    
        # Components for debugging/analysis
        reward_components = {
            'velocity': rew_velocity,
            'orientation': rew_orientation,
            'z_pos': rew_z_pos,
            'smoothness': rew_smoothness,
            'dof_limit': rew_dof_limit
        }
    
        return total_reward, reward_components
