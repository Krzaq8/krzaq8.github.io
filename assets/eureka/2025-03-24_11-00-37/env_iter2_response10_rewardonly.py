    def compute_reward(self):
        env = self.env
    
        # Adjusted weights for different parts of the reward to achieve balance
        velocity_weight = 1.0
        z_pos_weight = 0.5
        orientation_weight = 0.3
        dof_limit_penalty_weight = 0.1
        smoothness_weight = 0.05  # Reduced to make action exploration less penalized
    
        # Velocity
        desired_velocity_x = 2.0
        velocity_error = torch.abs(env.root_states[:, 7] - desired_velocity_x)
        velocity_reward = (-velocity_error).exp()
    
        # Z Position Stability
        desired_z_pos = 0.34
        z_pos_error = torch.abs(env.root_states[:, 2] - desired_z_pos)
        z_pos_reward = (-z_pos_error).exp()
    
        # Orientation towards gravity
        desired_orientation_cos = torch.tensor([1.0], device=env.device)  # Assuming perfect alignment gives cos(angle)=1
        current_orientation_cos = torch.abs(torch.sum(env.base_quat * torch.tensor([1., 0., 0., 0.], device=env.device), dim=1))
        orientation_reward = (desired_orientation_cos - current_orientation_cos).exp()
    
        # DOF Limit Penalty
        dof_limit_penalty = torch.sum(torch.relu(torch.abs(env.dof_pos - env.default_dof_pos) - (env.dof_pos_limits[:, 1] - env.dof_pos_limits[:, 0]).unsqueeze(0) / 2), axis=1)
        dof_limit_penalty_reward = -dof_limit_penalty_weight * dof_limit_penalty
    
        # Action Smoothness
        action_diff = torch.mean(torch.abs(env.actions - env.last_actions), dim=1)
        action_smoothness_reward = -smoothness_weight * action_diff
    
        # Total Reward Composition
        total_reward = velocity_weight * velocity_reward + \
                       z_pos_weight * z_pos_reward + \
                       orientation_weight * orientation_reward + \
                       dof_limit_penalty_reward + \
                       action_smoothness_reward
    
        # Individual Reward Components for Logging
        reward_components = {
            'velocity': velocity_reward,
            'z_pos': z_pos_reward,
            'orientation': orientation_reward,
            'dof_limit_penalty': dof_limit_penalty_reward,
            'action_smoothness': action_smoothness_reward
        }
    
        return total_reward, reward_components
