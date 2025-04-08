    def compute_reward(self):
        env = self.env  # Essential for accessing environment variables.
        
        # Adjusted weights for reward components
        velocity_weight = 0.5
        z_pos_weight = 0.3
        orientation_weight = 0.4
        dof_limit_penalty_weight = 0.3  # Increased to ensure DOF limits are respected
        smoothness_weight = 0.1  # Decreased to encourage exploration
    
        # Velocity component
        desired_velocity_x = 2.0
        velocity_error = torch.abs(env.root_states[:, 7] - desired_velocity_x)
        velocity_reward = torch.exp(-velocity_weight * velocity_error)
    
        # Z Position component
        desired_z_pos = 0.34
        z_pos_error = torch.abs(env.root_states[:, 2] - desired_z_pos)
        z_pos_reward = torch.exp(-z_pos_weight * z_pos_error)
    
        # Orientation component
        desired_orientation = torch.tensor([0., 0., 0., 1.], device=env.device)
        orientation_error = 1 - torch.abs(torch.sum(env.base_quat * desired_orientation, dim=1))
        orientation_reward = torch.exp(-orientation_weight * orientation_error)
    
        # DOF Limit Penalty component
        dof_limit_error = torch.max(torch.zeros_like(env.dof_pos), torch.abs(env.dof_pos - env.default_dof_pos) - (env.dof_pos_limits[:, 1] - env.dof_pos_limits[:, 0]).unsqueeze(0)/2).sum(axis=1)
        dof_limit_penalty = -dof_limit_penalty_weight * dof_limit_error
    
        # Smoothness component
        action_diff = torch.mean(torch.abs(env.actions - env.last_actions), dim=1)
        smoothness_penalty = -smoothness_weight * action_diff
    
        # Total reward
        total_reward = velocity_reward + z_pos_reward + orientation_reward + dof_limit_penalty + smoothness_penalty
    
        # Components for debugging
        reward_components = {
            'velocity': velocity_reward,
            'z_pos': z_pos_reward,
            'orientation': orientation_reward,
            'dof_limit_penalty': dof_limit_penalty,
            'action_smoothness': smoothness_penalty
        }
    
        return total_reward, reward_components
