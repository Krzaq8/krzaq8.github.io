    def compute_reward(self):
        env = self.env
        
        # Adjusted weights based on analysis for better balance
        velocity_weight = 2.0
        z_pos_weight = 0.5
        orientation_weight = 0.5
        dof_limit_penalty_weight = 0.02
        smoothness_weight = 0.01  # Significantly reduced to mitigate heavy penalties
        
        desired_velocity_x = 2.0
        desired_z_pos = 0.34
        desired_orientation = torch.tensor([0., 0., 0., 1.], device=env.device)
    
        velocity_error = torch.abs(env.root_states[:, 7] - desired_velocity_x)
        z_pos_error = torch.abs(env.root_states[:, 2] - desired_z_pos)
        orientation_error = 1 - torch.abs(torch.sum(env.base_quat * desired_orientation, dim=1))
        
        dof_limit_penalty = torch.sum(torch.relu(torch.abs(env.dof_pos - env.default_dof_pos) - (env.dof_pos_limits[:, 1] - env.dof_pos_limits[:, 0]).unsqueeze(0)/2), axis=1)
        
        action_smoothness = torch.mean(torch.abs(env.actions - env.last_actions), dim=1)
    
        # Re-scaled to ensure a balanced influence among components
        reward = (velocity_weight * (1 - velocity_error / desired_velocity_x)) + \
                 (z_pos_weight * torch.exp(-z_pos_error)) + \
                 (orientation_weight * torch.exp(-orientation_error)) - \
                 (dof_limit_penalty_weight * dof_limit_penalty) - \
                 (smoothness_weight * action_smoothness)
    
        reward_components = {
            'velocity': velocity_weight * (1 - velocity_error / desired_velocity_x),
            'z_pos': z_pos_weight * torch.exp(-z_pos_error),
            'orientation': orientation_weight * torch.exp(-orientation_error),
            'dof_limit_penalty': -dof_limit_penalty_weight * dof_limit_penalty,
            'action_smoothness': -smoothness_weight * action_smoothness
        }
    
        return reward, reward_components
