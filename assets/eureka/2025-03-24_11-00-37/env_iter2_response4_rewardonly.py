    def compute_reward(self):
        env = self.env
    
        # Adjusting weights for each component based on analysis
        velocity_weight = 1.5
        z_pos_weight = 0.2
        orientation_weight = 0.3
        dof_limit_penalty_weight = 0.02
        action_smoothness_weight = 0.01
    
        # Component calculations
        desired_velocity_x = 2.0
        desired_z_pos = 0.34
        desired_orientation = torch.tensor([0., 0., 0., 1.], device=env.device)
    
        # Error calculations
        velocity_error = torch.abs(env.root_states[:, 7] - desired_velocity_x) ** 2
        z_pos_error = torch.abs(env.root_states[:, 2] - desired_z_pos) ** 2
        orientation_error = 1 - torch.sum(env.base_quat * desired_orientation, dim=1) ** 2
    
        # Penalizations
        dof_limit_penalty = torch.sum(torch.relu(torch.abs(env.dof_pos - env.default_dof_pos) - (env.dof_pos_limits[:, 1] - env.dof_pos_limits[:, 0]).unsqueeze(0)/2), dim=1) ** 2
        action_smoothness = torch.mean(torch.abs(env.actions - env.last_actions), dim=1) ** 2
    
        # Reward combination with adjusted weights
        total_reward = (velocity_weight * (1 - velocity_error)) + \
                       (z_pos_weight * (1 - z_pos_error)) + \
                       (orientation_weight * (1 - orientation_error)) - \
                       (dof_limit_penalty_weight * dof_limit_penalty) - \
                       (action_smoothness_weight * action_smoothness)
    
        # Reward components dictionary for debugging and analysis
        reward_components = {
            'velocity': velocity_weight * (1 - velocity_error),
            'z_pos': z_pos_weight * (1 - z_pos_error),
            'orientation': orientation_weight * (1 - orientation_error),
            'dof_limit_penalty': -dof_limit_penalty_weight * dof_limit_penalty,
            'action_smoothness': -action_smoothness_weight * action_smoothness
        }
    
        return total_reward, reward_components
