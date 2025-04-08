    def compute_reward(self):
        env = self.env
        
        # Adjusted weights
        velocity_weight = 1.5
        z_pos_weight = 0.2
        orientation_weight = 0.15
        dof_limit_penalty_weight = 0.01
        smoothness_weight = 0.01
        
        # Objectives
        desired_velocity_x = 2.0
        desired_z_pos = 0.34
        desired_orientation = torch.tensor([0., 0., 0., 1.], device=env.device)
    
        # Calculations
        velocity_error = torch.abs(env.root_states[:, 7] - desired_velocity_x)
        z_pos_error = torch.abs(env.root_states[:, 2] - desired_z_pos)
        orientation_error = 1 - torch.abs(torch.sum(env.base_quat * desired_orientation, dim=1))
        
        dof_limit_penalty = torch.sum(torch.relu(torch.abs(env.dof_pos - env.default_dof_pos) - (env.dof_pos_limits[:, 1] - env.dof_pos_limits[:, 0]).unsqueeze(0)/2), axis=1)
        
        action_smoothness = torch.mean(torch.abs(env.actions - env.last_actions), dim=1)
    
        # Scoring
        velocity_score = velocity_weight * (1 - torch.tanh(velocity_error))
        z_pos_score = z_pos_weight * (1 - torch.tanh(z_pos_error))
        orientation_score = orientation_weight * (1 - torch.tanh(orientation_error))
        dof_limit_penalty_score = -dof_limit_penalty_weight * dof_limit_penalty
        smoothness_score = -smoothness_weight * action_smoothness
        
        total_reward = velocity_score + z_pos_score + orientation_score + dof_limit_penalty_score + smoothness_score
    
        reward_components = {
            'velocity': velocity_score,
            'z_pos': z_pos_score,
            'orientation': orientation_score,
            'dof_limit_penalty': dof_limit_penalty_score,
            'action_smoothness': smoothness_score
        }
    
        return total_reward, reward_components
