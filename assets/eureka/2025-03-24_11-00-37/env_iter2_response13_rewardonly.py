    def compute_reward(self):
        env = self.env
        
        # Adjusted weights for each reward component
        velocity_weight = 1.5  # Adjusted to balance with other components
        z_pos_weight = 0.2  # Reducing influence to balance total reward
        orientation_weight = 0.2  # Slight decrease to encourage balance
        dof_limit_penalty_weight = 0.1  # Increased to enhance its effect
        smoothness_weight = 0.02  # Significantly reduced weight
        
        desired_velocity_x = 2.0
        desired_z_pos = 0.34
        desired_orientation_quat = torch.tensor([0., 0., 0., 1.], device=env.device)
        
        # Compute component errors
        velocity_error = torch.abs(env.root_states[:, 7] - desired_velocity_x)
        z_pos_error = torch.abs(env.root_states[:, 2] - desired_z_pos)
        orientation_error = 1 - torch.abs(torch.sum(env.base_quat * desired_orientation_quat, dim=1))
        
        # Calculate penalties and rewards
        dof_limit_penalty = torch.sum(torch.relu(env.dof_pos - env.dof_pos_limits[:, 1]) + torch.relu(env.dof_pos_limits[:, 0] - env.dof_pos), dim=1)
        
        # Smoother transitions
        action_smoothness = torch.mean(torch.abs(env.actions - env.last_actions), dim=1)
        
        velocity_reward = 1.0 - torch.tanh(velocity_error)
        z_pos_reward = 1.0 - torch.tanh(z_pos_error)
        orientation_reward = 1.0 - torch.tanh(orientation_error)
    
        # Aggregate total reward
        total_reward = (velocity_weight * velocity_reward) + \
                       (z_pos_weight * z_pos_reward) + \
                       (orientation_weight * orientation_reward) - \
                       (dof_limit_penalty_weight * dof_limit_penalty) - \
                       (smoothness_weight * action_smoothness)
        
        reward_components = {
            'velocity': velocity_reward,
            'z_pos': z_pos_reward,
            'orientation': orientation_reward,
            'dof_limit_penalty': -dof_limit_penalty,
            'action_smoothness': -action_smoothness
        }
        
        return total_reward, reward_components
