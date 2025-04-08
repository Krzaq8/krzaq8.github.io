    def compute_reward(self):
        env = self.env  
    
        # Adjust weights for different parts of the reward based on policy feedback
        velocity_weight = 1.0
        z_pos_weight = 0.5
        orientation_weight = 0.3  # Slightly increased based on analysis
        dof_limit_penalty_weight = 0.05  # Decreased based on analysis
        smoothness_weight = 0.05  # Significantly decreased to be less punitive
    
        # Calculate velocity reward closer to desired with softened penalty/proximity approach
        velocity_error = torch.abs(env.root_states[:, 7] - 2.0)
        velocity_reward = torch.exp(-velocity_weight * velocity_error)
    
        # Z position reward similar approach, focusing on precision
        z_pos_reward = torch.exp(-z_pos_weight * torch.abs(env.root_states[:, 2] - 0.34))
    
        # Orientation reward using quaternion dot to expected orientation (1 indicates perfect alignment)
        desired_orientation = torch.tensor([1., 0., 0., 0.], device=env.device)
        orientation_reward = torch.exp(-orientation_weight * (1 - torch.abs(torch.sum(env.base_quat * desired_orientation, dim=1))))
    
        # DOF limit penalty using softer approach
        dof_limit_violation = torch.relu(torch.abs(env.dof_pos - env.default_dof_pos) - (env.dof_pos_limits[:, 1] - env.dof_pos_limits[:, 0]).unsqueeze(0)/2).sum(axis=1)
        dof_limit_penalty = torch.exp(-dof_limit_penalty_weight * dof_limit_violation)
    
        # Action smoothness adjusted for less punitive measures
        action_smoothness_penalty = torch.mean(torch.abs(env.actions - env.last_actions), dim=1)
        action_smoothness_reward = torch.exp(-smoothness_weight * action_smoothness_penalty)
    
        # Calculate total reward with adjusted rewards and penalties
        total_reward = velocity_reward + z_pos_reward + orientation_reward + dof_limit_penalty + action_smoothness_reward
    
        # Construct detailed reward components for debugging/analysis
        reward_components = {
            'velocity_reward': velocity_reward,
            'z_pos_reward': z_pos_reward,
            'orientation_reward': orientation_reward,
            'dof_limit_penalty': dof_limit_penalty,
            'action_smoothness_reward': action_smoothness_reward
        }
    
        return total_reward, reward_components
