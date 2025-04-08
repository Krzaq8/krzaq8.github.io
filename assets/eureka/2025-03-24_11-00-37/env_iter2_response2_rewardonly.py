    def compute_reward(self):
        env = self.env
        
        # Adjusting weights based on analysis
        velocity_weight = 1.5  # Important for the task, ensure it's emphasized but balanced.
        z_pos_weight = 0.1  # Minor adjustments for balance.
        orientation_weight = 0.2  # Ensure orientation is maintained but not overemphasized.
        dof_limit_penalty_weight = 0.01  # Reduce to lessen its impact and avoid discouraging movement.
        smoothness_weight = 0.01  # Reduce to encourage exploration of action space.
    
        desired_velocity_x = 2.0
        desired_z_pos = 0.34
        desired_orientation = torch.tensor([0., 0., 0., 1.], device=env.device)
    
        velocity_error = (env.root_states[:, 7] - desired_velocity_x) ** 2
        z_pos_error = (env.root_states[:, 2] - desired_z_pos) ** 2
        orientation_error = 1 - torch.abs(torch.sum(env.base_quat * desired_orientation, dim=1))
        
        dof_extension = torch.abs(env.dof_pos - env.default_dof_pos) / (env.dof_pos_limits[:, 1] - env.dof_pos_limits[:, 0]).unsqueeze(0)
        dof_limit_penalty = torch.sum(dof_extension, dim=1)
        
        action_smoothness = torch.mean((env.actions - env.last_actions) ** 2, dim=1)
    
        # Rebalance and normalize individual components
        velocity_reward = velocity_weight * (-velocity_error)
        z_pos_reward = z_pos_weight * (-z_pos_error)
        orientation_reward = orientation_weight * (-orientation_error)
        dof_limit_penalty_reward = -dof_limit_penalty_weight * dof_limit_penalty
        action_smoothness_reward = -smoothness_weight * action_smoothness
    
        total_reward = velocity_reward + z_pos_reward + orientation_reward + dof_limit_penalty_reward + action_smoothness_reward
    
        reward_components = {
            'velocity': velocity_reward,
            'z_pos': z_pos_reward,
            'orientation': orientation_reward,
            'dof_limit_penalty': dof_limit_penalty_reward,
            'action_smoothness': action_smoothness_reward
        }
    
        return total_reward, reward_components
