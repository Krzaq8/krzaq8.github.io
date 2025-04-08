    def compute_reward(self):
        env = self.env
      
        # Scaling factors adjusted based on analysis
        velocity_weight = 0.8  # Adjusted for balance
        z_pos_weight = 0.5  # Increased to emphasize maintaining a stable height
        orientation_weight = 0.1  # Reduced to balance with other rewards
        dof_limit_penalty_weight = 0.05  # Slightly increased impact
        smoothness_weight = 0.025  # Reduced to lessen penalty impact
    
        # Calculating errors
        desired_velocity_x = 2.0  # Target velocity
        velocity_error = torch.abs(env.root_states[:, 7] - desired_velocity_x)
    
        desired_z_pos = 0.34  # Target height
        z_pos_error = torch.abs(env.root_states[:, 2] - desired_z_pos)
    
        # Orientation towards gravity should be as aligned as possible (quaternion dot product)
        gravity_alignment = torch.abs(torch.sum(env.base_quat * torch.tensor([0., 0., 0., 1.], device=env.device), dim=1))
    
        # Ensure DOF positions stay within soft limits
        dof_violations = torch.relu(torch.abs(env.dof_pos - env.default_dof_pos) - (env.dof_pos_limits[:, 1] - env.dof_pos_limits[:, 0]).unsqueeze(0)/2).sum(dim=1)
    
        # Actions should change smoothly from one timestep to the next
        action_delta = torch.mean(torch.abs(env.actions - env.last_actions), dim=1)
    
        # Reward calculations
        velocity_reward = velocity_weight * (1 - torch.tanh(velocity_error))
        z_pos_reward = z_pos_weight * (1 - torch.tanh(z_pos_error))
        orientation_reward = orientation_weight * (gravity_alignment)
        dof_limit_penalty = -dof_limit_penalty_weight * dof_violations
        smoothness_penalty = -smoothness_weight * action_delta
    
        total_reward = velocity_reward + z_pos_reward + orientation_reward + dof_limit_penalty + smoothness_penalty
    
        reward_components = {
            'velocity': velocity_reward,
            'z_pos': z_pos_reward,
            'orientation': orientation_reward,
            'dof_limit_penalty': dof_limit_penalty,
            'action_smoothness': smoothness_penalty
        }
    
        return total_reward, reward_components
