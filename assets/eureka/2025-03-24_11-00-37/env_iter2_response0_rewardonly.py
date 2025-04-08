    def compute_reward(self):
        env = self.env
        
        # Adjusted weights for balancing
        velocity_weight = 1.0  # Emphasize reaching the desired velocity, adjusted for balance.
        z_pos_weight = 0.2  # Decreased to ensure balance with other components.
        orientation_weight = 0.1  # Slightly decreased; orientation is being learned well.
        dof_limit_penalty_weight = 0.2  # Increased emphasis to better manage joint movements.
        smoothness_weight = -0.1  # Softened impact to encourage action variability but still penalize erratic behavior.
        
        # Deriving reward components
        desired_velocity_x = 2.0
        velocity_error = torch.abs(env.root_states[:, 7] - desired_velocity_x)  # Velocity in the x-direction
        
        z_pos_error = torch.abs(env.root_states[:, 2] - 0.34)  # Deviation from desired z position
        
        # Error in orientation relative to being upright (aligned with gravity)
        desired_orientation = torch.tensor([0., 0., 0., 1.], device=env.device)
        orientation_error = 1 - torch.abs(torch.sum(env.base_quat * desired_orientation, dim=1))
        
        # Penalizing deviation from initial DOF position, mitigated by softened limits
        dof_limit_penalty = torch.sum(torch.relu(torch.abs(env.dof_pos - env.default_dof_pos) - (env.dof_pos_limits[:, 1] - env.dof_pos_limits[:, 0]).unsqueeze(0) / 2), axis=1)
        
        action_smoothness = torch.mean(torch.abs(env.actions - env.last_actions), dim=1)
        
        # Calculating total reward with adjusted component weights
        total_reward = (velocity_weight * (-velocity_error)) + \
                       (z_pos_weight * (-z_pos_error)) + \
                       (orientation_weight * (-orientation_error)) - \
                       (dof_limit_penalty_weight * dof_limit_penalty) - \
                       (smoothness_weight * action_smoothness)
        
        # Reward components dictionary
        reward_components = {
            'velocity': velocity_weight * (-velocity_error),
            'z_pos': z_pos_weight * (-z_pos_error),
            'orientation': orientation_weight * (-orientation_error),
            'dof_limit_penalty': -dof_limit_penalty_weight * dof_limit_penalty,
            'action_smoothness': smoothness_weight * (-action_smoothness)  # Negated for clarification; smoothness penalty is inverted here
        }
    
        return total_reward, reward_components
