    def compute_reward(self):
        env = self.env
        
        # Adjusted weights
        velocity_weight = 1.0  # Adjusted for a more balanced contribution
        z_pos_weight = 0.5  # Slightly increased to emphasize maintaining a stable height
        orientation_weight = 0.1  # Reduced to prevent overwhelming contribution
        dof_limit_penalty_weight = 0.05  # Increased to further discourage reaching DOF limits
        smoothness_weight = 0.005  # Significantly reduced to allow for action variability
        
        # Objectives
        desired_velocity_x = 2.0  # m/s
        desired_z_pos = 0.34  # meters
        desired_orientation = env.projected_gravity  # Assuming this aligns with the gravity direction
        
        # Calculations
        velocity_error = torch.abs(env.root_states[:, 7] - desired_velocity_x)
        z_pos_error = torch.abs(env.root_states[:, 2] - desired_z_pos)
        orientation_error = 1 - torch.sum(env.projected_gravity * desired_orientation, dim=1)
        
        dof_limit_penalty = torch.sum(torch.relu(torch.abs(env.dof_pos - env.default_dof_pos) - (env.dof_pos_limits[:, 1] - env.dof_pos_limits[:, 0]).unsqueeze(0) / 2), dim=1)
        
        action_smoothness = torch.mean(torch.abs(env.actions - env.last_actions), dim=1)
        
        # Scoring
        velocity_score = velocity_weight * (1 - torch.sigmoid(10 * velocity_error))  # Using sigmoid for a gentler slope
        z_pos_score = z_pos_weight * (1 - torch.sigmoid(10 * z_pos_error))  # Sigmoid for controlled error sensitivity
        orientation_score = orientation_weight * (1 - torch.sigmoid(10 * orientation_error))  # Encouraging upright posture
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
