    def compute_reward(self):
        env = self.env
    
        # Adjusted weightings to balance the components more effectively
        velocity_weight = 1.0  # Scaled down to not overpower other components
        z_pos_weight = 0.5     # Increased slightly for more emphasis
        orientation_weight = 0.1   # Scaled down to balance its impact
        smoothness_weight = 0.02   # Scaled down for softer impact
        dof_limit_penalty_weight = 0.05  # Increased slightly to ensure DOF limits are respected
        
        # Objectives
        desired_velocity_x = 2.0
        desired_z_pos = 0.34
        
        velocity_x = env.root_states[:, 7]
        z_pos = env.root_states[:, 2]
        
        # Computing errors
        velocity_error = torch.abs(velocity_x - desired_velocity_x)
        z_pos_error = torch.abs(z_pos - desired_z_pos)
        orientation_error = 1 - torch.abs(torch.sum(env.base_quat * torch.tensor([1., 0., 0., 0.], device=env.device), dim=1))
        
        dof_limit_penalty = torch.sum(torch.relu(torch.abs(env.dof_pos - env.default_dof_pos) - (env.dof_pos_limits[:, 1] - env.dof_pos_limits[:, 0]).unsqueeze(0)/2), dim=1)
        action_smoothness = torch.mean(torch.abs(env.actions - env.last_actions), dim=1)
        
        # Reward calculations
        velocity_score = torch.exp(-velocity_weight * velocity_error)
        z_pos_score = torch.exp(-z_pos_weight * z_pos_error)
        orientation_score = torch.exp(-orientation_weight * orientation_error)
        smoothness_score = -smoothness_weight * action_smoothness
        dof_penalty_score = -dof_limit_penalty_weight * dof_limit_penalty
        
        # Total Reward
        total_reward = velocity_score + z_pos_score + orientation_score + smoothness_score + dof_penalty_score
    
        # Reward components for debugging and tuning purposes
        reward_components = {
            'velocity': velocity_score,
            'z_pos': z_pos_score,
            'orientation': orientation_score,
            'smoothness': smoothness_score,
            'dof_limit_penalty': dof_penalty_score,
        }
    
        return total_reward, reward_components
