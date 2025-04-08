    def compute_reward(self):
        env = self.env
        
        # Updated weights for balanced contributions from each part
        velocity_weight = 1.0
        z_pos_weight = 1.0
        orientation_weight = 0.5
        dof_limit_penalty_weight = 0.1
        smoothness_weight = 0.05  # Reduced weight for action smoothness
        
        # Desired values remain the same
        desired_velocity_x = 2.0
        desired_z_pos = 0.34
        desired_orientation = torch.tensor([0., 0., 0., 1.], device=env.device)
        
        # Calculations for reward components with slight adjustments
        
        velocity_error = torch.abs(env.root_states[:, 7] - desired_velocity_x)
        z_pos_error = torch.abs(env.root_states[:, 2] - desired_z_pos)
        
        # Using quaternion dot product for orientation alignment
        orientation_error = 1 - torch.abs(torch.sum(env.base_quat * desired_orientation, dim=1))
        
        # Adjustment for dof limit check, focusing on normalization and scaling
        dof_limit_violation = torch.clamp((env.dof_pos - env.default_dof_pos)**2 - (env.dof_pos_limits[:, 1] - env.dof_pos_limits[:, 0])**2 / 4, min=0).sum(axis=1)
        
        # Smoother action changes
        action_smoothness = torch.mean(torch.square(env.actions - env.last_actions), dim=1)
        
        # Recalculating total reward with adjusted components
        reward = (velocity_weight * (1 - velocity_error / desired_velocity_x)) + \
                 (z_pos_weight * torch.exp(-z_pos_error)) + \
                 (orientation_weight * (1 - orientation_error)) - \
                 (dof_limit_penalty_weight * dof_limit_violation) - \
                 (smoothness_weight * action_smoothness)
        
        # Detailed components for analysis
        reward_components = {
            'velocity': velocity_weight * (1 - velocity_error / desired_velocity_x),
            'z_pos': z_pos_weight * torch.exp(-z_pos_error),
            'orientation': orientation_weight * (1 - orientation_error),
            'dof_limit_penalty': -dof_limit_penalty_weight * dof_limit_violation,
            'action_smoothness': -smoothness_weight * action_smoothness
        }
        
        return reward, reward_components
