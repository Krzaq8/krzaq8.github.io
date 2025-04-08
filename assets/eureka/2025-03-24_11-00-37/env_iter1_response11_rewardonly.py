    def compute_reward(self):
        env = self.env
    
        # Adjusted weights
        velocity_weight = 2.0
        z_pos_weight = 1.0
        orientation_weight = 0.5
        dof_limit_penalty_weight = 0.05
        smoothness_weight = 0.1
    
        # Desired velocity and positional values
        desired_velocity_x = 2.0
        desired_z_pos = 0.34
        desired_orientation = torch.tensor([0., 0., 0., 1.], device=env.device)
    
        # Velocity reward
        actual_velocity_x = env.root_states[:, 7]
        velocity_error = torch.abs(actual_velocity_x - desired_velocity_x)
        velocity_reward = velocity_weight * torch.exp(-velocity_error)
    
        # Z position reward
        actual_z_pos = env.root_states[:, 2]
        z_pos_error = torch.abs(actual_z_pos - desired_z_pos)
        z_pos_reward = z_pos_weight * torch.exp(-z_pos_error)
    
        # Orientation reward
        orientation_error = 1 - torch.abs(torch.sum(env.base_quat * desired_orientation, dim=1))
        orientation_reward = orientation_weight * orientation_error
    
        # Soft DOF limit penalty
        dof_limit_violation = torch.relu(torch.abs(env.dof_pos - env.default_dof_pos) - (env.dof_pos_limits[:, 1] - env.dof_pos_limits[:, 0])/2).sum(axis=1)
        dof_limit_penalty = -dof_limit_penalty_weight * dof_limit_violation
    
        # Action smoothness reward
        action_diff = torch.mean(torch.abs(env.actions - env.last_actions), dim=1)
        action_smoothness_reward = -smoothness_weight * action_diff
    
        # Total reward
        total_reward = velocity_reward + z_pos_reward + orientation_reward + dof_limit_penalty + action_smoothness_reward
    
        # Reward components for analysis
        reward_components = {
            'velocity': velocity_reward,
            'z_pos': z_pos_reward,
            'orientation': orientation_reward,
            'dof_limit_penalty': dof_limit_penalty,
            'action_smoothness': action_smoothness_reward
        }
        
        return total_reward, reward_components
