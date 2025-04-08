    def compute_reward(self):
        env = self.env
        
        # Component weights
        velocity_weight = 1.5  # Increased emphasis on reaching and maintaining desired velocity
        z_pos_weight = 1.0  # Ensure maintaining altitude is prioritized
        orientation_weight = 0.5  # Reduce to balance with velocity and z position components
        dof_limit_penalty_weight = 2.0  # Increase to ensure DOF limits are respected
        smoothness_weight = 0.1  # Significantly reduce to encourage exploration
        
        desired_velocity_x = 2.0
        desired_z_pos = 0.34
        orientation_target = torch.tensor([0., 0., 0., 1.], device=env.device)
        
        # Calculate errors
        velocity_error = (env.root_states[:, 7] - desired_velocity_x).abs()
        z_pos_error = (env.root_states[:, 2] - desired_z_pos).abs()
        orientation_error = 1 - torch.abs(torch.sum(env.base_quat * orientation_target, dim=1))
        dof_limit_penalty = torch.sum(torch.relu(env.dof_pos - env.dof_pos_limits[:, 1])**2 + torch.relu(env.dof_pos_limits[:, 0] - env.dof_pos)**2, axis=1)
        action_smoothness = torch.mean((env.actions - env.last_actions).abs(), dim=1)
        
        # Compute the rewards and penalties
        velocity_reward = torch.exp(-velocity_error * velocity_weight)
        z_pos_reward = torch.exp(-z_pos_error * z_pos_weight)
        orientation_reward = torch.exp(-orientation_error * orientation_weight)
        action_smoothness_penalty = -smoothness_weight * action_smoothness
        dof_limit_penalty = -dof_limit_penalty_weight * dof_limit_penalty
    
        # Total reward
        total_reward = velocity_reward + z_pos_reward + orientation_reward + action_smoothness_penalty + dof_limit_penalty
        
        reward_components = {
            'velocity_reward': velocity_reward,
            'z_pos_reward': z_pos_reward,
            'orientation_reward': orientation_reward,
            'action_smoothness_penalty': action_smoothness_penalty,
            'dof_limit_penalty': dof_limit_penalty,
        }
        
        return total_reward, reward_components
