    def compute_reward(self):
        env = self.env
        
        # Configured weights after analysis
        velocity_weight = 0.5  # Slightly decrease to balance with other components
        z_pos_weight = 0.15  # Adjust based on importance in task
        orientation_weight = 0.1  # Decrease to ensure balance
        dof_limit_penalty_weight = 0.05  # Increase slightly to ensure adherence to joint limits
        smoothness_weight = 0.05  # Increase to encourage smoother actions without heavy penalties
        
        # Objectives constants
        desired_velocity_x = 2.0
        desired_z_pos = 0.34
        up_quat = torch.tensor([1., 0., 0., 0.], device=env.device)  # Quaternion pointing up for comparison
    
        # Error Computations
        velocity_error = torch.abs(env.root_states[:, 7] - desired_velocity_x)
        z_pos_error = torch.abs(env.root_states[:, 2] - desired_z_pos)
        orientation_error = torch.abs(torch.sum(gymtorch.quat_mul(env.base_quat, gymtorch.quat_conjugate(up_quat)), dim=1) - 1)
        dof_limit_penalty = torch.sum(torch.relu(torch.abs(env.dof_pos - env.default_dof_pos) - (0.5 * (env.dof_pos_limits[:, 1] - env.dof_pos_limits[:, 0])).unsqueeze(0)), dim=1)
        action_smoothness = torch.mean(torch.abs(env.actions - env.last_actions), dim=1)
    
        # Reward Components
        velocity_reward = (1.0 - torch.exp(-velocity_weight * velocity_error))
        z_pos_reward = (1.0 - torch.exp(-z_pos_weight * z_pos_error))
        orientation_reward = (1.0 - torch.exp(-orientation_weight * orientation_error))
        dof_limit_penalty = -dof_limit_penalty_weight * dof_limit_penalty
        action_smoothness_reward = -smoothness_weight * action_smoothness
    
        # Total Reward
        total_reward = velocity_reward + z_pos_reward + orientation_reward + dof_limit_penalty + action_smoothness_reward
        
        # Returning the reward and its components
        reward_components = {
            'velocity': velocity_reward,
            'z_pos': z_pos_reward,
            'orientation': orientation_reward,
            'dof_limit_penalty': dof_limit_penalty,
            'action_smoothness': action_smoothness_reward
        }
    
        return total_reward, reward_components
