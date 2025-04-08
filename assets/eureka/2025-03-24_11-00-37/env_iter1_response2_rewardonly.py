    def compute_reward(self):
        env = self.env
    
        # Adjusting weights for a balanced influence
        velocity_weight = 2.0
        z_pos_weight = 1.0
        orientation_weight = 1.0
        dof_limit_penalty_weight = 0.05  # Softening to avoid overly harsh penalties
        smoothness_weight = 0.1  # Reduced to diminish its dominant effect
    
        # Desired values
        desired_velocity_x = 2.0  # m/s
        desired_z_pos = 0.34  # meters
    
        # Calculate velocity error and reward
        actual_velocity_x = env.root_states[:, 7]  # Global linear velocity x component
        velocity_error = (actual_velocity_x - desired_velocity_x).abs()
        velocity_reward = torch.exp(-velocity_error)  # Exponential decay to encourage precise matching
    
        # Calculate Z position error and reward
        z_pos_error = (env.root_states[:, 2] - desired_z_pos).abs()  # Global Z position error
        z_pos_reward = torch.exp(-z_pos_error)  # Exponential to focus on staying close
    
        # Calculate orientation error and reward
        orientation_error = 1 - torch.abs(torch.sum(env.base_quat * torch.tensor([0., 0., 0., 1.], device=env.device), dim=1))
        orientation_reward = torch.exp(-2 * orientation_error)  # Exponential decay to penalize deviations more sharply
    
        # DOF limit penalty softly encourages not hitting the limits without being overtly punitive
        dof_limit_penalty = torch.clamp(torch.max(torch.zeros_like(env.dof_pos), torch.abs(env.dof_pos - env.default_dof_pos) - (env.dof_pos_limits[:, 1] - env.dof_pos_limits[:, 0]).unsqueeze(0)/2), max=0).sum(axis=1)
    
        # Action smoothness to encourage subtler changes without being too dominant
        action_smoothness_penalty = torch.mean((env.actions - env.last_actions).abs(), dim=1)
    
        # Calculate total reward with adjusted components
        reward = velocity_weight * velocity_reward + \
                 z_pos_weight * z_pos_reward + \
                 orientation_weight * orientation_reward - \
                 dof_limit_penalty_weight * dof_limit_penalty - \
                 smoothness_weight * action_smoothness_penalty
    
        # Detailed reward components for debugging/analysis
        reward_components = {
            'velocity_reward': velocity_reward,
            'z_pos_reward': z_pos_reward,
            'orientation_reward': orientation_reward,
            'dof_limit_penalty': -dof_limit_penalty,
            'action_smoothness_penalty': -action_smoothness_penalty
        }
    
        return reward, reward_components
