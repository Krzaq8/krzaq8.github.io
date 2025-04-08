    def compute_reward(self):
        env = self.env
        
        # Adjusted weights for reward components based on analysis
        velocity_weight = 1.0  # Balanced emphasis
        z_pos_weight = 0.5  # Slightly reduce
        orientation_weight = 0.1  # Reduce weight, as it is learned well
        dof_limit_penalty_weight = 0.05  # Keep low to avoid harsh penalties
        smoothness_weight = 0.01  # Significantly reduce to encourage action variability
        
        # Velocity reward focused on reaching and maintaining the desired velocity
        desired_velocity_x = 2.0
        velocity_reward = 1.0 - torch.abs(env.root_states[:, 7] - desired_velocity_x) / desired_velocity_x
        
        # Z position reward for maintaining the desired height
        desired_z_pos = 0.34
        z_pos_reward = -(env.root_states[:, 2] - desired_z_pos).pow(2)
        
        # Orientation reward for aligning with the upward z-axis
        desired_orientation = torch.tensor([0., 0., 0., 1.], device=env.device)
        orientation_reward = 1.0 - torch.abs(torch.sum(env.base_quat * desired_orientation, dim=1))
        
        # DOF limit penalty to discourage reaching the limits of joint movements
        dof_limit_penalty = -torch.sum(torch.relu(torch.abs(env.dof_pos - env.default_dof_pos) - 0.5 * (env.dof_pos_limits[:, 1] - env.dof_pos_limits[:, 0]).unsqueeze(0) ), axis=1)
        
        # Action smoothness to encourage smooth transitions between actions
        action_smoothness = -torch.mean(torch.abs(env.actions - env.last_actions), dim=1)
        
        # Combine rewards with the adjusted weights
        total_reward = (velocity_weight * velocity_reward) + \
                       (z_pos_weight * z_pos_reward) + \
                       (orientation_weight * orientation_reward) + \
                       (dof_limit_penalty_weight * dof_limit_penalty) + \
                       (smoothness_weight * action_smoothness)
        
        # Reward components breakdown
        reward_components = {
            'velocity': velocity_weight * velocity_reward,
            'z_pos': z_pos_weight * z_pos_reward,
            'orientation': orientation_weight * orientation_reward,
            'dof_limit_penalty': dof_limit_penalty_weight * dof_limit_penalty,
            'action_smoothness': smoothness_weight * action_smoothness
        }
        
        return total_reward, reward_components
