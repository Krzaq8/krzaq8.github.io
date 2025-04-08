    def compute_reward(self):
        env = self.env
        
        # Adjusted weights for reward components based on feedback
        velocity_weight = 1.5
        z_pos_weight = 0.2
        orientation_weight = 0.1  # Reduced to balance out the impact
        dof_limit_penalty_weight = -0.02  # Reduce the impact
        smoothness_weight = -0.005  # Significantly reduce to allow for explorative actions
        
        # Velocity reward focused more directly on matching the specific target velocity
        desired_velocity_x = 2.0
        velocity_reward = -torch.abs(env.root_states[:, 7] - desired_velocity_x) * velocity_weight
        
        # Z position reward adjusted for directness and simplicity
        desired_z_pos = 0.34
        z_pos_reward = -(torch.abs(env.root_states[:, 2] - desired_z_pos)) * z_pos_weight
        
        # Orientation reward only penalizes significant deviations
        desired_up = torch.tensor([0., 0., 1., 0.], device=env.device)  # Representing upright orientation
        current_up = quat_rotate(env.base_quat, desired_up)
        orientation_reward = torch.sum(current_up * env.projected_gravity, dim=1) * orientation_weight
        
        # Simplify dof limit penalty to encourage staying within a comfortable range without overly harsh penalizations
        dof_limit_penalty = torch.sum(torch.clamp(torch.abs(env.dof_pos - env.default_dof_pos) - (env.dof_pos_limits[:, 1] - env.dof_pos_limits[:, 0]).unsqueeze(0) / 2, min=0), axis=1) * dof_limit_penalty_weight
        
        # Action smoothness to encourage smoother transitions without overly penalizing necessary quick adjustments
        action_smoothness = torch.mean(torch.abs(env.actions - env.last_actions), dim=1) * smoothness_weight
        
        # Calculating total reward
        reward = velocity_reward + z_pos_reward + orientation_reward + dof_limit_penalty + action_smoothness
        
        reward_components = {
            'velocity_reward': velocity_reward,
            'z_pos_reward': z_pos_reward,
            'orientation_reward': orientation_reward,
            'dof_limit_penalty': dof_limit_penalty,
            'action_smoothness': action_smoothness,
        }
        
        return reward, reward_components
