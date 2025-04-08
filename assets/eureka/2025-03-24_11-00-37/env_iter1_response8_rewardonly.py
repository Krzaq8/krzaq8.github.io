    def compute_reward(self):
        env = self.env  # Do not skip this line. Afterwards, use env.{parameter_name} to access parameters of the environment.
    
        # Adjusted weights
        velocity_weight = 1.0
        z_pos_weight = 0.3
        orientation_weight = 0.1
        dof_limit_penalty_weight = 0.05
        smoothness_weight = 0.05
    
        # Desired values
        desired_velocity_x = 2.0
        desired_z_pos = 0.34
        desired_orientation = torch.tensor([0., 0., 0., 1.], device=env.device)
    
        # Velocity error - emphasis on exact match using squared error
        velocity_error = (env.root_states[:, 7] - desired_velocity_x) ** 2
    
        # Z position error - minor refinement
        z_pos_error = (env.root_states[:, 2] - desired_z_pos) ** 2
    
        # Orientation error adjustment
        orientation_error = 1 - torch.abs(torch.sum(env.base_quat * desired_orientation, dim=1)) ** 2
    
        # Adjusted DOF limit penalty - less punitive
        dof_limit_penalty = torch.mean(torch.clamp(torch.abs(env.dof_pos - env.default_dof_pos) - (env.dof_pos_limits[:, 1] - env.dof_pos_limits[:, 0]).unsqueeze(0)/2, min=0), axis=1)
        
        # Smoothness - less weight
        action_smoothness = torch.mean(torch.abs(env.actions - env.last_actions), dim=1)
    
        # Total reward calculation with adjusted weights and formulas
        reward = velocity_weight * torch.exp(-velocity_error) + \
                 z_pos_weight * torch.exp(-z_pos_error) + \
                 orientation_weight * torch.exp(-orientation_error) - \
                 dof_limit_penalty_weight * dof_limit_penalty - \
                 smoothness_weight * action_smoothness
    
        reward_components = {
            'velocity': velocity_weight * torch.exp(-velocity_error),
            'z_pos': z_pos_weight * torch.exp(-z_pos_error),
            'orientation': orientation_weight * torch.exp(-orientation_error),
            'dof_limit_penalty': -dof_limit_penalty_weight * dof_limit_penalty,
            'action_smoothness': -smoothness_weight * action_smoothness
        }
    
        return reward, reward_components
