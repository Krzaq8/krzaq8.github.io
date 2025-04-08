    def compute_reward(self):
        env = self.env  # Do not skip this line. Afterwards, use env.{parameter_name} to access parameters of the environment.
        
        # Define weights for different parts of the reward
        velocity_weight = 2.0  # Increased emphasis on reaching desired velocity
        z_pos_weight = 0.2  # Reduced weight keeping important without dominating
        orientation_weight = 1.0  # Important for posture
        dof_limit_penalty_weight = 0.05  # Reduced weight to balance with other rewards
        smoothness_weight = 0.1  # Decreased to reduce dominance
        
        # Desired values
        desired_velocity_x = 2.0  # m/s
        desired_z_pos = 0.34  # meters
    
        # Calculate velocity error using squared error for stronger gradient when far from goal
        actual_velocity_x = env.root_states[:, 7]  # Global linear velocity x component
        velocity_error = (actual_velocity_x - desired_velocity_x) ** 2
    
        # Calculate Z position error
        actual_z_pos = env.root_states[:, 2]  # Global Z position
        z_pos_error = (actual_z_pos - desired_z_pos) ** 2
    
        # Calculate orientation error with dot-product for clearer directionality
        desired_orientation = torch.tensor([1., 0., 0., 0.], device=env.device)  # Align orientation with gravity
        orientation_error = 1 - torch.abs(torch.sum(env.base_quat * desired_orientation, dim=1)) ** 2
    
        # Simplified DOF limit penalty to emphasize staying within soft limits without complex calculations
        dof_violations = torch.relu(torch.abs(env.dof_pos - env.default_dof_pos).sum(dim=1) - 1.0)
    
        # Action smoothness incentive
        action_smoothness = torch.mean(torch.abs(env.actions - env.last_actions) ** 2, dim=1)
    
        # Calculate total reward with new scaling and adjustments
        reward = velocity_weight * torch.exp(-velocity_error) + \
                 z_pos_weight * torch.exp(-z_pos_error) + \
                 orientation_weight * (1 - orientation_error) - \
                 dof_limit_penalty_weight * dof_violations - \
                 smoothness_weight * action_smoothness
    
        # Construct detailed reward components for debugging/analysis
        reward_components = {
            'velocity': velocity_weight * torch.exp(-velocity_error),
            'z_pos': z_pos_weight * torch.exp(-z_pos_error),
            'orientation': orientation_weight * (1 - orientation_error),
            'dof_limit_penalty': -dof_limit_penalty_weight * dof_violations,
            'action_smoothness': -smoothness_weight * action_smoothness
        }
    
        return reward, reward_components
