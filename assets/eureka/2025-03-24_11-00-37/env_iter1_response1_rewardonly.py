    def compute_reward(self):
        env = self.env  # Do not skip this line. Afterwards, use env.{parameter_name} to access parameters of the environment.
        
        # Define weights for different parts of the reward
        velocity_weight = 1.0
        z_pos_weight = 0.5
        orientation_weight = 0.5
        dof_limit_penalty_weight = 0.05  # Reduced weighting
        smoothness_weight = 0.1  # Reduced weighting for less dominance
    
        # Desired values
        desired_velocity_x = 2.0  # m/s
        desired_z_pos = 0.34  # meters
        desired_orientation = torch.tensor([0., 0., 0., 1.], device=env.device)  # WXYZ quaternion for no rotation
    
        # Calculate velocity reward, focus on maintaining desired velocity
        velocity_error = torch.abs(env.root_states[:, 7] - desired_velocity_x)
        velocity_reward = torch.exp(-velocity_weight * velocity_error)
        
        # Calculate Z position reward, encourage maintaining desired z position
        z_pos_error = torch.abs(env.root_states[:, 2] - desired_z_pos)
        z_pos_reward = torch.exp(-z_pos_weight * z_pos_error)
    
        # Calculate Orientation reward, encourage orientation to remain perpendicular to gravity
        orientation_error = 1 - torch.abs(torch.sum(env.base_quat * desired_orientation, dim=1))
        orientation_reward = torch.exp(orientation_weight * (1 - orientation_error))
    
        # Calculate DOF limit penalty, encourage smooth movement within soft limits
        dof_limit_penalty = torch.mean(torch.clamp(torch.abs(env.dof_pos - env.default_dof_pos) - (env.dof_pos_limits[:, 1] - env.dof_pos_limits[:, 0]).unsqueeze(0)/2, min=0.0), axis=1)
        dof_limit_penalty_reward = torch.exp(-dof_limit_penalty_weight * dof_limit_penalty)
    
        # Calculate Action smoothness reward, incentivize gradual changes in actions
        action_smoothness = torch.mean(torch.abs(env.actions - env.last_actions), dim=1)
        smoothness_reward = torch.exp(-smoothness_weight * action_smoothness)
    
        # Calculate total reward with new scaling
        reward = velocity_reward + z_pos_reward + orientation_reward + dof_limit_penalty_reward + smoothness_reward
    
        # Construct detailed reward components for debugging/analysis
        reward_components = {
            'velocity_reward': velocity_reward,
            'z_pos_reward': z_pos_reward,
            'orientation_reward': orientation_reward,
            'dof_limit_penalty_reward': dof_limit_penalty_reward,
            'smoothness_reward': smoothness_reward
        }
    
        return reward, reward_components
