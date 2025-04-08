    def compute_reward(self):
        env = self.env  # Do not skip this line. Afterwards, use env.{parameter_name} to access parameters of the environment.
    
        # Define weights for different parts of the reward
        velocity_weight = 1.0
        z_pos_weight = 1.0
        orientation_weight = 0.5
        dof_limit_penalty_weight = 0.1
        smoothness_weight = 0.05  # Reduced weight for action smoothness to lessen its impact
        
        # Desired values
        desired_velocity_x = 2.0  # m/s
        desired_z_pos = 0.34  # meters
        desired_orientation = torch.tensor([0., 0., 0., 1.], device=env.device)  # WXYZ quaternion for no rotation
    
        # Velocity reward, focusing on x-direction velocity
        velocity_error = torch.abs(env.root_states[:, 7] - desired_velocity_x)
        velocity_reward = torch.exp(-velocity_error)  # Using exponential decay instead of inverse proportion
    
        # Z position reward, focusing on maintaining a specific height
        z_pos_error = torch.abs(env.root_states[:, 2] - desired_z_pos)
        z_pos_reward = torch.exp(-z_pos_error)
    
        # Orientation reward, ensuring orientation is kept upright
        orientation_error = 1 - torch.abs(torch.sum(env.base_quat * desired_orientation, dim=1))
        orientation_reward = torch.exp(-orientation_error)
    
        # DOF limit penalty, discourage extreme joint angles but less punitive scaling
        dof_limit_penalty = torch.mean(torch.max(torch.zeros_like(env.dof_pos), torch.abs(env.dof_pos - env.default_dof_pos) - (env.dof_pos_limits[:, 1] - env.dof_pos_limits[:, 0]) * 0.5), dim=1)
    
        # Action smoothness reward, the smaller difference between actions, the better
        action_smoothness_penalty = torch.mean(torch.abs(env.actions - env.last_actions), dim=1)
    
        # Calculate total reward
        total_reward = velocity_weight * velocity_reward + \
                       z_pos_weight * z_pos_reward + \
                       orientation_weight * orientation_reward - \
                       dof_limit_penalty_weight * dof_limit_penalty - \
                       smoothness_weight * action_smoothness_penalty
    
        # Detailed reward components
        reward_components = {
            'velocity': velocity_reward,
            'z_pos': z_pos_reward,
            'orientation': orientation_reward,
            'dof_limit_penalty': -dof_limit_penalty,
            'action_smoothness': -action_smoothness_penalty
        }
    
        return total_reward, reward_components
