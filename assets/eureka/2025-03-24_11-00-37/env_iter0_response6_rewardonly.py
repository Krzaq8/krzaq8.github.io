    def compute_reward(self):
        env = self.env  # Do not skip this line
        
        # The desired forward velocity in the x direction (global frame).
        desired_velocity_x = 2.0  
        # The ideal z position of the robot's torso.
        desired_torso_z_pos = 0.34
        
        # Extract the relevant variables from the environment
        root_linear_velocity = env.root_states[:, 7:10]  # Linear velocity of the base [x, y, z]
        torso_z_pos = env.root_states[:, 2]  # Z position of the torso
        base_orientation_error = torch.abs(env.projected_gravity - torch.tensor([0, 0, -1.], device=env.device).repeat((env.num_envs, 1)))  # Measure of orientation error
        
        # Calculate velocity error (we want the x component of the velocity to be as close as possible to desired_velocity_x)
        velocity_error = torch.abs(root_linear_velocity[:, 0] - desired_velocity_x)
    
        # Calculate torso position error (we want the z position to be as close as possible to desired_torso_z_pos)
        torso_pos_error = torch.abs(torso_z_pos - desired_torso_z_pos)
        
        # Calculate the smoothness of leg movements by the sum of squared velocities and distance from limit violation
        leg_movement_smoothness = -(torch.sum(env.dof_vel ** 2, dim=(1, 2)) + torch.sum(torch.max(env.dof_pos - env.dof_pos_limits[:, 1], torch.zeros_like(env.dof_pos)) ** 2 + torch.max(env.dof_pos_limits[:, 0] - env.dof_pos, torch.zeros_like(env.dof_pos)) ** 2, dim=(1, 2)))
        
        # Overall orientation error considering both yaw and pitch from the base_orientation_error but ignore roll as it's not critical for forward movement
        orientation_error = torch.sum(base_orientation_error[:, :2], dim=1)
        
        # Define weights for the different components of the reward
        weights = {
            'velocity_error': -10.0,
            'torso_pos_error': -10.0,
            'leg_movement_smoothness': 0.1,
            'orientation_error': -1.0,
        }
    
        # Combine the components into a total reward
        reward = weights['velocity_error'] * velocity_error + \
                 weights['torso_pos_error'] * torso_pos_error + \
                 weights['leg_movement_smoothness'] * leg_movement_smoothness + \
                 weights['orientation_error'] * orientation_error
        
        # Store each reward component for debugging and analysis
        component_rewards = {
            'velocity_error': velocity_error,
            'torso_pos_error': torso_pos_error,
            'leg_movement_smoothness': leg_movement_smoothness,
            'orientation_error': orientation_error,
        }
        
        # While training in simulation, the actual values of these variables can be slightly different due to simulation inaccuracies.
        
        return reward, component_rewards
