    def compute_reward(self):
        env = self.env  # Do not skip this line. Afterwards, use env.{parameter_name} to access parameters of the environment.
    
        # Velocity reward component
        target_velocity_x = 2.0  # Target velocity in m/s in the x direction
        x_velocity = env.root_states[:, 7]  # Linear velocity of the base in the x direction
        velocity_error = (x_velocity - target_velocity_x).abs()
        velocity_reward = torch.exp(-velocity_error * 5.0)  # Exponential decay based on the error
    
        # Torso height reward component
        target_z_position = 0.34  # Target z position of the torso in meters
        z_position = env.root_states[:, 2]  # Z position of the base
        height_error = (z_position - target_z_position).abs()
        height_reward = torch.exp(-height_error * 100.0)  # Exponential decay based on the error
    
        # Orientation reward component
        target_projected_gravity = torch.tensor([0., 0., 1.], device=env.device).repeat((env.num_envs, 1))  # Down direction in the robot frame when perfectly upright
        gravity_alignment_error = torch.norm(env.projected_gravity - target_projected_gravity, dim=1)
        orientation_reward = torch.exp(-gravity_alignment_error * 100.0)  # Exponential decay based on the error
    
        # DOF limit avoidance reward component
        dof_limit_reward = torch.tensor(0.0).to(env.device)
        soft_limit_violations = torch.lt(env.dof_pos, env.dof_pos_limits[:, 0]) | torch.gt(env.dof_pos, env.dof_pos_limits[:, 1])
        dof_limit_reward -= soft_limit_violations.sum(dim=1).float() * 0.1  # Penalize for each violation
    
        # Smoothness reward component (Minimize the rate of change of actions)
        action_smoothness_reward = -torch.norm(env.actions - env.last_actions, dim=1) * 0.01
    
        # Total reward
        total_reward = 0.5 * velocity_reward + 0.2 * height_reward + 0.2 * orientation_reward + 0.05 * dof_limit_reward + 0.05 * action_smoothness_reward
    
        env.extras['velocity_error'] = velocity_error.mean()
        env.extras['height_error'] = height_error.mean()
        env.extras['gravity_alignment_error'] = gravity_alignment_error.mean()
        env.extras['dof_limit_violations'] = soft_limit_violations.float().mean()
        env.extras['action_smoothness'] = action_smoothness_reward.mean()
    
        return total_reward, env.extras
