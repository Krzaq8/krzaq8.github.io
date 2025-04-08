    def compute_reward(self):
        env = self.env  # Do not skip this line. Use env to access the parameters.
    
        # Initialize reward components
        velocity_reward = 0.0
        position_reward = 0.0
        orientation_reward = 0.0
        smoothness_reward = 0.0
        dof_limit_penalty = 0.0
    
        # Velocity reward for moving forward at 2 m/s
        target_velocity = 2.0  # target velocity in m/s
        x_velocity = env.root_states[:, 7]  # X component of the velocity
        velocity_reward = -torch.abs(x_velocity - target_velocity)
    
        # Position reward for keeping the torso at z=0.34
        target_height = 0.34  # m
        z_position = env.root_states[:, 2]  # Z component of the root position
        position_reward = -torch.abs(z_position - target_height)
    
        # Orientation reward for keeping the robot perpendicular to gravity
        down_vector = torch.tensor([0, 0, 1], dtype=torch.float, device=self.env.device)
        robot_down = env.projected_gravity / torch.norm(env.projected_gravity, dim=1, keepdim=True)
        orientation_cos_angle = torch.sum(robot_down * down_vector, dim=1)
        orientation_reward = orientation_cos_angle
    
        # Smoothness reward for minimal action difference
        action_diff = env.actions - env.last_actions
        smoothness_reward = -torch.mean(torch.abs(action_diff), dim=1)
    
        # Penalty for exceeding DOF limits
        dof_exceed_lower = torch.lt(env.dof_pos, env.dof_pos_limits[:, 0])
        dof_exceed_upper = torch.gt(env.dof_pos, env.dof_pos_limits[:, 1])
        dof_limit_violations = torch.any(torch.logical_or(dof_exceed_lower, dof_exceed_upper), dim=1)
        dof_limit_penalty = -dof_limit_violations.to(torch.float)
    
        # Calculate total reward
        reward = velocity_reward + position_reward + orientation_reward + smoothness_reward + dof_limit_penalty
    
        # Construct reward components dictionary
        reward_components = {
            'velocity_reward': velocity_reward,
            'position_reward': position_reward,
            'orientation_reward': orientation_reward,
            'smoothness_reward': smoothness_reward,
            'dof_limit_penalty': dof_limit_penalty
        }
    
        return reward, reward_components
