    def compute_reward(self):
        env = self.env  # Do not skip this line. Afterwards, use env.{parameter_name} to access parameters of the environment.
    
        # Velocity reward: Incentivize the robot to maintain a forward velocity of 2.0 m/s
        target_velocity = 2.0  # Target velocity in m/s
        velocity_x = env.root_states[:, 7]  # Extracting the x-component of the linear velocity
        velocity_rew = -torch.abs(velocity_x - target_velocity)
        
        # Height reward: Make sure the torso remains at a target z-height of 0.34 meters
        target_height = 0.34  # Target height in meters
        current_height = env.root_states[:, 2]  # Extracting the z-component of the position
        height_rew = -torch.abs(current_height - target_height)
    
        # Orientation reward: Ensure robot's body is upright; perpendicular to gravity
        gravity_down = torch.tensor([0., 0., 1.], device=env.device)  # Assuming gravity is in the positive Z direction
        robot_down = env.projected_gravity  # Projected gravity direction in robot's frame
        orientation_rew = -torch.norm(robot_down - gravity_down, dim=1)
    
        # Action smoothness reward: Penalize large changes in action values to ensure smooth leg movements
        action_diff = env.actions - env.last_actions
        smoothness_rew = -torch.norm(action_diff, dim=1)
    
        # Joint limit avoidance: Penalize positions close to the soft limits
        dof_mid = (env.dof_pos_limits[:, 0] + env.dof_pos_limits[:, 1]) / 2.0
        dof_range = env.dof_pos_limits[:, 1] - env.dof_pos_limits[:, 0]
        norm_dof_pos = torch.abs((env.dof_pos - dof_mid) / (dof_range / 2.0))  # Normalize to [-1, 1]
        joint_limit_rew = -torch.mean(torch.max(torch.zeros_like(norm_dof_pos), norm_dof_pos - 0.8), dim=1)  # Punish when beyond 80% of range
    
        # Aggregate rewards
        reward_components = {
            "velocity_rew": velocity_rew,
            "height_rew": height_rew,
            "orientation_rew": orientation_rew,
            "smoothness_rew": smoothness_rew,
            "joint_limit_rew": joint_limit_rew
        }
    
        # Weight individual components (tune as needed)
        weights = {
            "velocity_rew": 1.0,
            "height_rew": 0.5,
            "orientation_rew": 0.2,
            "smoothness_rew": 0.3,
            "joint_limit_rew": 0.1
        }
    
        total_reward = sum(weights[comp] * reward_components[comp] for comp in reward_components)
    
        return total_reward, reward_components
