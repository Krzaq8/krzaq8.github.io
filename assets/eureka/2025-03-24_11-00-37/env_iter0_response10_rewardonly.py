    def compute_reward(self):
        env = self.env  # Do not skip this line. Afterwards, use env.{parameter_name} to access parameters of the environment.
    
        # Desired forward velocity and altitude
        desired_velocity = 2.0
        desired_z_position = 0.34
    
        # Calculate velocity error
        forward_velocity = env.root_states[:, 7]
        velocity_error = torch.abs(forward_velocity - desired_velocity)
    
        # Torso altitude error
        z_position_error = torch.abs(env.root_states[:, 2] - desired_z_position)
    
        # Orientation error: Measuring how perpendicular the orientation is to gravity
        # The more aligned the projected_gravity with the global z-axis, the more upright the robot is
        projected_gravity = env.projected_gravity
        uprightness = torch.abs(projected_gravity[:, 2])  # Want this to be close to 1
    
        # Stability: minimize the change in actions to ensure smooth leg movements and penalize large torques
        action_diff = torch.mean(torch.abs(env.actions - env.last_actions), dim=1)
        torque_penalty = torch.mean(env.torques.abs(), dim=1)
    
        # DOF position limit proximity: Compute how close any joint is to reaching its soft limit
        dof_limit_proximity = torch.max(
            (env.dof_pos - env.dof_pos_limits[:, 0]) / (env.dof_pos_limits[:, 1] - env.dof_pos_limits[:, 0]),
            dim=1
        ).values
        # Invert since we want proximity to be low (far from limits)
        dof_limit_proximity = 1 - dof_limit_proximity
    
        # Reward components
        velocity_reward = (1.0 - torch.tanh(10.0 * velocity_error)) * 2.0  # Encourage exactness in velocity
        z_position_reward = (1.0 - torch.tanh(10.0 * z_position_error)) * 2.0  # Encourage maintaining height
        uprightness_reward = torch.pow(uprightness, 2) * 2.0  # Encourage uprightness
        action_smoothing_reward = (1.0 - torch.tanh(10.0 * action_diff)) * 0.5  # Encourage action smoothness
        torque_penalty_reward = (1.0 - torch.sigmoid(torque_penalty)) * 0.5  # Penalize large torques
        dof_limit_avoidance_reward = torch.pow(dof_limit_proximity, 2) * 0.5  # Encourage staying away from DOF limits
    
        # Total reward
        reward = velocity_reward + z_position_reward + uprightness_reward + action_smoothing_reward + torch_penalty_reward + dof_limit_avoidance_reward
    
        # Debug info
        reward_components = {
            "velocity_reward": velocity_reward,
            "z_position_reward": z_position_reward,
            "uprightness_reward": uprightness_reward,
            "action_smoothing_reward": action_smoothing_reward,
            "torque_penalty_reward": torque_penalty_reward,
            "dof_limit_avoidance_reward": dof_limit_avoidance_reward,
        }
    
        return reward, reward_components
