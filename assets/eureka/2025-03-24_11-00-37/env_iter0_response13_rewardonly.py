    def compute_reward(self):
        env = self.env  # Do not skip this line. Afterwards, use env.{parameter_name} to access parameters of the environment.
    
        desired_velocity = 2.0  # m/s
        desired_height = 0.34  # meters
        velocity_reward_scale = 1.0
        height_reward_scale = 1.0
        orientation_reward_scale = 1.0
        smoothness_reward_scale = 0.5
        limit_avoidance_reward_scale = 0.5
    
        # Calculate the forward velocity error
        forward_velocity = env.root_states[:, 7]  # Global X velocity
        velocity_error = torch.abs(forward_velocity - desired_velocity)
    
        # Calculate height error
        current_height = env.root_states[:, 2]  # Z position
        height_error = torch.abs(current_height - desired_height)
    
        # Calculate orientation error (robot should be perpendicular to gravity, so projected gravity should be [0, 0, -1])
        desired_projected_gravity = torch.tensor([0., 0., -1.], device=env.device).unsqueeze(0).repeat((env.num_envs, 1))
        orientation_error = torch.norm(env.projected_gravity - desired_projected_gravity, dim=1)
    
        # Calculate action smoothness
        dof_vel_change = torch.abs(env.dof_vel - env.last_dof_vel)
        action_difference = torch.abs(env.actions - env.last_actions)
        smoothness_error = (dof_vel_change.mean(dim=1) + action_difference.mean(dim=1)) / 2.0
    
        # Calculate DOF limit avoidance
        dof_limit_violation = torch.clamp(torch.min(env.dof_pos - env.dof_pos_limits[:, 0].unsqueeze(0), env.dof_pos_limits[:, 1].unsqueeze(0) - env.dof_pos), min=0).mean(dim=1)
    
        # Calculate individual rewards (negative because they are errors)
        velocity_reward = -velocity_error * velocity_reward_scale
        height_reward = -height_error * height_reward_scale
        orientation_reward = -orientation_error * orientation_reward_scale
        smoothness_reward = -smoothness_error * smoothness_reward_scale
        limit_avoidance_reward = -dof_limit_violation * limit_avoidance_reward_scale
    
        # Aggregate rewards
        total_reward = velocity_reward + height_reward + orientation_reward + smoothness_reward + limit_avoidance_reward
    
        # Prepare rewards dictionary
        rewards_dict = {'velocity_reward': velocity_reward,
                        'height_reward': height_reward,
                        'orientation_reward': orientation_reward,
                        'smoothness_reward': smoothness_reward,
                        'limit_avoidance_reward': limit_avoidance_reward}
    
        return total_reward, rewards_dict
