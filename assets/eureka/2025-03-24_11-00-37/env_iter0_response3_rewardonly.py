    def compute_reward(self):
        env = self.env  # Do not skip this line. Afterwards, use env.{parameter_name} to access parameters of the environment.
        
        # Velocity reward - aiming for 2.0m/s in the positive x direction
        forward_velocity = env.root_states[:, 7]  # Extract x component of the velocity
        velocity_reward = -torch.abs(forward_velocity - 2.0)  # Negative absolute difference from target
        
        # Torso height reward - aim for z position of 0.34m
        torso_height = env.root_states[:, 2]  # Extract z component of the position
        torso_height_reward = -torch.abs(torso_height - 0.34)
        
        # Orientation reward - torso should be perpendicular to gravity (projected_gravity should align with [0,0,-1])
        desired_projection = to_torch([0., 0., -1.], device=env.device).repeat((env.num_envs, 1))
        orientation_error = torch.norm(env.projected_gravity - desired_projection, dim=1)
        orientation_reward = -orientation_error
        
        # Smoothness reward - punish the change rate of actions
        action_diff = env.actions - env.last_actions
        smoothness_reward = -torch.norm(action_diff, p=2, dim=1)
        
        # DOF limit avoidance reward
        dof_limit_reward = torch.zeros(env.num_envs, device=env.device)
        for i in range(env.num_dof):
            lower_lim = env.dof_pos_limits[i, 0]
            upper_lim = env.dof_pos_limits[i, 1]
            dof_pos = env.dof_pos[:, i]
            # Subtract margin rewards for positions approaching the limits
            margin = 0.1 * (upper_lim - lower_lim)
            dof_limit_reward -= torch.clamp(lower_lim + margin - dof_pos, min=0)
            dof_limit_reward -= torch.clamp(dof_pos - (upper_lim - margin), min=0)
        
        # Combining the rewards with different weights 
        reward = 0.5 * velocity_reward + \
                 0.2 * torso_height_reward + \
                 0.2 * orientation_reward + \
                 0.05 * smoothness_reward + \
                 0.05 * dof_limit_reward
        
        # Return the final reward and the dictionary of reward components
        return reward, {
            'velocity_reward': velocity_reward,
            'torso_height_reward': torso_height_reward,
            'orientation_reward': orientation_reward,
            'smoothness_reward': smoothness_reward,
            'dof_limit_reward': dof_limit_reward
        }
