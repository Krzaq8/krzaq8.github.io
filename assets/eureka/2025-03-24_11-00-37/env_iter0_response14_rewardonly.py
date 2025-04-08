    def compute_reward(self):
        env = self.env  # Do not skip this line. Afterwards, use env.{parameter_name} to access parameters of the environment.
    
        desired_velocity = 2.0
        desired_height = 0.34
        # Velocity reward: encourage the agent to match the target forward velocity.
        velocity_reward = -torch.abs(env.root_states[:, 7] - desired_velocity)
    
        # Height reward: encourage the agent to maintain torso height close to desired level.
        height_penalty = -torch.abs(env.root_states[:, 2] - desired_height)
    
        # Orientation reward: encourage the agent to maintain upright orientation.
        down_vec = torch.tensor([0., 0., -1.], device=env.device).unsqueeze(0).repeat((env.num_envs, 1))
        current_orientation = env.projected_gravity
        orientation_penalty = -torch.nn.functional.cosine_similarity(current_orientation, down_vec, dim=1)
    
        # DOF limits penalty: discourage extreme positions to ensure smooth movement and joint safety.
        dof_limit_penalty = torch.sum(torch.max(env.dof_pos - env.dof_pos_limits[:, 1], torch.tensor(0., device=env.device)) + 
                                      torch.max(env.dof_pos_limits[:, 0] - env.dof_pos, torch.tensor(0., device=env.device)), dim=1)
    
        # Action rate penalty: minimize the change in action between steps to encourage smooth transitions.
        action_change = torch.abs(env.actions - env.last_actions).sum(dim=1)
        action_rate_penalty = -action_change
    
        # Aggregate rewards and penalties
        aggregate_reward = velocity_reward + height_penalty + orientation_penalty + dof_limit_penalty + action_rate_penalty
    
        reward = aggregate_reward
    
        return reward, {
            "velocity_reward": velocity_reward,
            "height_penalty": height_penalty,
            "orientation_penalty": orientation_penalty,
            "dof_limit_penalty": dof_limit_penalty,
            "action_rate_penalty": action_rate_penalty,
        }
