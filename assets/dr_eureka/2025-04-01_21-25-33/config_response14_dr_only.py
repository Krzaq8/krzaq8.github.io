        # Friction is crucial for different terrains as it affects the grip of the robot's feet on the surface.
        friction_range = [0.2, 0.8]  # Choose a mid-range to simulate both slippery and grippy surfaces.
        
        # Restitution affects how much objects bounce off each other, which might not be critical for running but can impact interaction with uneven terrains.
        restitution_range = [0.1, 0.5]  # Slight variability to account for changes in terrain density but keep it low to avoid unrealistic bouncing.
        
        # Added mass can simulate carrying varying loads or changes in the robot's mass due to different equipment for indoor vs. outdoor.
        added_mass_range = [0.0, 3.0]  # Allow for some variability without making the robot too heavy or unrealistically light.
        
        # Center of mass displacement affects stability and agility. Real-world scenarios might cause slight shifts in the robot's weight distribution.
        com_displacement_range = [-0.05, 0.05]  # Small range to ensure the robot learns to compensate for minor weight shifts without destabilizing.
        
        # Motor strength is critical as it might vary with wear and tear or battery level changes in the real world.
        motor_strength_range = [0.8, 1.5]  # Ensuring that the robot can operate under different power levels.
        
        # PID controller parameters (Kp and Kd factors) influence the robot's movement control. Real systems might need adjustments due to wear or environmental factors.
        Kp_factor_range = [1.0, 1.5]  # To adapt to variance in control response.
        Kd_factor_range = [0.5, 1.5]  # To deal with changes in damping requirements.
        
        # Stiffness, damping, friction, and armature of DOF are likely to be constant in a well-designed robot; randomizing these might only introduce unnecessary noise.
        # Hence, kept as constants.
        dof_stiffness_range = [0.1, 0.1]  # No variability needed.
        dof_damping_range = [0.01, 0.01]  # Keep consistent to avoid artificial behavior.
        dof_friction_range = [0.01, 0.01]  # Constant to ensure consistency in joint movements.
        dof_armature_range = [0.01, 0.01]  # No change as it's more of a design-specific parameter.
        
        # Push velocity in x and y can simulate sudden impacts or forces the robot might encounter, like wind or collisions.
        push_vel_xy_range = [0.0, 1.0]  # Small magnitude to simulate realistic disturbances without overwhelming the robot.
        
        # Gravity changes can simulate slight variations in elevation or to prepare the robot for possible deployment on different planets or simulation inaccuracies.
        gravity_range = [-0.2, 0.2]  # Small changes around earth's gravity to ensure adaptability to minor variances.
