        # Friction varies significantly between different surfaces like asphalt, grass, tiles, etc.
        friction_range = [0.2, 0.8]
        
        # Restitution affects how much energy is lost in collisions, important but real-world surfaces won't vary widely in this aspect.
        restitution_range = [0.2, 0.5]
        
        # Added mass simulates carrying a load, or the effect of mud on legs, so keep this modest to ensure adaptability without overwhelming the learning process.
        added_mass_range = [0.0, 2.0]
        
        # Center of mass displacement can greatly affect stability, but too much variation is unrealistic.
        com_displacement_range = [-0.05, 0.05]
        
        # Motor strength varies especially under load or battery condition, but the range shouldn't be too wide to ensure learning stability.
        motor_strength_range = [0.8, 1.5]
        
        # Proportional (Kp) and Derivative (Kd) gain factors affect control response, significant for adaptation but keep in a reasonable range.
        Kp_factor_range = [1.0, 1.5]
        Kd_factor_range = [0.5, 1.0]
        
        # Keeping DOF stiffness and damping constant as they are typically hardware constants and do not vary.
        dof_stiffness_range = [0.1, 0.1]
        dof_damping_range = [0.01, 0.01]
        
        # Joint friction, armature are hardware specific and unlikely to vary in real-world deployment scenarios.
        dof_friction_range = [0.01, 0.01]
        dof_armature_range = [0.01, 0.01]
        
        # Simulating external disturbances can be crucial for real-world adaptability. This encompasses sudden pushes or wind.
        push_vel_xy_range = [0.0, 3.0]
        
        # Gravity simulation should be kept realistic with minor variations to simulate small inclines or declines in terrain.
        gravity_range = [-0.2, 0.2]  # Small adjustments to gravity to account for variations in terrain and elevation.
