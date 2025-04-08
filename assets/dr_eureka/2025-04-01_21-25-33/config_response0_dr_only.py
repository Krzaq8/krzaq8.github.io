        # Friction affects how the robot's feet interact with different surfaces. Real-world terrains vary greatly in friction properties,
        # so we should allow for a broad range of values. However, too low values might make learning hard and unrealistic.
        friction_range = [0.1, 0.9]
        
        # Restitution affects how much energy is returned after a collision, like when the robot's feet hit the ground. Real-world materials
        # won't perfectly return energy, but considering indoor and outdoor varieties, a mid-range that represents common surfaces well is ideal.
        restitution_range = [0.2, 0.7]
        
        # Added mass simulates carrying different payloads. For a quadruped designed to be versatile, it should learn to manage a range of additional weights.
        # However, too much variation might hinder the learning process without added benefit.
        added_mass_range = [0.0, 3.0]
        
        # Center of mass displacement affects stability. A smaller range can simulate slight load imbalances or wear.
        com_displacement_range = [-0.05, 0.05]
        
        # Motor strength randomization ensures the robot can adapt to different terrains and slopes where different force outputs are necessary.
        motor_strength_range = [0.7, 1.5]
        
        # Kp (proportional gain) and Kd (derivative gain) factors affect the control system's responsiveness and stability.
        # Randomizing these within a moderate range can make the control system more robust to real-world conditions.
        Kp_factor_range = [1.0, 1.8]
        Kd_factor_range = [0.2, 1.5]
        
        # DOF (Degrees of Freedom) parameters such as stiffness, damping, friction, and armature, given their provided ranges are constants,
        # should not be randomized since they are already set to presumably optimal values for simulation and likely represent the physical robot accurately.
        # Hence, we keep them as is.
        dof_stiffness_range = [0.1, 0.1]
        dof_damping_range = [0.01, 0.01]
        dof_friction_range = [0.01, 0.01]
        dof_armature_range = [0.01, 0.01]
        
        # Push velocity in the xy plane simulates external forces such as wind or collisions with small objects.
        # This should have a range, but not too wide as to make the environment overly difficult or unrepresentative of typical conditions.
        push_vel_xy_range = [0.0, 2.0]
        
        # Gravity variations can simulate slight inclines or declines in the terrain as well as testing robustness to small variations in perceived gravity due to acceleration.
        # However, the range here should be small to avoid unrealistic scenarios.
        gravity_range = [-0.2, 0.2]
