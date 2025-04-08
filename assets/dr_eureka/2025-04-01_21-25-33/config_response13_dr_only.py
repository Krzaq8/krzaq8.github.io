        # The friction between the robot feet and the surface plays a critical role in both stability and mobility. 
        # Since real-world terrains can vary greatly (e.g., smooth tile floors, rough asphalt, grassy fields), 
        # a moderate range can prepare the robot for these conditions without making the learning problem intractably difficult.
        friction_range = [0.2, 0.9]
        
        # Restitution affects how bouncy interactions are, which might not vary too much in realistic terrains, especially for a heavy quadruped.
        # A narrower range focuses on more realistic interaction with the ground, where noticeable energy is lost upon impact.
        restitution_range = [0.3, 0.6]
        
        # Added mass can simulate the effect of carrying varying loads or differences in the robot's weight due to battery depletion or equipment carried.
        # A small range suffices to teach the model robustness to weight variations without overly complicating the task.
        added_mass_range = [0.0, 2.0]
        
        # The center of mass (com) displacement might change slightly with different payloads,
        # but extreme changes can make the model learn unrealistic body dynamics. Hence, a small, realistic adjustment range is chosen.
        com_displacement_range = [0.0, 0.5]
        
        # Motor strength variability is critical, as motors do not always perform consistently, especially under different loads and after wear.
        # This range should encompass potential weakening or slight overperformance.
        motor_strength_range = [0.75, 1.5]
        
        # The Kp (proportional gain) and Kd (derivative gain) factors directly influence control responsiveness.
        # Slight variations in these can simulate different control hardware or settings.
        Kp_factor_range = [0.95, 1.5]
        Kd_factor_range = [0.1, 1.5]
        
        # Stiffness, damping, and friction within the degrees of freedom (dof) are typically constant for a given hardware setup,
        # and varying these would not represent real-world scenarios. These parameters are thus kept constant.
        dof_stiffness_range = [0.1, 0.1]
        dof_damping_range = [0.01, 0.01]
        dof_friction_range = [0.01, 0.01]
        
        # The armature adds nonlinearity to the motor dynamics; however, for a given robot design, this would be a constant.
        dof_armature_range = [0.01, 0.01]
        
        # An external push helps in learning recovery strategies, simulating interactions with moving obstacles or wind.
        # However, the force should not be overly disruptive, to focus on locomotion under typical conditions.
        push_vel_xy_range = [0.0, 2.0]
        
        # Gravity changes are unrealistic in most scenarios but small variations can simulate lighter or heavier effective weights,
        # such as operating on a slight incline or under unusual atmospheric conditions.
        gravity_range = [-0.1, 0.1]
