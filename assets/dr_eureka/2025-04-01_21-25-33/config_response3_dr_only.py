        # Since the robot will encounter different terrains with varying frictions, randomizing friction is crucial for realistic training.
        # However, extreme values may not be as common, so a moderate range could simulate a variety of common surfaces.
        friction_range = [0.2, 0.8]
        
        # Restitution (bounciness) varies less in practical scenarios, especially for running, but slight randomization can help adapt to unexpected impacts.
        restitution_range = [0.3, 0.6]
        
        # Added mass simulates carrying variable loads or dealing with different body weights due to, for example, mud. However, large values may impair learning efficiency.
        added_mass_range = [0.0, 3.0]
        
        # Center of mass displacement is critical for balance, particularly over uneven terrain. Small disturbances are reasonable.
        com_displacement_range = [-0.05, 0.05]
        
        # Motor strength variability helps the robot adapt to situations where maximum power cannot be achieved, such as battery degradation.
        motor_strength_range = [0.8, 1.5]
        
        # Control gains (Kp and Kd) tuning for variability in the robot's response to control inputs; slight variations can represent different hardware calibrations.
        Kp_factor_range = [1.0, 1.5]
        Kd_factor_range = [0.2, 1.5]
        
        # The following parameters are typically constants specific to the robot's physical configuration and might not vary significantly in real-world scenarios.
        # Thus, keeping them constant can help focus the learning process on more impactful aspects of the simulation.
        dof_stiffness_range = [0.1, 0.1]
        dof_damping_range = [0.01, 0.01]
        dof_friction_range = [0.01, 0.01]
        dof_armature_range = [0.01, 0.01]
        
        # External push forces help the model learn to recover from disturbances or unexpected forces.
        push_vel_xy_range = [0.0, 3.0]
        
        # Gravity should remain consistent with real-world physics for a realistic simulation. However, slight variations can simulate running on slight inclines or declines.
        gravity_range = [-0.1, 0.1]  # Extends the gravity vector slightly in any direction to simulate minor inclines or declines.
