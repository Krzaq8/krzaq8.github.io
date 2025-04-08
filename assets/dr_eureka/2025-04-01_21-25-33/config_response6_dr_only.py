        # Friction is crucial for realistic interaction with different terrains.
        # I'll use a broad range to cover various terrains, from slippery ice to rough asphalt.
        friction_range = [0.2, 0.9]
        
        # Restitution (bounciness) is less critical for running but might affect how the robot interacts with unexpected obstacles or uneven terrain.
        # A moderate range should suffice, as extreme values wouldn’t be common.
        restitution_range = [0.3, 0.7]
        
        # Added mass can simulate carrying different loads or the effect of the robot's weight variance.
        # A small range around zero will help learn stability without drastically changing dynamics.
        added_mass_range = [0.0, 2.0]
        
        # Center of mass displacement is critical for balance and stability under various payloads or body configurations.
        # Keeping this range conservative will aid in learning a stable gait under slight variations.
        com_displacement_range = [-0.05, 0.05]
        
        # Motor strength variation simulates different motor performances and wear over time.
        # Covering a range around the nominal value ensures the learned policy is robust to motor performance changes.
        motor_strength_range = [0.8, 1.2]
        
        # Kp (proportional gain) and Kd (derivative gain) factors directly affect the control loop's responsiveness and stability.
        # Small variations will help ensure the robot can adapt to different dynamics without becoming uncontrollable.
        Kp_factor_range = [0.95, 1.5]
        Kd_factor_range = [0.1, 1.0]
        
        # Stiffness, damping, friction, and armature of degrees of freedom (DoFs) should likely remain constant,
        # as these are intrinsic mechanical properties that are not expected to vary significantly in real-world scenarios.
        dof_stiffness_range = [0.1, 0.1]  # Keeping constant as suggested.
        dof_damping_range = [0.01, 0.01]  # Keeping constant as suggested.
        dof_friction_range = [0.01, 0.01]  # Keeping constant as suggested.
        dof_armature_range = [0.01, 0.01]  # Keeping constant as suggested.
        
        # Random external pushes can help train the robot to recover from disturbances.
        # Small to moderate disturbances will simulate realistic scenarios like being nudged or impacted by small objects.
        push_vel_xy_range = [0.0, 3.0]
        
        # Gravity variation can simulate slight inclines or declines and other anomalies in terrain, enhancing adaptability.
        # However, extreme deviations from Earth's gravity (-9.8 m/s^2) are unrealistic, so we apply only slight modifications.
        gravity_range = [-0.2, 0.2]  # Adds variability without unrealistic scenarios.
