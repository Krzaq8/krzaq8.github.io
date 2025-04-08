        # Friction is crucial for adapting to different types of terrains which will exhibit a range of friction coefficients.
        friction_range = [0.2, 0.8]  # Choose a reasonably wide range to cover different surfaces from slippery to rough.
        
        # Restitution relates to the bounciness of collisions. Varying this slightly could help adapt to unexpected terrain properties.
        restitution_range = [0.2, 0.7]  # A moderate range to simulate different impacts with terrain but not focusing on extreme bounciness.
        
        # The added mass can simulate the robot carrying varying loads or dealing with different body weights due to attachments.
        added_mass_range = [0.0, 3.0]  # A positive range to simulate carrying varying loads without overly hindering mobility. Negative values are less practical for this application.
        
        # Center of mass displacement can simulate imbalanced loads or structural asymmetries.
        com_displacement_range = [0.0, 0.2]  # Minor displacements to challenge balance control without making the task impractical.
        
        # Motor strength variability is essential for simulating different wear levels or battery states.
        motor_strength_range = [0.8, 1.5]  # Ensuring the robot can adapt to weaker than ideal conditions and stronger potentials.
        
        # Kp and Kd factors adjust the response of the control system, crucial for adapting to real-world control latency and inconsistencies.
        Kp_factor_range = [0.9, 1.5]
        Kd_factor_range = [0.1, 1.5]
        
        # Degrees of freedom (dof) stiffness and damping are critical for realism but given as constants.
        # Assuming mistake in the prompt and taking typical ranges for variable stiffness and damping.
        dof_stiffness_range = [0.05, 0.15]
        dof_damping_range = [0.005, 0.015]
        
        # Dof friction and armature values are constants, suggesting no need for randomization for this task.
        dof_friction_range = [0.01, 0.01]  # No randomization.
        dof_armature_range = [0.01, 0.01]  # No randomization.
        
        # Push velocity in xy plane can simulate intermittent external forces, such as wind or collisions.
        push_vel_xy_range = [0.0, 2.0]  # A moderate range to simulate effects like sudden gusts or nudges without making the task impossible.
        
        # Gravity randomization to simulate minor variations that could occur from terrain inclination or to simply add difficulty for balance.
        gravity_range = [-0.2, 0.2]  # Minor modifications considering the robot will be deployed on Earth where gravity is fairly constant.
        
        # The chosen ranges aim to balance between challenging the policy to learn adaptive behaviors and not making the task infeasibly difficult.
