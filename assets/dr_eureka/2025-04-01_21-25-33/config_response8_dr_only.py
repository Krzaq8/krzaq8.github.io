        # Friction is crucial for simulating different terrains (e.g., grass, pavement, mud).
        friction_range = [0.2, 0.8]  # Slightly wider range to cover a variety of common terrains.
        
        # Restitution affects how bouncy surfaces are, which might not vary significantly for our use-case.
        restitution_range = [0.2, 0.5]  # Limited range to simulate most ground materials realistically.
        
        # Added mass can simulate carrying different loads, but extreme values could hinder basic locomotion learning.
        added_mass_range = [0.0, 3.0]  # Modest range to simulate light to moderate additional weight.
        
        # COM displacement simulates changes in the robot's load distribution or wear and tear.
        com_displacement_range = [0.0, 0.2]  # Small displacements to ensure the model can adapt to slight imbalances.
        
        # Motor strength variation can simulate wear and tear or battery level changes.
        motor_strength_range = [0.7, 1.5]  # Allow for some variation to ensure robustness to motor performance changes.
        
        # Kp (proportional gain) and Kd (derivative gain) factors influence controller responsiveness.
        Kp_factor_range = [1.0, 1.5]  # Keep within a moderate range to ensure controlled motion without instability.
        Kd_factor_range = [0.5, 1.5]  # Moderate variation for programming derivative gain.
        
        # Stiffness, damping, friction, and armature of degrees of freedom are crucial for realistic joint behavior.
        # Given there's no variability in their ranges, no randomization is applied here.
        dof_stiffness_range = [0.1, 0.1]
        dof_damping_range = [0.01, 0.01]
        dof_friction_range = [0.01, 0.01]
        dof_armature_range = [0.01, 0.01]
        
        # Push velocity simulates external forces, e.g., wind or collisions.
        push_vel_xy_range = [0.0, 1.0]  # Small to moderate external disturbances.
        
        # Gravity variations can simulate different inclines or declines in the terrain.
        gravity_range = [-0.3, 0.3]  # Small alterations to simulate running up/downhill or uneven weight distribution.
