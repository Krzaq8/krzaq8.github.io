        # Reasoning over parameters:
        
        # 1. friction_range: Varies widely in the real world, from slippery indoor floors to rough outdoor terrain.
        friction_range = [0.2, 0.8] # This range is narrowed down to exclude extremely slippery and extremely sticky conditions, but still accommodates a wide variety of common terrains.
        
        # 2. restitution_range: This measures how bouncy the interaction is between the robot and the terrain. Real-world surfaces vary, but not as widely as friction.
        restitution_range = [0.2, 0.6] # Most real surfaces aren't very bouncy (like concrete, dirt, grass), so a moderate range should cover it.
        
        # 3. added_mass_range: Simulates additional load on the robot, which it might experience due to carrying weights or due to its own weight variations.
        added_mass_range = [0.0, 3.0] # A range that includes no added weight to a moderate amount, preparing the robot for potential payload or weight variance.
        
        # 4. com_displacement_range: Shifts the center of mass, simulating various payloads or uneven weight distribution.
        com_displacement_range = [-0.05, 0.05] # Small displacements to ensure stability isn't overly compromised, aiming to adapt to slight payload variations.
        
        # 5. motor_strength_range: Motor strength can vary due to battery level, wear and tear, or manufacturing variances.
        motor_strength_range = [0.8, 1.5] # Ensures robustness against motor performance variations while avoiding extremes that might not be encountered.
        
        # 6. Kp_factor_range and Kd_factor_range: Control gains that might need adjustment based on robot wear and tear or other factors.
        Kp_factor_range = [1.0, 1.5] # Modest range to ensure stability and responsiveness under different conditions without overfitting to extreme values.
        Kd_factor_range = [0.1, 1.0] # Damping should be adaptable but within a range that ensures control smoothness.
        
        # 7. dof_stiffness_range, dof_damping_range, dof_friction_range, dof_armature_range: These relate to the joints' mechanical properties, which are unlikely to vary in the short term and can be precisely controlled in manufacturing.
        # No need to randomize as these values are constant and precisely engineered.
        
        # 8. push_vel_xy_range: Simulates external forces like wind or interaction with obstacles.
        push_vel_xy_range = [0.0, 2.0] # Mild external forces to ensure robustness without simulating extreme conditions unlikely to be faced regularly.
        
        # 9. gravity_range: While gravity is constant, slight variations in the simulation can help adapt to unexpected situations, like moving on a small hill or variations in terrain elevation.
        gravity_range = [-0.1, 0.1] # Small variations to adapt to minor inclines or declines without simulating unrealistic environments.
