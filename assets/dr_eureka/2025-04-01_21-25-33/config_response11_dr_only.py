        # Friction can vary significantly across different surfaces such as asphalt, grass, tiles, etc.
        # Therefore, introducing variability in friction is critical for real-world deployment.
        friction_range = [0.1, 0.8]  # A reasonable span covering surfaces from slippery to rough.
        
        # Restitution, or how bouncy surfaces are, is generally less variable for the terrains the robot will run on,
        # but should still be considered to a small extent to cover cases like indoor rubber mats or outdoor soft terrains.
        restitution_range = [0.2, 0.6]  # Moderately varied to represent different bounciness levels.
        
        # Added mass can simulate carrying additional weight or dealing with external forces (e.g., wind).
        # Introducing a moderate range can make the robot more robust to changes in loaded weight.
        added_mass_range = [0.0, 3.0]  # Covers scenarios with no extra load to carrying a moderate extra weight.
        
        # The center of mass (com) displacement might vary if the robot carries different payloads.
        # However, substantial variability might make learning excessively hard. Hence, a minor range is chosen.
        com_displacement_range = [0.0, 0.05]  # Small displacement to simulate slight shifts in the center of mass.
        
        # Motor strength variations simulate different wear and tear levels or battery charge states.
        motor_strength_range = [0.7, 1.5]  # Covers a range from slightly worn out or low battery to fully operational.
        
        # Kp_factor (proportional gain) and Kd_factor (derivative gain) affect the control response. 
        # Randomizing these within a moderate range simulates variability in robot response due to tuning or hardware differences.
        Kp_factor_range = [0.95, 1.5]
        Kd_factor_range = [0.1, 1.5]
        
        # DOF stiffness, damping, and friction primarily relate to mechanical properties that are not likely to vary in real-world conditions.
        # Therefore, we will keep these parameters fixed as randomizing them would not contribute significantly to real-world adaptation.
        dof_stiffness_range = [0.1, 0.1]  # Fixed, as it does not vary significantly in the real world.
        dof_damping_range = [0.01, 0.01]  # Fixed, similar reason.
        dof_friction_range = [0.01, 0.01]  # Fixed, similar reason.
        dof_armature_range = [0.01, 0.01]  # Fixed, similar reason.
        
        # Push velocity simulates external forces like wind or interactions with moving objects.
        # Introducing a small variability can help the robot learn to stabilize against such forces.
        push_vel_xy_range = [0.0, 2.0]  # Moderate external forces representation from different directions.
        
        # Gravity variations might simulate slight changes in terrain elevation or running on very different terrains, like very soft ground.
        # However, excessive variability in gravity does not reflect real-world conditions and could hinder learning.
        gravity_range = [0.0, 0.0]  # Keeping gravity constant as variability here doesn't match real-world conditions accurately.
