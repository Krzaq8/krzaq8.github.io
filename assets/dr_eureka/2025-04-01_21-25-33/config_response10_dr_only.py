        # Friction is crucial as it directly affects the robot's ability to move and turn on different terrains.
        friction_range = [0.2, 0.8]  # Slightly broadened to account for the variety from slippery indoor floors to rough outdoor surfaces.
        
        # Restitution affects how much energy is returned upon impact (bouncing behavior), which could vary significantly.
        restitution_range = [0.2, 0.7]  # Tweaked to simulate from very damp surfaces to slightly bouncy ones, but extremes are rare in real terrains.
        
        # Added mass represents external forces acting on the robot, such as carrying a load or wind resistance.
        added_mass_range = [0.0, 3.0]  # Accounting for possible small loads or wind, but not overloading the robot unrealistically.
        
        # Center of mass displacement impacts stability and maneuverability.
        com_displacement_range = [0.0, 0.05]  # Small randomizations to simulate minor shifts, e.g., due to wear and tear or uneven weight distribution.
        
        # Motor strength variability is realistic, considering wear and tear or battery charge levels.
        motor_strength_range = [0.8, 1.5]  # Allows for considerable variation in motor output to mimic real-world scenarios.
        
        # Kp (Proportional gain) and Kd (Derivative gain) factors control the robot's response to position and speed errors, respectively.
        Kp_factor_range = [1.0, 1.8]  # Varied within a realistic range considering different controller tunings.
        Kd_factor_range = [0.2, 1.5]  # Reflects a range where the robot can still perform well but experiences different damping effects.
        
        # Degrees of freedom (dof) parameters should remain constant as they are more related to the mechanical construction.
        dof_stiffness_range = [0.1, 0.1]  # Kept constant as changes here would mean altering the physical properties of joints.
        dof_damping_range = [0.01, 0.01]  # Likewise, constant to avoid unrealistic behavior changes.
        dof_friction_range = [0.01, 0.01]  # Constant, simulating internal joint resistance which shouldn't vary without physical modifications.
        dof_armature_range = [0.01, 0.01]  # Unchanged for consistency in the model's internal joint dynamics.
        
        # Push velocity in the xy direction simulates unexpected lateral forces, such as side winds or collisions.
        push_vel_xy_range = [0.0, 3.0]  # Allows for unexpected interactions but not excessively disruptive to learning.
        
        # Gravity effects are extremely relevant for a quadruped. Small variations can simulate different loading conditions or minor inclines.
        gravity_range = [-0.2, 0.2]  # Slight adjustments to simulate small elevation changes or carrying different loads.
