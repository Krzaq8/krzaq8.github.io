        # Each parameter is considered for the task of training a quadruped robot to run on a variety of terrains.
        
        # Friction plays a crucial role in the interaction between the robot and different surfaces.
        # We should randomize this parameter to prepare the robot for both slippery and grippy terrains.
        friction_range = [0.2, 0.8]
        
        # Restitution affects how bouncy the interactions are. 
        # Varied levels may be less critical for running but can impact stability on different terrains.
        # Small variation may help in learning robustness without making the problem unnecessarily difficult.
        restitution_range = [0.3, 0.5]
        
        # Added mass simulates carrying varying loads or changes in the robot's weight.
        # Useful for ensuring the robot remains steady under different loading conditions.
        added_mass_range = [0.0, 3.0]
        
        # Center of mass displacement simulates weight distribution changes.
        # This can critically affect balance and should be varied slightly to ensure adaptability.
        com_displacement_range = [0.0, 0.5]
        
        # Motor strength variability prepares the robot for scenarios where it may have to exert different levels of force, 
        # such as running up an incline or dealing with obstacles.
        motor_strength_range = [0.75, 1.25]
        
        # Kp (proportional gain) and Kd (differential gain) factors for the robot's PID controllers can vary 
        # due to changes in terrain or mechanical wear and tear.
        Kp_factor_range = [1.0, 1.5]
        Kd_factor_range = [0.1, 1.5]
        
        # Degrees of freedom (DOF) parameters such as stiffness, damping, friction, and armature
        # may not need randomization for a running task on varied terrain, as these are often constants for a given robot design.
        
        # Push velocity in the xy plane simulates external forces like wind or collisions with small objects.
        # Small randomizations here can help the robot to learn to maintain its path and speed.
        push_vel_xy_range = [0.0, 2.0]
        
        # Gravity variations help adapt the model to small inaccuracies in the simulation's physics engine compared to real-world physics.
        # A small randomization can simulate different scenarios, although significant changes might make the task infeasible to learn.
        gravity_range = [-0.1, 0.1]
