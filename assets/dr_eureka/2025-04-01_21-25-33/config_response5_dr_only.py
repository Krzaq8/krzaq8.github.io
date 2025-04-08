        # Friction is a crucial parameter for running on different types of terrains. 
        # A broader range will help the model learn to adapt to different surfaces, from slippery to rough.
        friction_range = [0.2, 0.8]
        
        # Restitution affects how bouncy the surfaces are. Since most real-world surfaces the robot will 
        # encounter are not very bouncy (like concrete, grass, indoor floors), we keep this range limited.
        restitution_range = [0.2, 0.6]
        
        # Added mass can simulate carrying different payloads or changes in the robot's weight due to modifications or damage.
        # A small range here forces the robot to adapt to slight weight variations without drastically affecting the learning process.
        added_mass_range = [0.0, 2.0]
        
        # Center of mass displacement can simulate inaccuracies in the robot's physical construction and wear over time. 
        # However, too much variation can make the task unnecessarily difficult, so we limit the range.
        com_displacement_range = [0.0, 0.5]
        
        # Motor strength variability can simulate wear and tear or battery level effects on the motors' performance.
        motor_strength_range = [0.8, 1.5]
        
        # Kp_factor (Proportional gain) and Kd_factor (Derivative gain) control the robot's motor controllers.
        # Adjusting these can help the model learn to deal with variations in control response due to hardware inconsistencies.
        Kp_factor_range = [1.0, 1.5]
        Kd_factor_range = [0.2, 1.2]
        
        # Since dof_stiffness, dof_damping, dof_friction, and dof_armature are given with no variability (all have a range from a single value to the same value), 
        # we assume these are physical characteristics of the robot that are consistent and thus do not randomize them.
        
        # Push velocity in the XY plane can simulate unexpected lateral forces like wind or collisions with objects. 
        # A moderate range here allows the robot to learn stability against such forces.
        push_vel_xy_range = [0.0, 3.0]
        
        # Gravity adjustments can simulate slight variations in terrain incline or running on slightly different elevations;
        # however, drastic changes in gravity wouldn't be realistic for the task, so we keep this subtle.
        gravity_range = [-0.2, 0.2]
