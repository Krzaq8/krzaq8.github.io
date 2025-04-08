        # Given the task of training a quadruped robot to run on various terrains, it's critical to choose domain randomization parameters that closely mimic real-world conditions. Here's the reasoning behind each parameter:
        
        # Friction is crucial as it varies significantly across different terrains (e.g., asphalt, grass, dirt). However, too wide a range can hinder learning.
        friction_range = [0.2, 0.8]  # Slightly narrowed to focus on a realistic spectrum of common terrains.
        
        # Restitution (elasticity of collisions) affects how the robot interacts with the terrain during running, particularly over uneven surfaces.
        restitution_range = [0.2, 0.6]  # Adjusted to represent a range of surface hardness.
        
        # Added mass can simulate carrying varying loads or the impact of the robot's body design changes.
        added_mass_range = [0.0, 3.0]  # Limited range to realistic additional payloads.
        
        # The center of mass displacement impacts stability and maneuverability. 
        com_displacement_range = [0.0, 0.5]  # Narrow range focusing on maintaining stability while allowing slight variations.
        
        # Motor strength directly affects the robot's ability to move and adapt to different terrains.
        motor_strength_range = [0.8, 1.5]  # Adjusted to simulate variance in motor performance under different loads and conditions.
        
        # Kp (proportional gain) and Kd (derivative gain) factors are crucial for controlling movement dynamics.
        Kp_factor_range = [1.0, 1.5]  # Refined to ensure controlled movement while allowing for adaptation.
        Kd_factor_range = [0.1, 1.5]  # Wide range to explore diverse damping behaviors.
        
        # Since dof_stiffness, dof_damping, dof_friction, and dof_armature are provided with singular values, suggesting they may not need randomization or could represent a fixed aspect of the robot's physical structure, we'll leave them as is.
        # However, for completeness in simulation, minor variability could be introduced if deemed necessary, but will keep as provided for the purpose of this task.
        
        # Unexpected pushes can simulate random external forces the robot might encounter.
        push_vel_xy_range = [0.0, 2.0]  # Limited to simulate realistic disturbances without overwhelming the robot.
        
        # Modifying gravity can help the robot adapt to minor differences in environmental conditions, though real-world deviations are small.
        gravity_range = [-0.2, 0.2]  # Slightly varied to help the model generalize better without unrealistic conditions.
        
        # Outputs
        print(f"friction_range = {friction_range}")
        print(f"restitution_range = {restitution_range}")
        print(f"added_mass_range = {added_mass_range}")
        print(f"com_displacement_range = {com_displacement_range}")
        print(f"motor_strength_range = {motor_strength_range}")
        print(f"Kp_factor_range = {Kp_factor_range}")
        print(f"Kd_factor_range = {Kd_factor_range}")
        # Note: dof_stiffness, dof_damping, dof_friction, and dof_armature will be kept as is, based on the provided valid ranges or the lack of specific requirements for randomization.
        print(f"push_vel_xy_range = {push_vel_xy_range}")
        print(f"gravity_range = {gravity_range}")
