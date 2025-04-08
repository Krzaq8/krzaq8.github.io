        # Friction is crucial for diverse terrains. We want the robot to learn on a range that covers slippery surfaces (e.g., ice, wet floors) to rough, high-friction surfaces (e.g., gravel, grass). However, extreme values might hinder learning without contributing realistic benefits.
        friction_range = [0.2, 0.9]
        
        # Restitution (elasticity) has a minor role in running scenarios since it's more about bouncing behavior. Given terrains are not typically very elastic, we opt for a lower variability that reflects hard surfaces.
        restitution_range = [0.1, 0.5]
        
        # Added mass simulates the carrying of loads or facing wind resistance. It's realistic for a quadruped to adapt to minor load variations, but extreme variations could derail learning for the primary task of steady running.
        added_mass_range = [0.0, 3.0]
        
        # Center of mass displacement has profound effects on balance. While extreme changes are unrealistic, we should prepare the robot for slight imbalances due to uneven terrain or carrying goods.
        com_displacement_range = [-0.05, 0.05]
        
        # Motor strength variability helps simulate different wear and tear levels of motors, as well as battery level fluctuations. We lean into a moderate range to ensure the robot can adapt without overcomplicating its learning environment.
        motor_strength_range = [0.8, 1.5]
        
        # The proportional (Kp) and derivative (Kd) gains are critical for control stability but should not diverge much from tuned values. Minor variations would teach adaptation to slight changes in those parameters.
        Kp_factor_range = [0.95, 1.2]
        Kd_factor_range = [0.1, 1.5]
        
        # The following parameters (dof_stiffness, dof_damping, dof_friction, dof_armature) do not have variability in the provided ranges, so domain randomization is not applicable here.
        
        # Push velocity in the xy direction simulates external forces such as wind or interaction with moving objects. A moderate range prepares the robot for unexpected forces without overwhelming its ability to move forward.
        push_vel_xy_range = [0.0, 2.0]
        
        # Gravity variations help simulate slopes and minor planetary differences (highly specific use case). A slight variation prepares the robot for inclines or declines without significantly altering the fundamental dynamics it must learn.
        gravity_range = [-0.2, 0.2]
