from roboticstoolbox import DHRobot, RevoluteDH, models

# Global DH parameters based on our derived values
dh_params = [
    {'a': 0.0, 'd': 0.1626713656238066, 'alpha': 0.0, 'theta': 4.44e-08},
    {'a': 0.0001074284466366568, 'd': 0.0, 'alpha': 1.568900299583206, 'theta': 5.32e-06},
    {'a': -0.4252792228483518, 'd': 0.0, 'alpha': 3.140215546292493, 'theta': 3.141590053180454},
    {'a': -0.3925816029631302, 'd': 0.1336958186051931, 'alpha': 0.004715220293048735, 'theta': 1.16e-05},
    {'a': -4.514033180005977e-06, 'd': 0.0001769098165466073, 'alpha': 1.569022747082532, 'theta': 1.47e-06},
    {'a': 1.040383298500094e-05, 'd': 9.21302418261068e-05, 'alpha': 1.571720506610457, 'theta': -3.141592419450557}
]




# def inverse_kinematics(target_position, target_rpy):
#     # Convert target_position and target_rpy to a 4x4 transformation matrix
#     R = rpy_to_rotation_matrix(target_rpy)  # Assuming you have this function from before
#     T_target = SE3(R, target_position)
#
#     q, *_ = ur5e_robot.ikine(T_target)  # Compute joint angles
#
#     return q

if __name__ == "__main__":
    # Define UR5e using the derived DH parameters
    links = [
        RevoluteDH(a=params['a'], d=params['d'], alpha=params['alpha'], offset=params['theta'])
        for params in dh_params
    ]

    ur5e_robot = DHRobot(links, name="UR5e")
    q = [-0.28394961, -2.12822716, -0.59661657, -1.06940711, 0.25196928, -0.11292072]

    # Compute forward kinematics
    T = ur5e_robot.fkine(q)
    print(T)

    robot = models.UR5()
    Te = robot.fkine(q)
    print(Te)
