import numpy as np
from math import cos, sin, atan2, sqrt
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink

# Hardcoded parameters extracted from the URDF file
JOINT_NAMES = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
               'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
JOINT_ORIGINS_XYZ = ['0 0 0.1626713656238066',
                     '0.0001074284466366568 0 0',
                     '-0.4252792228483518 0 0',
                     '-0.3925816029631302 -0.0006304099090212775 0.1336958186051931',
                     '-4.514033180005977e-06 -0.09974721170876895 0.0001769098165466073',
                     '1.040383298500094e-05 0.09968862557388704 9.21302418261068e-05']
JOINT_ORIGINS_RPY = ['0 0 4.440162011676296e-08',
                     '1.568900299583206 0 5.317153005565982e-06',
                     '3.140215546292493 3.138992367298761 3.141590053180454',
                     '0.004715220293048735 -0.001298588720806687 1.157534920017808e-05',
                     '1.569022747082532 0 1.466406059833641e-06',
                     '1.571720506610457 3.141592653589793 -3.141592419450557']
JOINT_AXES_XYZ = ['0 0 1', '0 0 1', '0 0 1', '0 0 1', '0 0 1', '0 0 1']


def matrix_exponential(axis_angle):
    """
    Compute the matrix exponential for a 3x3 matrix representing an axis-angle rotation.
    This is a simplified version specific to our 3x3 rotation matrices.
    """
    theta = np.linalg.norm(axis_angle)
    if theta < 1e-10:
        return np.eye(3)

    axis = axis_angle / theta
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])

    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)
    return R


def compute_transformation(xyz, rpy):
    """
    Generate a 4x4 transformation matrix based on position (x, y, z) and orientation (roll, pitch, yaw).
    """
    x, y, z = xyz
    roll, pitch, yaw = rpy

    # Compute rotation matrix from rpy
    R_x = np.array([
        [1, 0, 0],
        [0, cos(roll), -sin(roll)],
        [0, sin(roll), cos(roll)]
    ])

    R_y = np.array([
        [cos(pitch), 0, sin(pitch)],
        [0, 1, 0],
        [-sin(pitch), 0, cos(pitch)]
    ])

    R_z = np.array([
        [cos(yaw), -sin(yaw), 0],
        [sin(yaw), cos(yaw), 0],
        [0, 0, 1]
    ])

    R = np.dot(R_z, np.dot(R_y, R_x))

    # Constructing the transformation matrix
    T = np.identity(4)
    T[0:3, 0:3] = R
    T[0, 3] = x
    T[1, 3] = y
    T[2, 3] = z

    return T


def forward_kinematics(joint_angles):
    """
    Forward kinematics function using hardcoded parameters.
    """
    T_matrices = []

    # Compute transformation matrices for each joint using hardcoded values
    for idx, joint_name in enumerate(JOINT_NAMES):
        xyz = [float(val) for val in JOINT_ORIGINS_XYZ[idx].split()]
        rpy = [float(val) for val in JOINT_ORIGINS_RPY[idx].split()]
        T_static = compute_transformation(xyz, rpy)

        # If the joint is revolute, compute the transformation due to joint rotation
        axis = np.array([float(val) for val in JOINT_AXES_XYZ[idx].split()])
        if np.linalg.norm(axis) > 1e-10:
            R_rot = matrix_exponential(axis * joint_angles[idx])
            T_rot = np.identity(4)
            T_rot[0:3, 0:3] = R_rot
            T = np.dot(T_static, T_rot)
        else:
            T = T_static

        T_matrices.append(T)

    # Compute the end-effector pose by multiplying the transformation matrices in sequence
    T_ee = np.identity(4)
    for T in T_matrices:
        T_ee = np.dot(T_ee, T)

    x = T_ee[0, 3]
    y = T_ee[1, 3]
    z = T_ee[2, 3]

    r11 = T_ee[0, 0]
    r21 = T_ee[1, 0]
    r31 = T_ee[2, 0]
    r32 = T_ee[2, 1]
    r33 = T_ee[2, 2]

    pitch = atan2(-r31, sqrt(r11 ** 2 + r21 ** 2))
    roll = atan2(r21 / cos(pitch), r11 / cos(pitch))
    yaw = atan2(r32 / cos(pitch), r33 / cos(pitch))

    # output = np.array([x, y, z, yaw(Z), pitch(Y), roll(X)])
    return x, y, z, yaw, pitch, roll


def inverse_kinematics(target_position, target_rpy):
    """
    Corrected inverse kinematics for UR5 using IKPy with URDFLink constructed from transformation matrix.
    """

    # Initialize a list with the OriginLink
    links = [OriginLink()]

    # Define which links are active. Assuming a 6-DOF robot like the UR5,
    # the base link is inactive, and the 6 robot joints are active.
    active_links_mask = [False, True, True, True, True, True, True]

    for idx, joint_name in enumerate(JOINT_NAMES):
        xyz = [float(val) for val in JOINT_ORIGINS_XYZ[idx].split()]
        rpy = [float(val) for val in JOINT_ORIGINS_RPY[idx].split()]
        rotation = [float(val) for val in JOINT_AXES_XYZ[idx].split()]

        # Construct the link from the transformation matrix
        link = URDFLink(
            name=joint_name,
            origin_translation=np.array(xyz),
            origin_orientation=np.array([0, 0, 0]),
            rotation=np.array(rotation),  # This should be the rotation axis, not RPY
            bounds=(-3.14, 3.14),
        )
        links.append(link)

    # Construct the UR5 chain with the constructed links
    ur5e_chain = Chain(links=links, active_links_mask=active_links_mask, name='ur5e')

    # Compute the IK
    joint_angles = ur5e_chain.inverse_kinematics(target_position=target_position, target_orientation=target_rpy, orientation_mode='all')

    return joint_angles


if __name__ == '__main__':
    xyz_yawpitchroll = forward_kinematics(np.array([-0.28394961, -2.12822716, -0.59661657, -1.06940711,  0.25196928, -0.11292072]))
    print(xyz_yawpitchroll)
    joint_angles = inverse_kinematics(xyz_yawpitchroll[:3], xyz_yawpitchroll[3:])
    print(joint_angles)