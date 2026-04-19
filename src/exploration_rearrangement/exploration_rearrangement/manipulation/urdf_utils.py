"""
Stretch Robot URDF utilities.

Loads the Stretch URDF, strips non-arm links/joints, injects virtual
mobile-base joints, and writes a modified URDF to the package assets dir
at import time.
"""

import shutil
from pathlib import Path

import numpy as np
import urchin as urdfpy

MOBILE_BASE_EFFORT_LIMIT = 100.0
MOBILE_BASE_VELOCITY_LIMIT = 1.0
MOBILE_BASE_TRANSLATION_LIMIT = 1.0

_PKG_ROOT = Path(__file__).resolve().parents[2]
_ASSETS_DIR = _PKG_ROOT / 'assets'
STRETCH_URDF_PATH = _ASSETS_DIR / 'stretch.urdf'
MODIFIED_URDF_PATH = _ASSETS_DIR / 'stretch_ik.urdf'

LINKS_TO_REMOVE = [
    'link_right_wheel', 'link_left_wheel', 'caster_link', 'link_head', 'link_head_pan',
    'link_head_tilt', 'link_aruco_right_base', 'link_aruco_left_base', 'link_aruco_shoulder',
    'link_aruco_top_wrist', 'link_aruco_inner_wrist', 'camera_bottom_screw_frame', 'camera_link',
    'camera_depth_frame', 'camera_depth_optical_frame', 'camera_infra1_frame', 'camera_infra1_optical_frame',
    'camera_infra2_frame', 'camera_infra2_optical_frame', 'camera_color_frame', 'camera_color_optical_frame',
    'camera_accel_frame', 'camera_accel_optical_frame', 'camera_gyro_frame', 'camera_gyro_optical_frame',
    'gripper_camera_bottom_screw_frame', 'gripper_camera_link', 'gripper_camera_depth_frame',
    'gripper_camera_depth_optical_frame', 'gripper_camera_infra1_frame', 'gripper_camera_infra1_optical_frame',
    'gripper_camera_infra2_frame', 'gripper_camera_infra2_optical_frame', 'gripper_camera_color_frame',
    'gripper_camera_color_optical_frame', 'laser', 'base_imu', 'respeaker_base', 'link_wrist_quick_connect',
    'link_gripper_finger_right', 'link_gripper_fingertip_right', 'link_aruco_fingertip_right',
    'link_gripper_finger_left', 'link_gripper_fingertip_left', 'link_aruco_fingertip_left',
    'link_aruco_d405', 'link_head_nav_cam',
]

JOINTS_TO_REMOVE = [
    'joint_right_wheel', 'joint_left_wheel', 'caster_joint', 'joint_head', 'joint_head_pan',
    'joint_head_tilt', 'joint_aruco_right_base', 'joint_aruco_left_base', 'joint_aruco_shoulder',
    'joint_aruco_top_wrist', 'joint_aruco_inner_wrist', 'camera_joint', 'camera_link_joint',
    'camera_depth_joint', 'camera_depth_optical_joint', 'camera_infra1_joint', 'camera_infra1_optical_joint',
    'camera_infra2_joint', 'camera_infra2_optical_joint', 'camera_color_joint', 'camera_color_optical_joint',
    'camera_accel_joint', 'camera_accel_optical_joint', 'camera_gyro_joint', 'camera_gyro_optical_joint',
    'gripper_camera_joint', 'gripper_camera_link_joint', 'gripper_camera_depth_joint',
    'gripper_camera_depth_optical_joint', 'gripper_camera_infra1_joint', 'gripper_camera_infra1_optical_joint',
    'gripper_camera_infra2_joint', 'gripper_camera_infra2_optical_joint', 'gripper_camera_color_joint',
    'gripper_camera_color_optical_joint', 'joint_laser', 'joint_base_imu', 'joint_respeaker',
    'joint_wrist_quick_connect', 'joint_gripper_finger_right', 'joint_gripper_fingertip_right',
    'joint_aruco_fingertip_right', 'joint_gripper_finger_left', 'joint_gripper_fingertip_left',
    'joint_aruco_fingertip_left', 'joint_aruco_d405', 'joint_head_nav_cam',
]

def make_virtual_base_joints():
    """Return (joints, links) that give the IK solver a virtual mobile base."""
    joint_rot = urdfpy.Joint(
        name='joint_base_rotation',
        parent='base_link',
        child='link_base_rotation',
        joint_type='revolute',
        axis=np.array([0.0, 0.0, 1.0]),
        origin=np.eye(4, dtype=np.float64),
        limit=urdfpy.JointLimit(
            effort=MOBILE_BASE_EFFORT_LIMIT,
            velocity=MOBILE_BASE_VELOCITY_LIMIT,
            lower=-np.pi, upper=np.pi,
        ),
    )
    link_rot = urdfpy.Link(name='link_base_rotation',
                           inertial=None, visuals=None, collisions=None)

    joint_trans = urdfpy.Joint(
        name='joint_base_translation',
        parent='link_base_rotation',
        child='link_base_translation',
        joint_type='prismatic',
        axis=np.array([1.0, 0.0, 0.0]),
        origin=np.eye(4, dtype=np.float64),
        limit=urdfpy.JointLimit(
            effort=MOBILE_BASE_EFFORT_LIMIT,
            velocity=MOBILE_BASE_VELOCITY_LIMIT,
            lower=-MOBILE_BASE_TRANSLATION_LIMIT,
            upper=MOBILE_BASE_TRANSLATION_LIMIT,
        ),
    )
    link_trans = urdfpy.Link(name='link_base_translation',
                             inertial=None, visuals=None, collisions=None)

    return [joint_rot, joint_trans], [link_rot, link_trans]


def build_modified_urdf():
    """Build the IK-ready URDF and save it to *MODIFIED_URDF_PATH*."""
    urdf = urdfpy.URDF.load(str(STRETCH_URDF_PATH))
    modified = urdf.copy()

    links_set = set(LINKS_TO_REMOVE)
    joints_set = set(JOINTS_TO_REMOVE)
    modified._links = [l for l in modified._links if l.name not in links_set]
    modified._joints = [j for j in modified._joints if j.name not in joints_set]

    virt_joints, virt_links = make_virtual_base_joints()
    modified._joints.extend(virt_joints)
    modified._links.extend(virt_links)

    for joint in modified._joints:
        if joint.name == 'joint_mast':
            joint.parent = 'link_base_translation'

    modified.save(str(MODIFIED_URDF_PATH))
    fix_urchin_mesh_paths()


def fix_urchin_mesh_paths():
    """Fix mangled mesh paths (./meshes/./meshes/...) produced by urchin's save()
    and remove the duplicated mesh directory it creates."""
    text = MODIFIED_URDF_PATH.read_text()
    text = text.replace('./meshes/./meshes/', './meshes/')
    MODIFIED_URDF_PATH.write_text(text)
    nested_meshes = _ASSETS_DIR / 'meshes' / 'meshes'
    if nested_meshes.is_dir():
        shutil.rmtree(nested_meshes)


if __name__ == "__main__":
    build_modified_urdf()
