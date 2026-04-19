from setuptools import find_packages, setup
from glob import glob

package_name = 'exploration_rearrangement'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/rviz', glob('rviz/*.rviz')),
    ],
    install_requires=[
        'setuptools',
        'numpy',
        'opencv-python',
        'pyyaml',
        'openai>=1.30.0',
        'ultralytics>=8.3.0',
    ],
    zip_safe=True,
    maintainer='Haokun Zhu',
    maintainer_email='haokunz@andrew.cmu.edu',
    description='Stretch 3 autonomous exploration + object rearrangement (greedy + VLM planner).',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'exploration_node = exploration_rearrangement.exploration_node:main',
            'object_detector_node = exploration_rearrangement.object_detector_node:main',
            'fine_object_detector_node = exploration_rearrangement.fine_object_detector_node:main',
            'region_manager_node = exploration_rearrangement.region_manager_node:main',
            'task_planner_node = exploration_rearrangement.task_planner_node:main',
            'task_executor_node = exploration_rearrangement.task_executor_node:main',
            'manipulation_node = exploration_rearrangement.manipulation_node:main',
            'head_scan_node = exploration_rearrangement.head_scan_node:main',
            'fake_sim_node = exploration_rearrangement.sim.fake_sim_node:main',
            'fake_planner_inputs = exploration_rearrangement.sim.fake_planner_inputs:main',
            'set_up_yolo_e = exploration_rearrangement.set_up_yolo_e:main',
            'visual_grasp_node = exploration_rearrangement.visual_grasp_node:main',
            'visual_servo_arm_node = exploration_rearrangement.visual_servo_arm_node:main',
        ],
    },
)
