from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'model_path',
            default_value='model_path',
            description='Path to the model checkpoint file'
        ),
        
        Node(
            package='ufgsim_pkg',
            executable='riunet_inf',
            name='riunet_inf',
            output='screen',
            parameters=[{
                'fov_up': 15.0,
                'fov_down': -15.0,
                'width': 440,
                'height': 16,
                'yaml_path': os.path.join(get_package_share_directory('ufgsim_pkg'),'config','ufg-sim.yaml'),
                'model_path': LaunchConfiguration('model_path'),
            }]
        )
    ])
