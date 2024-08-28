from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'ufgsim_pkg'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
        (os.path.join('share', package_name, 'models'), glob('models/*')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*')),
    ],
    install_requires=[
        'setuptools',
        'ros2_numpy',
        'transforms3d',
    ],
    zip_safe=True,
    maintainer='air',
    maintainer_email='air@todo.todo',
    description='The UFGSim dataset inference package for ROS 2',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'riunet_inf = ufgsim_pkg.riunet_inf:main'
        ],
    },
)
