# UFGSim Package

## Overview

The UFGSim package provides a ROS 2 node for running semantic segmentation inference using the RIUNet model.

## Features

- **Semantic Segmentation**: The node uses the RIUNet model for semantic segmentation on point cloud data.
- **Customizable Parameters**: You can adjust various parameters such as the field of view, image dimensions, and paths to the model and configuration files.
- **ROS 2 Integration**: The node is fully integrated with ROS 2, making it easy to incorporate into larger robotics systems.

## Requirements

Ensure you have the following dependencies installed:

- **ROS 2 (Humble or later)**
- **Python 3.10 or later**
- **PyTorch**
- **transforms3d**: Install via pip:
  ```bash
  pip install transforms3d
  ```
- **tf_transformations**: This is a ROS package. You can install it via apt:
  ```bash
  sudo apt-get install ros-${ROS_DISTRO}-tf-transformations
  ```
- **[ros2_numpy](https://github.com/Box-Robotics/ros2_numpy)**: Clone and build it in your ROS 2 workspace, e.g.:
  ```bash
  cd ~/ros2_ws/src
  git clone https://github.com/Box-Robotics/ros2_numpy
  cd ~/ros2_ws
  colcon build
  ```
- **[colorcloud](https://github.com/AIR-UFG/colorcloud.git)**: The main dependency for this package. Make sure to install it following the instructions in the repository.
    ```bash
    git clone -b UFGSim https://github.com/AIR-UFG/colorcloud.git
    cd colorcloud
    pip install -e '.[dev]'
    ```
## Installation

Clone the repository into your ROS 2 workspace and build it:

```bash
cd ~/ros2_ws/src
git clone https://github.com/AIR-UFG/ufgsim_pkg.git
cd ~/ros2_ws
colcon build
```

Make sure to source your workspace:

```bash
source ~/ros2_ws/install/setup.bash
```

## Running the Node

You can run the node directly using `ros2 run` or via a launch file.

### Using `ros2 run`

To run the node directly:

```bash
ros2 run ufgsim_pkg riunet_inf --ros-args -p model_path:=/path/to/your/model.ckpt
```

### Parameters

The following parameters can be set when running the node:

- **fov_up**: The field of view upwards in degrees. Default is `15.0`.
- **fov_down**: The field of view downwards in degrees. Default is `-15.0`.
- **width**: The width of the projection image. Default is `440`.
- **height**: The height of the projection image. Default is `16`.
- **yaml_path**: Path to the YAML configuration file. Default is set based on the package configuration directory. E.g. `~/ros2_ws/src/ufgsim_pkg/config/ufg-sim.yaml`.
- **model_path**: Path to the RIUNet model checkpoint file.

### Using `ros2 launch`

You can also use a launch file to set parameters and run the node. This is particularly useful when you want to reuse the configuration or manage multiple nodes.

#### Running the Launch File

To run the node using the launch file, execute the following command:

```bash
ros2 launch ufgsim_pkg riunet_inf.launch.py model_path:=/path/to/your/model.ckpt
```

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

