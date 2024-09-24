FROM ros:humble

ENV COLORCLOUD_BRANCH=UFGSim
ENV ROS_DISTRO=humble

RUN apt update && apt install -y \
    git \
    nano \
    python3-pip \
    ros-${ROS_DISTRO}-cv-bridge \
    ros-${ROS_DISTRO}-tf-transformations \
    ros-${ROS_DISTRO}-rviz2 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install transforms3d

WORKDIR /root/

RUN cd /root/ && \
    git clone -b ${COLORCLOUD_BRANCH} https://github.com/AIR-UFG/colorcloud.git && \
    cd colorcloud && \
    pip install -e '.[dev]'

COPY . /root/ros2_ws/src/ufgsim_pkg

RUN cd /root/ros2_ws/src && \
    git clone -b ${ROS_DISTRO} https://github.com/Box-Robotics/ros2_numpy.git && \
    cd .. && \
    /bin/bash -c "source /opt/ros/${ROS_DISTRO}/setup.bash && colcon build --symlink-install"

RUN echo "#!/bin/bash" > /root/entrypoint.sh \
    && echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /root/entrypoint.sh \
    && echo "source /root/ros2_ws/install/setup.bash" >> /root/entrypoint.sh \
    && echo "exec \"\$@\"" >> /root/entrypoint.sh \
    && sudo chmod +x /root/entrypoint.sh \
    && echo "source /root/entrypoint.sh" >> /root/.bashrc

# Set the entrypoint
ENTRYPOINT ["/root/entrypoint.sh"]

# Set the default command
CMD ["/bin/bash"]

