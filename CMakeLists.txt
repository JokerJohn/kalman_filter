cmake_minimum_required(VERSION 3.0.2)
project(kalman_filter)

add_compile_options(-std=c++11)
find_package(Eigen3 QUIET)
find_package(catkin REQUIRED COMPONENTS
        roscpp
        rospy
        sensor_msgs
        std_msgs
        )
catkin_package()

include_directories(
        include
        ${catkin_INCLUDE_DIRS}
        ${Eigen3_INCLUDE_DIRS}
)

set(SOURCE
        src/fusion_ekf.cpp
        src/kalman_filter.cpp
        src/main_ekf.cpp
        src/tools.cpp)
add_executable(ekf_node ${SOURCE})
target_link_libraries(ekf_node
        ${catkin_LIBRARIES}
        )

add_executable(ukf_node src/fusion_ukf.cpp src/main_ukf.cpp src/tools.cpp)
target_link_libraries(ukf_node
        ${catkin_LIBRARIES}
        )