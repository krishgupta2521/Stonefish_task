cmake_minimum_required(VERSION 3.10)

project(Task2Baseline)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# =========================================================
# STONEFISH
# =========================================================

set(STONEFISH_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../../beg_task/stonefish_learning/stonefish")

set(STONEFISH_BUILD_DIR
    "${STONEFISH_DIR}/build")

set(DATA_DIR_PATH
    "${STONEFISH_DIR}/Tests/Data/")


# =========================================================
# DEPENDENCIES
# =========================================================

# Keep linking the legacy libGL (the existing behaviour) and silence the
# "OpenGL_GL_PREFERENCE has not been set" developer warning.
set(OpenGL_GL_PREFERENCE LEGACY)

find_package(OpenGL REQUIRED)

find_package(SDL2 REQUIRED)

find_package(Freetype REQUIRED)


# ROS 2
# ROS 2 must be sourced first: it sets the paths CMake, its Python helpers and
# the finished program all need. Fail with the fix instead of "Could not find
# rclcpp".

if(NOT DEFINED ENV{AMENT_PREFIX_PATH})
    message(FATAL_ERROR
        "ROS 2 is not sourced in this terminal. Run:\n"
        "  source /opt/ros/jazzy/setup.bash\n"
        "then run cmake again. To do this in every new terminal:\n"
        "  echo 'source /opt/ros/jazzy/setup.bash' >> ~/.bashrc")
endif()

find_package(rclcpp REQUIRED)

find_package(nav_msgs REQUIRED)

find_package(sensor_msgs REQUIRED)


# =========================================================
# SOURCE FILES
# =========================================================

add_executable(
    Task2Baseline

    src/main.cpp
    src/Task2Manager.cpp
)


# =========================================================
# INCLUDE DIRECTORIES
# =========================================================

target_include_directories(
    Task2Baseline
    PRIVATE

    "${STONEFISH_DIR}/build"

    "${STONEFISH_DIR}/Library/include"

    "${STONEFISH_DIR}/3rdparty"

    ${OPENGL_INCLUDE_DIRS}

    ${SDL2_INCLUDE_DIRS}

    ${FREETYPE_INCLUDE_DIRS}

    ${rclcpp_INCLUDE_DIRS}

    ${nav_msgs_INCLUDE_DIRS}

    ${sensor_msgs_INCLUDE_DIRS}
)


# =========================================================
# COMPILE DEFINITIONS
# =========================================================

target_compile_definitions(
    Task2Baseline
    PRIVATE

    BT_EULER_DEFAULT_ZYX

    BT_USE_DOUBLE_PRECISION

    DATA_DIR_PATH="${DATA_DIR_PATH}"

    SHADER_DIR_PATH="${STONEFISH_DIR}/Library/shaders/"
)


# =========================================================
# LINK LIBRARIES
# =========================================================

target_link_libraries(
    Task2Baseline
    PRIVATE

    "${STONEFISH_BUILD_DIR}/libStonefish_test.so"

    ${OPENGL_LIBRARIES}

    ${SDL2_LIBRARIES}

    ${FREETYPE_LIBRARIES}

    rclcpp::rclcpp

    nav_msgs::nav_msgs__rosidl_typesupport_cpp

    sensor_msgs::sensor_msgs__rosidl_typesupport_cpp
)


# =========================================================
# RUNTIME LIBRARY PATH
# =========================================================

set_target_properties(
    Task2Baseline
    PROPERTIES

    BUILD_RPATH
    "${STONEFISH_BUILD_DIR}"
)
