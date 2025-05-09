# ----------------------------------------------------------------------------
# Set the minimum CMake version required
# ----------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.20.0)

# ----------------------------------------------------------------------------
# Define the project and the languages it will use
# ----------------------------------------------------------------------------
project(YOLO-TensorRT LANGUAGES CXX CUDA)

# ----------------------------------------------------------------------------
# Set the C++ standard to C++17 and make it required
# ----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Release)

# ----------------------------------------------------------------------------
# Create output directories for libraries and binaries
# ----------------------------------------------------------------------------
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

# ----------------------------------------------------------------------------
# Define where the compiled libraries and executables will be stored
# ----------------------------------------------------------------------------
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

# ----------------------------------------------------------------------------
# Find required packages: OpenCV and CUDA Toolkit
# ----------------------------------------------------------------------------
find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)

# ----------------------------------------------------------------------------
# Set TensorRT path based on the environment variable or default to /usr/local/tensorrt
# ----------------------------------------------------------------------------
if(DEFINED ENV{TENSORRT_PATH})
  set(TENSORRT_PATH $ENV{TENSORRT_PATH})
else()
  set(TENSORRT_PATH "/usr/local/tensorrt")
endif()

# ----------------------------------------------------------------------------
# Set CUDA architecture versions
# ----------------------------------------------------------------------------
set(CMAKE_CUDA_ARCHITECTURES 75 86)

# ----------------------------------------------------------------------------
# Gather source files for the deploy library
# ----------------------------------------------------------------------------
file(GLOB_RECURSE DEPLOY_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/deploy/core/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/deploy/utils/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/deploy/infer/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/deploy/infer/*.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/deploy/model.cpp
)

# ----------------------------------------------------------------------------
# Create a shared library for deployment
# ----------------------------------------------------------------------------
add_library(Deploy SHARED ${DEPLOY_SOURCES})

# ----------------------------------------------------------------------------
# Set include directories for the Deploy target
# ----------------------------------------------------------------------------
target_include_directories(Deploy PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${CUDAToolkit_INCLUDE_DIRS}
  ${TENSORRT_PATH}/include
)

# ----------------------------------------------------------------------------
# Set compile options depending on the compiler and configuration
# ----------------------------------------------------------------------------
if(MSVC)
  target_compile_options(Deploy PRIVATE $<$<CONFIG:Release>:/O2>)
else()
  target_compile_options(Deploy PRIVATE -O3)
endif()

# ----------------------------------------------------------------------------
# Link TensorRT libraries and other necessary libraries to the Deploy target
# ----------------------------------------------------------------------------
target_link_directories(Deploy PRIVATE ${TENSORRT_PATH}/lib)
target_link_libraries(Deploy PRIVATE
  ${OpenCV_LIBS}
  CUDA::cudart
  nvinfer
  nvinfer_plugin
  nvonnxparser
)

# ----------------------------------------------------------------------------
# Create the executable for detection
# ----------------------------------------------------------------------------
add_executable(detection ${CMAKE_CURRENT_SOURCE_DIR}/src/YoloDetector.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)

# ----------------------------------------------------------------------------
# Set include directories for the detection executable
# ----------------------------------------------------------------------------
target_include_directories(detection PRIVATE 
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${TENSORRT_PATH}/include
  ${CUDAToolkit_INCLUDE_DIRS}
)

# ----------------------------------------------------------------------------
# Link the Deploy library and OpenCV libraries to the detection executable
# ----------------------------------------------------------------------------
target_link_libraries(detection PRIVATE Deploy ${OpenCV_LIBS})

# ----------------------------------------------------------------------------
# Set runtime path for the detection executable
# ----------------------------------------------------------------------------
set_target_properties(detection PROPERTIES
  INSTALL_RPATH "${TENSORRT_PATH}/lib:${CMAKE_CURRENT_SOURCE_DIR}/lib"
  BUILD_WITH_INSTALL_RPATH TRUE
)

# ----------------------------------------------------------------------------
# Install the Deploy library to the lib directory
# ----------------------------------------------------------------------------
install(TARGETS Deploy
  LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/lib
)

# ----------------------------------------------------------------------------
# Install the detection executable to the bin directory
# ----------------------------------------------------------------------------
install(TARGETS detection
  RUNTIME DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/bin
)
