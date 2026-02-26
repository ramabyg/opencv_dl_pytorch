cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(Training-Caltech-dataset-in-Libtorch)


find_package(Torch REQUIRED PATHS "../libtorch")
find_package(OpenCV REQUIRED PATHS "/home/chetan/cv2/OpenCV/lib/cmake/opencv4")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
include_directories(${OpenCV_INCLUDE_DIRS})


add_executable(Libtorch-week15-trainCNN Caltech_training.cpp)
target_link_libraries(Libtorch-week15-trainCNN  ${OpenCV_LIBS})
target_link_libraries(Libtorch-week15-trainCNN "${TORCH_LIBRARIES}")
set_property(TARGET Libtorch-week15-trainCNN PROPERTY CXX_STANDARD 14)
