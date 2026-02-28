cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(from-pytorch-to-libtorch)


find_package(Torch REQUIRED PATHS "../../../../libtorch-win-shared-with-deps-2.10.0+cpu/libtorch" )

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Zi /Od")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")



add_executable(run-traced-model load_traced_LeNet.cpp)
target_link_libraries(run-traced-model "${TORCH_LIBRARIES}")
set_property(TARGET run-traced-model PROPERTY CXX_STANDARD 17)

add_executable(test-combined test_combined.cpp)
target_link_libraries(test-combined "${TORCH_LIBRARIES}")
set_property(TARGET test-combined PROPERTY CXX_STANDARD 17)

