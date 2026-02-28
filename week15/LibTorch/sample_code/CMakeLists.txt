cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(sample-code)

find_package(Torch REQUIRED PATHS "../../../../libtorch-win-shared-with-deps-2.10.0+cpu/libtorch" )

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Zi /Od")

add_executable(sample sample.cpp)
target_link_libraries(sample "${TORCH_LIBRARIES}")
# ensure target uses C++17 as well
set_property(TARGET sample PROPERTY CXX_STANDARD 17)
