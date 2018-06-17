cmake_minimum_required(VERSION 3.5.1)

include_directories(include)
add_definitions("-std=c++1z -O3")
add_executable(kaspar src/main.cpp)
