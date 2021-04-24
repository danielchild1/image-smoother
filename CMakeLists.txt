# CMakeList.txt : CMake project for image-smoother, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project ("image-smoother")

# Add source to this project's executable.
add_executable (image-smoother "image-smoother.cpp" "image-smoother.h")

# TODO: Add tests and install targets if needed.
