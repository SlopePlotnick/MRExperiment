cmake_minimum_required(VERSION 3.9)
project(FaceRecognition)

set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lstdc++")

#find_library(OpenCV)
find_package(OpenCV)

include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(Eigenfaces Eigenfaces.cpp)
add_executable(Fisherfaces Fisherfaces.cpp)

target_link_libraries(Eigenfaces ${OpenCV_LIBS})
target_link_libraries(Fisherfaces ${OpenCV_LIBS})
