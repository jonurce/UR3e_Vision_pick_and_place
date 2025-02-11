cmake_minimum_required(VERSION 3.15)
project(UR3e_Vision_pick_and_place)

set(CMAKE_CXX_STANDARD 17)
set(OpenCV_DIR "C:/dev/opencv/build")

find_package(OpenCV REQUIRED)

add_subdirectory(aruco)

file(COPY "data" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")

add_executable(face face.cpp)
target_link_libraries(face PRIVATE ${OpenCV_LIBS})

add_executable(aruco_cam aruco.cpp)
target_link_libraries(aruco_cam PRIVATE ${OpenCV_LIBS} aruco)



