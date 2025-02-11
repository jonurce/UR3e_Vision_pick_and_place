cmake_minimum_required(VERSION 3.15)
project(UR3e_Vision_pick_and_place)

set(CMAKE_CXX_STANDARD 17)
set(OpenCV_DIR "C:/dev/opencv/build")

find_package(OpenCV REQUIRED)

file(COPY "data" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")

add_executable(main main.cpp)
target_link_libraries(main PRIVATE ${OpenCV_LIBS})
target_include_directories(main PRIVATE ${OpenCV_INCLUDE_DIRS})




