cmake_minimum_required(VERSION 3.18)
project(UR3e_Vision_pick_and_place)

set(CMAKE_CXX_STANDARD 17)
set(OpenCV_DIR "C:/dev/opencv/build")
set(Torch_DIR "C:/dev/libtorch/share/cmake/Torch")

find_package(OpenCV REQUIRED)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS})

add_subdirectory(aruco)

file(COPY "data" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")

add_executable(face face.cpp)
target_link_libraries(face PRIVATE ${OpenCV_LIBS})
target_link_libraries(face PRIVATE ${TORCH_LIBRARIES})
set_property(TARGET face PROPERTY CXX_STANDARD 17)

add_executable(aruco_cam aruco.cpp)
target_link_libraries(aruco_cam PRIVATE ${OpenCV_LIBS} aruco)

if (MSVC)
    file(GLOB TORCH_DLLS PRIVATE ${TORCH_INSTALL_PREFIX}/lib/*.dll)
    add_custom_command(TARGET face
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:face>)
endif (MSVC)

