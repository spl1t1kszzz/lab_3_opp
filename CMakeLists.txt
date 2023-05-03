cmake_minimum_required(VERSION 3.24)
project(lab_3_opp)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_FLAGS -O3 -lblas)
add_executable(lab_3_opp main.cpp)

#set(CMAKE_CXX_COMPILER mpicxx)

#target_link_libraries(lab_3_opp ${MPI_LIBRARIES})


