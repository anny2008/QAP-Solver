cmake_minimum_required(VERSION 3.15)
project(RTL1_SCIP_Solver)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find SCIP
find_package(SCIP REQUIRED CONFIG)

# Include SCIP headers
include_directories(${SCIP_INCLUDE_DIRS})

# Add executable
add_executable(rtl1_solver rtl1_solver.cpp)

# Link SCIP libraries - use full path to force static linking where possible
target_link_directories(rtl1_solver PRIVATE /home/local.isima.fr/antran/UFF/SCIPOptSuite-9.2.0-Linux/lib)
target_link_libraries(rtl1_solver /home/local.isima.fr/antran/UFF/SCIPOptSuite-9.2.0-Linux/lib/libscip.so)
target_link_libraries(rtl1_solver /home/local.isima.fr/antran/UFF/SCIPOptSuite-9.2.0-Linux/lib/libsoplexshared.so)
target_link_libraries(rtl1_solver pthread dl)

# Print info
message(STATUS "SCIP found at: ${SCIP_DIR}")
message(STATUS "SCIP libraries: ${SCIP_LIBRARIES}")
message(STATUS "SCIP include dirs: ${SCIP_INCLUDE_DIRS}")
