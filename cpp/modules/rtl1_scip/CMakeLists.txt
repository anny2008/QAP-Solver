cmake_minimum_required(VERSION 3.10)
project(rtl1_scip)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find SCIP library
# Update the path based on your SCIP installation
set(SCIP_DIR "/home/local.isima.fr/antran/UFF/scip-9.2.0")
find_package(SCIP REQUIRED PATHS ${SCIP_DIR})

# Include SCIP headers
include_directories(${SCIP_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Create shared library for Python ctypes
add_library(rtl1_scip_shared SHARED
    rtl1_solver.cpp
    rtl1_solver_c.cpp
)

# Link SCIP library
target_link_libraries(rtl1_scip_shared PRIVATE ${SCIP_LIBRARIES})

# Set output directory
set_target_properties(rtl1_scip_shared PROPERTIES
    OUTPUT_NAME "rtl1_scip"
    PREFIX "lib"
)

# Installation
install(TARGETS rtl1_scip_shared
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)
