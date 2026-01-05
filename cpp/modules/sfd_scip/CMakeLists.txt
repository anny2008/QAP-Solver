cmake_minimum_required(VERSION 3.10)
project(sfd_scip)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find SCIP library
set(SCIP_DIR "/home/local.isima.fr/antran/UFF/scip-9.2.0")
find_package(SCIP REQUIRED PATHS ${SCIP_DIR})

# Include SCIP headers
include_directories(${SCIP_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../core)

# Create executable
add_executable(sfd_solver
    sfd_solver.cpp
)

# Link SCIP library
target_link_libraries(sfd_solver PRIVATE ${SCIP_LIBRARIES})

# Installation
install(TARGETS sfd_solver
        RUNTIME DESTINATION bin)
