####################################################################
# Copyright © 2018 Computational Molecular Biology Group,          #
#                  Freie Universität Berlin (GER)                  #
#                                                                  #
# Redistribution and use in source and binary forms, with or       #
# without modification, are permitted provided that the            #
# following conditions are met:                                    #
#  1. Redistributions of source code must retain the above         #
#     copyright notice, this list of conditions and the            #
#     following disclaimer.                                        #
#  2. Redistributions in binary form must reproduce the above      #
#     copyright notice, this list of conditions and the following  #
#     disclaimer in the documentation and/or other materials       #
#     provided with the distribution.                              #
#  3. Neither the name of the copyright holder nor the names of    #
#     its contributors may be used to endorse or promote products  #
#     derived from this software without specific                  #
#     prior written permission.                                    #
#                                                                  #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND           #
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,      #
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF         #
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE         #
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR            #
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,     #
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,         #
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; #
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER #
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,      #
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)    #
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF      #
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                       #
####################################################################

if (POLICY CMP0048)
    cmake_policy(SET CMP0048 NEW)
endif (POLICY CMP0048)

cmake_minimum_required(VERSION 3.17) # minimum cmake version
SET(CMAKE_CXX_STANDARD 20) # we use c++20
SET(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PY_VERSION_SUFFIX "")  # e.g. b3.dev1, see PEP 440
set(PY_FULL_VERSION ${PROJECT_VERSION}${PY_VERSION_SUFFIX})

PROJECT(readdy_main VERSION 0.0 LANGUAGES C CXX)

# Python & pybind11
if (CMAKE_CROSSCOMPILING AND NOT (APPLE AND "$ENV{CIBUILDWHEEL}" STREQUAL "1"))
    find_package(Python3 REQUIRED COMPONENTS Development.Module)
else()
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
endif()

#####################################
#                                   #
# User configurable cache variables #
#                                   #
#####################################
set(INSTALL_DIR "${CMAKE_INSTALL_PREFIX}" CACHE PATH "Installation root for anything but python bindings and files")
set(PYTHON_INSTALL_DIR "${SP_DIR}/readdy" CACHE PATH "Installation directory relative to SP_DIR (python/site-packages)")

# test configuration
set(READDY_CREATE_TEST_TARGET ON CACHE BOOL "Determines if the test target should be added.")
set(READDY_INSTALL_UNIT_TEST_EXECUTABLE OFF CACHE BOOL "Determines if the unit test executable should be installed. Requires the cache variable READDY_CREATE_TEST_TARGET to be ON.")
set(READDY_KERNELS_TO_TEST "SingleCPU,CPU" CACHE STRING "Variable holding which kernels should be tested in the core library.")
set(READDY_INSTALL_HEADERS ON CACHE BOOL "Whether to install header files")

# decide if api wrappers are being generated
set(READDY_BUILD_PYTHON_WRAPPER ON CACHE BOOL "Build Python wrappers")

set(READDY_DEBUG_PYTHON_MODULES ON CACHE BOOL "If this flag is set to ON, the compiled python modules get installed directly into the source tree.")
set(READDY_DEBUG_CONDA_ROOT_DIR "" CACHE PATH "This only does something, if READDY_DEBUG_PYTHON_MODULES is set to ON. Then, it compiles the readdy libraries into the @conda@/lib directory.")

# documentation target (make doc)
set(READDY_GENERATE_DOCUMENTATION_TARGET OFF CACHE BOOL "Should generate target for documentation or not.")

# log configuration variables of interest
set(READDY_LOG_CMAKE_CONFIGURATION ON CACHE BOOL "If turned on, the status of relevant cmake variables is logged at configuration time.")

# build the mpi kernel
set(READDY_BUILD_MPI_KERNEL OFF CACHE BOOL "Whether to build the MPI kernel or not")

# build the scenarios executables for performance benchmarking
set(READDY_BUILD_SCENARIOS OFF CACHE BOOL "Whether to build the Scenarios executable or not")

#####################################
#                                   #
# Basic setup of the project        #
#                                   #
#####################################

# Extend cmake module path to include readdy custom modules
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Ensure that the build directory is not the source directory
include(MacroOutOfSourceBuild)
MACRO_ENSURE_OUT_OF_SOURCE_BUILD("${PROJECT_NAME} requires an out source build.")

# Makefile settings
SET(CMAKE_VERBOSE_MAKEFILE OFF)
SET(CMAKE_COLOR_MAKEFILE ON)

# Version
set(READDY_VERSION "v0.0" CACHE STRING "The version")
mark_as_advanced(READDY_VERSION)
set(READDY_BUILD_STRING "dev" CACHE STRING "Build string")
mark_as_advanced(READDY_BUILD_STRING)

message(STATUS "====================================================")
message(STATUS "CMake configure for readdy version ${READDY_VERSION}-${READDY_BUILD_STRING}")
message(STATUS "====================================================")

if(APPLE)
    set(CMAKE_MACOSX_RPATH TRUE)  # any shared lib being built has rpath enabled
endif()

# Set RPATH for Linux to handle library dependencies automatically
if(UNIX AND NOT APPLE)
    # Use $ORIGIN to make paths relative to the executable/library location
    set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
    
    # Add common library paths that might contain dependencies
    list(APPEND CMAKE_INSTALL_RPATH 
        "$ORIGIN" 
        "$ORIGIN/../lib" 
        "$ORIGIN/../lib64"
        "$ORIGIN/../../lib64"  # For Python site-packages structure
        "${CMAKE_INSTALL_PREFIX}/lib"
        "${CMAKE_INSTALL_PREFIX}/lib64"
    )
endif()

# Always build readdy as shared lib
SET(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared libs" FORCE)

# Directory Variables
SET(READDY_GLOBAL_DIR "${PROJECT_SOURCE_DIR}")
SET(READDY_LIBS_DIR "${PROJECT_SOURCE_DIR}/include/readdy/detail")
SET(READDY_READDY_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include/readdy")
SET(READDY_GRAPHS_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include")
SET(READDY_H5RD_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include")
SET(READDY_READDY_SRC_DIR "${PROJECT_SOURCE_DIR}/src")

# Compile in debug mode by default
IF (NOT CMAKE_BUILD_TYPE)
    MESSAGE(STATUS "Setting build type to 'DEBUG' as none was specified.")
    SET(CMAKE_BUILD_TYPE DEBUG)
ELSE (NOT CMAKE_BUILD_TYPE)
    MESSAGE(STATUS "Using specified '${CMAKE_BUILD_TYPE}' build type.")
ENDIF (NOT CMAKE_BUILD_TYPE)

SET(EXTRA_COMPILE_FLAGS "")
SET(EXTRA_LINK_FLAGS ${EXTRA_COMPILE_FLAGS})
IF (CMAKE_SYSTEM_NAME MATCHES "Linux")
    IF (NOT ANDROID)
        if (CMAKE_BUILD_TYPE STREQUAL "Release")
            SET(EXTRA_COMPILE_FLAGS "${EXTRA_COMPILE_FLAGS}")
        endif()
        SET(EXTRA_LINK_FLAGS "${EXTRA_LINK_FLAGS} -Wl,--no-as-needed -lrt")
    ENDIF (NOT ANDROID)
ENDIF(CMAKE_SYSTEM_NAME MATCHES "Linux")

# Project name
set(READDY_PROJECT_NAME "readdy")

if (READDY_LOG_CMAKE_CONFIGURATION)
    include(LogConfiguration)
endif (READDY_LOG_CMAKE_CONFIGURATION)

set(PYBIND11_FINDPYTHON ON)
set(WINDOWS_EXPORT_ALL_SYMBOLS ON)

#####################################
#                                   #
# setup testing                     #
#                                   #
#####################################

if (READDY_CREATE_TEST_TARGET)
    add_definitions(-DCATCH_CONFIG_NO_CPP17_UNCAUGHT_EXCEPTIONS)  # workaround for macos which chokes on some cpp17 stuff
    find_package(Catch2 REQUIRED)
    add_subdirectory(tests)
else()
    SET(READDY_CREATE_TEST_TARGET OFF) # disable testing if there was no submodule
    SET(READDY_INSTALL_UNIT_TEST_EXECUTABLE OFF) # no installation of test executable
endif()

#####################################
#                                   #
# readdy core                       #
#                                   #
#####################################

# Dependencies
find_package(Threads REQUIRED)
find_package(spdlog REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(fmt REQUIRED)
find_package(pybind11 REQUIRED)
find_package(blosc REQUIRED)
#include(cmake/PyBind11StubGen.cmake)

# ========== sources_common ========== #

set(SOURCES_DIR "${READDY_GLOBAL_DIR}/src/common")

# include dirs
set(COMMON_INCLUDE_DIRS "${READDY_LIBS_DIR};${blosc_INCLUDE_DIR}" CACHE INTERNAL "Common include dirs")

# dependent libraries
set(READDY_COMMON_LIBRARIES "nlohmann_json::nlohmann_json;spdlog::spdlog_header_only;blosc::blosc;fmt::fmt-header-only" CACHE INTERNAL "Common libraries")

# sources
list(APPEND READDY_COMMON_SOURCES "${SOURCES_DIR}/../api/KernelConfiguration.cpp")
list(APPEND READDY_COMMON_SOURCES "${SOURCES_DIR}/Utils.cpp")
list(APPEND READDY_COMMON_SOURCES "${SOURCES_DIR}/filesystem.cpp")
list(APPEND READDY_COMMON_SOURCES "${SOURCES_DIR}/Config.cpp")
list(APPEND READDY_COMMON_SOURCES "${SOURCES_DIR}/logging.cpp")
list(APPEND READDY_COMMON_SOURCES "${SOURCES_DIR}/Timer.cpp")

# all sources
list(APPEND READDY_ALL_SOURCES ${READDY_COMMON_SOURCES})

# ========== sources_io ========== #
# include dirs
SET(SOURCES_DIR "${READDY_GLOBAL_DIR}/src/io")

# hdf5
SET(HDF5_USE_STATIC_LIBRARIES OFF)
FIND_PACKAGE(HDF5 COMPONENTS HL REQUIRED)
MESSAGE(STATUS "HDF5_FOUND: ${HDF5_FOUND}, version ${HDF5_VERSION}")

# includes
SET(IO_INCLUDE_DIRS "${COMMON_INCLUDE_DIRS};${HDF5_INCLUDE_DIRS};${HDF5_HL_INCLUDE_DIR}" CACHE INTERNAL "IO Include dirs")

# libraries
SET(READDY_IO_LIBRARIES "${READDY_COMMON_LIBRARIES};${HDF5_LIBRARIES};${HDF5_HL_LIBRARIES}" CACHE INTERNAL "IO Libraries")
LIST(REMOVE_DUPLICATES READDY_IO_LIBRARIES)

# sources
LIST(APPEND READDY_IO_SOURCES "${SOURCES_DIR}/BloscFilter.cpp")
LIST(APPEND READDY_IO_SOURCES "${SOURCES_DIR}/blosc_filter.h")
LIST(APPEND READDY_IO_SOURCES "${SOURCES_DIR}/blosc_filter.c")
LIST(APPEND READDY_IO_SOURCES "${SOURCES_DIR}/blosc_filter.h")
LIST(APPEND READDY_IO_SOURCES "${SOURCES_DIR}/blosc_plugin.c")

# all sources
LIST(APPEND READDY_ALL_SOURCES ${READDY_IO_SOURCES})

# ========== sources_model ========== #

SET(SOURCES_DIR "${READDY_GLOBAL_DIR}/src/model")

# libraries
SET(READDY_MODEL_LIBRARIES "${READDY_IO_LIBRARIES}" CACHE INTERNAL "Model libraries")

# includes
SET(MODEL_INCLUDE_DIRS "${IO_INCLUDE_DIRS}" CACHE INTERNAL "Model include dirs")

# sources
LIST(APPEND READDY_MODEL_SOURCES "${SOURCES_DIR}/Context.cpp")
LIST(APPEND READDY_MODEL_SOURCES "${SOURCES_DIR}/StateModel.cpp")
LIST(APPEND READDY_MODEL_SOURCES "${SOURCES_DIR}/Particle.cpp")
LIST(APPEND READDY_MODEL_SOURCES "${SOURCES_DIR}/ParticleTypeRegistry.cpp")
LIST(APPEND READDY_MODEL_SOURCES "${SOURCES_DIR}/Actions.cpp")
LIST(APPEND READDY_MODEL_SOURCES "${SOURCES_DIR}/Utils.cpp")
LIST(APPEND READDY_MODEL_SOURCES "${SOURCES_DIR}/IOUtils.cpp")

# compartments
LIST(APPEND READDY_MODEL_SOURCES "${SOURCES_DIR}/compartments/Compartments.cpp")
LIST(APPEND READDY_MODEL_SOURCES "${SOURCES_DIR}/compartments/CompartmentRegistry.cpp")

# potentials
LIST(APPEND READDY_MODEL_SOURCES "${SOURCES_DIR}/potentials/Potentials.cpp")

# reactions
LIST(APPEND READDY_MODEL_SOURCES "${SOURCES_DIR}/reactions/Reactions.cpp")
LIST(APPEND READDY_MODEL_SOURCES "${SOURCES_DIR}/reactions/ReactionRegistry.cpp")

# topologies
LIST(APPEND READDY_MODEL_SOURCES "${SOURCES_DIR}/topologies/GraphTopology.cpp")
LIST(APPEND READDY_MODEL_SOURCES "${SOURCES_DIR}/topologies/TopologyRegistry.cpp")
LIST(APPEND READDY_MODEL_SOURCES "${SOURCES_DIR}/topologies/potentials/BondedPotential.cpp")
LIST(APPEND READDY_MODEL_SOURCES "${SOURCES_DIR}/topologies/potentials/AnglePotential.cpp")
LIST(APPEND READDY_MODEL_SOURCES "${SOURCES_DIR}/topologies/potentials/TorsionPotential.cpp")
LIST(APPEND READDY_MODEL_SOURCES "${SOURCES_DIR}/topologies/reactions/Recipe.cpp")
LIST(APPEND READDY_MODEL_SOURCES "${SOURCES_DIR}/topologies/reactions/TopologyReactionActions.cpp")
LIST(APPEND READDY_MODEL_SOURCES "${SOURCES_DIR}/topologies/reactions/StructuralTopologyReaction.cpp")
LIST(APPEND READDY_MODEL_SOURCES "${SOURCES_DIR}/topologies/reactions/SpatialTopologyReaction.cpp")

# observables
LIST(APPEND READDY_MODEL_SOURCES "${SOURCES_DIR}/observables/Types.cpp")
LIST(APPEND READDY_MODEL_SOURCES "${SOURCES_DIR}/observables/Trajectory.cpp")
LIST(APPEND READDY_MODEL_SOURCES "${SOURCES_DIR}/observables/HistogramAlongAxis.cpp")
LIST(APPEND READDY_MODEL_SOURCES "${SOURCES_DIR}/observables/Particles.cpp")
LIST(APPEND READDY_MODEL_SOURCES "${SOURCES_DIR}/observables/NParticles.cpp")
LIST(APPEND READDY_MODEL_SOURCES "${SOURCES_DIR}/observables/Forces.cpp")
LIST(APPEND READDY_MODEL_SOURCES "${SOURCES_DIR}/observables/Reactions.cpp")
LIST(APPEND READDY_MODEL_SOURCES "${SOURCES_DIR}/observables/ReactionCounts.cpp")
LIST(APPEND READDY_MODEL_SOURCES "${SOURCES_DIR}/observables/Energy.cpp")
LIST(APPEND READDY_MODEL_SOURCES "${SOURCES_DIR}/observables/Positions.cpp")
LIST(APPEND READDY_MODEL_SOURCES "${SOURCES_DIR}/observables/Topologies.cpp")
LIST(APPEND READDY_MODEL_SOURCES "${SOURCES_DIR}/observables/RadialDistribution.cpp")
LIST(APPEND READDY_MODEL_SOURCES "${SOURCES_DIR}/observables/Virial.cpp")

# all sources
LIST(APPEND READDY_ALL_SOURCES ${READDY_MODEL_SOURCES})

# ========== sources_plugin ========== #

SET(SOURCES_DIR "${READDY_GLOBAL_DIR}/src/plugin")

# includes
SET(PLUGIN_INCLUDE_DIRS "${MODEL_INCLUDE_DIRS}" CACHE INTERNAL "Plugin include dirs")

# libraries
SET(READDY_PLUGIN_LIBRARIES "${READDY_MODEL_LIBRARIES};${CMAKE_DL_LIBS};${READDY_SINGLECPU_DEPENDENT_LIBRARIES}" CACHE INTERNAL "Plugin libraries")

# sources
LIST(APPEND READDY_PLUGIN_SOURCES "${SOURCES_DIR}/KernelProvider.cpp")

# all sources
LIST(APPEND READDY_ALL_SOURCES ${READDY_PLUGIN_SOURCES})

# ========== sources_singlecpu ========== #
SET(SOURCES_DIR "${READDY_GLOBAL_DIR}/src/kernels/singlecpu/src")

# libraries
SET(READDY_SINGLECPU_DEPENDENT_LIBRARIES "${READDY_COMMON_LIBRARIES};${READDY_MODEL_LIBRARIES}" CACHE INTERNAL "Readdy libraries")
LIST(REMOVE_DUPLICATES READDY_SINGLECPU_DEPENDENT_LIBRARIES)

# sources
LIST(APPEND SINGLECPU_SOURCES "${SOURCES_DIR}/SCPUKernel.cpp")
LIST(APPEND SINGLECPU_SOURCES "${SOURCES_DIR}/SCPUActionFactory.cpp")
LIST(APPEND SINGLECPU_SOURCES "${SOURCES_DIR}/SCPUStateModel.cpp")

# --- actions ---
LIST(APPEND SINGLECPU_SOURCES "${SOURCES_DIR}/actions/SCPUCreateNeighborList.cpp")
LIST(APPEND SINGLECPU_SOURCES "${SOURCES_DIR}/actions/SCPUReactionImpls.cpp")
LIST(APPEND SINGLECPU_SOURCES "${SOURCES_DIR}/actions/SCPUEvaluateCompartments.cpp")
LIST(APPEND SINGLECPU_SOURCES "${SOURCES_DIR}/actions/SCPUEvaluateTopologyReactions.cpp")
LIST(APPEND SINGLECPU_SOURCES "${SOURCES_DIR}/actions/SCPUEulerABDIntegrator.cpp") # TESTING!

# --- model ---
LIST(APPEND SINGLECPU_SOURCES "${SOURCES_DIR}/model/SCPUParticleData.cpp")

# --- observables ---
LIST(APPEND SINGLECPU_SOURCES "${SOURCES_DIR}/observables/SCPUObservableFactory.cpp")

# --- topology actions ---
LIST(APPEND SINGLECPU_SOURCES "${SOURCES_DIR}/topologies/SCPUTopologyActionFactory.cpp")

# --- all sources ---
LIST(APPEND READDY_ALL_SOURCES ${SINGLECPU_SOURCES})

# ========== sources_cpu ========== #
SET(SOURCES_DIR "${READDY_GLOBAL_DIR}/src/kernels/cpu/src")
SET(CPU_INCLUDE_DIR "${READDY_GLOBAL_DIR}/src/kernels/cpu/include")

# --- main sources ---
LIST(APPEND CPU_SOURCES "${SOURCES_DIR}/CPUKernel.cpp")
LIST(APPEND CPU_SOURCES "${SOURCES_DIR}/CPUStateModel.cpp")
LIST(APPEND CPU_SOURCES "${SOURCES_DIR}/observables/CPUObservableFactory.cpp")
LIST(APPEND CPU_SOURCES "${SOURCES_DIR}/observables/CPUObservables.cpp")

# --- neighbor list ---
LIST(APPEND CPU_SOURCES "${SOURCES_DIR}/nl/CellLinkedList.cpp")
LIST(APPEND CPU_SOURCES "${SOURCES_DIR}/nl/ContiguousCellLinkedList.cpp")

# --- actions ---
LIST(APPEND CPU_SOURCES "${SOURCES_DIR}/actions/CPUActionFactory.cpp")
LIST(APPEND CPU_SOURCES "${SOURCES_DIR}/actions/CPUEulerBDIntegrator.cpp")
LIST(APPEND CPU_SOURCES "${SOURCES_DIR}/actions/CPUCalculateForces.cpp")
LIST(APPEND CPU_SOURCES "${SOURCES_DIR}/actions/CPUEvaluateCompartments.cpp")
LIST(APPEND CPU_SOURCES "${SOURCES_DIR}/actions/CPUEvaluateTopologyReactions.cpp")
LIST(APPEND CPU_SOURCES "${SOURCES_DIR}/actions/reactions/ReactionUtils.cpp")
LIST(APPEND CPU_SOURCES "${SOURCES_DIR}/actions/reactions/Event.cpp")
LIST(APPEND CPU_SOURCES "${SOURCES_DIR}/actions/reactions/CPUUncontrolledApproximation.cpp")
LIST(APPEND CPU_SOURCES "${SOURCES_DIR}/actions/reactions/CPUGillespie.cpp")
LIST(APPEND CPU_SOURCES "${SOURCES_DIR}/actions/topologies/CPUTopologyActions.cpp")
LIST(APPEND CPU_SOURCES "${SOURCES_DIR}/actions/topologies/CPUTopologyActionFactory.cpp")
LIST(APPEND CPU_SOURCES "${SOURCES_DIR}/actions/CPUEulerABDIntegrator.cpp") # TESTING!

LIST(APPEND READDY_ALL_SOURCES ${CPU_SOURCES})

# ========== sources_readdy ========== #
SET(SOURCES_DIR "${READDY_GLOBAL_DIR}/src")

# includes
LIST(REMOVE_DUPLICATES READDY_INCLUDE_DIRS)

# Includes
SET(READDY_INCLUDE_DIRS
        "${READDY_LIBS_DIR}"
        "${READDY_READDY_INCLUDE_DIR}"
        "${READDY_GRAPHS_INCLUDE_DIR}"
        "${READDY_H5RD_INCLUDE_DIR}"
        "${IO_INCLUDE_DIRS}"
        "${PLUGIN_INCLUDE_DIRS}"
        "${CMAKE_SOURCE_DIR}/src/kernels/cpu/include"
        "${CMAKE_SOURCE_DIR}/src/kernels/singlecpu/include"
        CACHE INTERNAL "Readdy include dirs"
)

# libraries
SET(READDY_DEPENDENT_LIBRARIES
        "${READDY_COMMON_LIBRARIES}"
        "${READDY_MODEL_LIBRARIES}"
        "${READDY_IO_LIBRARIES}"
        "${READDY_PLUGIN_LIBRARIES}"
        "${READDY_SINGLECPU_DEPENDENT_LIBRARIES}"
        CACHE INTERNAL "Readdy libraries"
)
LIST(REMOVE_DUPLICATES READDY_DEPENDENT_LIBRARIES)

# sources
LIST(APPEND READDY_MAIN_SOURCES "")

# all sources
LIST(APPEND READDY_ALL_SOURCES ${READDY_MAIN_SOURCES})

# Create library
message(STATUS "Libraries ReaDDy depends on: ${READDY_DEPENDENT_LIBRARIES}")
message(STATUS "Include directories: ${READDY_INCLUDE_DIRS}")

add_library(${READDY_PROJECT_NAME} ${READDY_ALL_SOURCES}) #DEBUGGING

target_include_directories(${READDY_PROJECT_NAME} PUBLIC ${READDY_INCLUDE_DIRS})
target_link_libraries(${READDY_PROJECT_NAME} PUBLIC ${READDY_DEPENDENT_LIBRARIES})

set_target_properties(${READDY_PROJECT_NAME} PROPERTIES
        LINK_FLAGS "${EXTRA_LINK_FLAGS}"
        COMPILE_FLAGS "${EXTRA_COMPILE_FLAGS}")

target_compile_definitions(${READDY_PROJECT_NAME} PUBLIC LIBRARY_EXPORTS=1)

#####################################
#                                   #
# readdy kernels                    #
#                                   #
#####################################
add_subdirectory(src/kernels)

#####################################
#                                   #
# installation (for subprojects     #
# handled in their respective       #
# build files)                      #
#                                   #
#####################################
if(READDY_INSTALL_HEADERS)
    install (
            DIRECTORY ${CMAKE_SOURCE_DIR}/include/readdy/
            DESTINATION ${INSTALL_DIR}/include/readdy
            FILES_MATCHING PATTERN "*.h*")
    install (
            DIRECTORY ${CMAKE_SOURCE_DIR}/src/kernels/singlecpu/include/
            DESTINATION ${INSTALL_DIR}/include/readdy/kernel/singlecpu
            FILES_MATCHING PATTERN "*.h*"
    )
    install (
            DIRECTORY ${CMAKE_SOURCE_DIR}/src/kernels/cpu/include/
            DESTINATION ${INSTALL_DIR}/include/readdy/kernel/cpu
            FILES_MATCHING PATTERN "*.h*"
    )
endif()
install(TARGETS ${READDY_PROJECT_NAME} LIBRARY DESTINATION "${INSTALL_DIR}/lib")

#####################################
#                                   #
# readdy wrappers                   #
#                                   #
#####################################
IF (READDY_BUILD_PYTHON_WRAPPER)
    message("[CMakeLists.txt] READDY_BUILD_PYTHON_WRAPPER: ${READDY_BUILD_PYTHON_WRAPPER}")
    add_subdirectory(src/python)
ENDIF (READDY_BUILD_PYTHON_WRAPPER)


#####################################
#                                   #
# tests                             #
#                                   #
#####################################
if (READDY_CREATE_TEST_TARGET)
    # New
    add_subdirectory(tests/core)
    add_test(NAME readdy-tests COMMAND runUnitTests)
    add_subdirectory(tests/kernels/singlecpu)
    add_subdirectory(tests/kernels/cpu)
endif ()

if(READDY_BUILD_SCENARIOS)
    add_subdirectory(examples/scenarios)
    if(READDY_BUILD_MPI_KERNEL)
        add_subdirectory(examples/mpi_scenarios)
    endif()
endif()


#####################################
#                                   #
# documentation (doxygen)           #
#                                   #
#####################################
IF(READDY_GENERATE_DOCUMENTATION_TARGET)
    FIND_PACKAGE (Doxygen REQUIRED)
    FIND_PROGRAM (DOXYFILE_MAKE make)
    MARK_AS_ADVANCED (DOXYFILE_MAKE)
    CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
    ADD_CUSTOM_TARGET(doc
            ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Generating API documentation with Doxygen" VERBATIM)
    IF(MAKEINDEX_COMPILER AND DOXYFILE_MAKE)
        ADD_CUSTOM_COMMAND(
                TARGET doc
                POST_BUILD
                COMMAND "${DOXYFILE_MAKE}"
                COMMENT	"Running Doxygen in ${CMAKE_CURRENT_BINARY_DIR}/docs..."
                WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/docs")
    ENDIF()
ENDIF(READDY_GENERATE_DOCUMENTATION_TARGET)

## Stub generation (optional)
#set(PY_STUBS_INSTALLDIR "" CACHE PATH "Installation directory for Python stubs (relative to Python package)")
#option(WITH_PY_STUBS "Generate Python stub files (.pyi)" ON)
#if (WITH_PY_STUBS)
#    include(cmake/PyBind11StubGen.cmake)
#    pybind11_stubgen(readdy
#            PACKAGE readdy
#            DESTINATION ${PY_STUBS_INSTALLDIR}
#            COMPONENT python_stubs
#    )
#endif()