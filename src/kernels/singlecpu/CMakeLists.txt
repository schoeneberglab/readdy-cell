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


cmake_minimum_required(VERSION 3.16)
project(readdy_kernel_singlecpu LANGUAGES CXX)

SET(SOURCES_DIR "${READDY_GLOBAL_DIR}/readdy/kernels/singlecpu/src")

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

# --- model ---
LIST(APPEND SINGLECPU_SOURCES "${SOURCES_DIR}/model/SCPUParticleData.cpp")

# --- observables ---
LIST(APPEND SINGLECPU_SOURCES "${SOURCES_DIR}/observables/SCPUObservableFactory.cpp")

# --- topology actions ---
LIST(APPEND SINGLECPU_SOURCES "${SOURCES_DIR}/topologies/SCPUTopologyActionFactory.cpp")

# --- all sources ---
LIST(APPEND READDY_ALL_SOURCES ${SINGLECPU_SOURCES})

# Source files
set(SINGLECPU_SOURCES
        src/SCPUKernel.cpp
        src/SCPUStateModel.cpp
        src/SCPUActionFactory.cpp

        src/actions/SCPUCreateNeighborList.cpp
        src/actions/SCPUEvaluateCompartments.cpp
        src/actions/SCPUEvaluateTopologyReactions.cpp
        src/actions/SCPUReactionImpls.cpp

        src/model/SCPUParticleData.cpp

        src/observables/SCPUObservableFactory.cpp

        src/topologies/SCPUTopologyActionFactory.cpp
)

# Header files
set(SINGLECPU_HEADERS
        include/SCPUKernel.h
        include/SCPUStateModel.h

        include/actions/SCPUActionFactory.h
        include/actions/SCPUBreakBonds.h
        include/actions/SCPUCalculateForces.h
        include/actions/SCPUCreateNeighborList.h
        include/actions/SCPUEulerBDIntegrator.h
        include/actions/SCPUEvaluateCompartments.h
        include/actions/SCPUEvaluateTopologyReactions.h
        include/actions/SCPUMiscActions.h
        include/actions/SCPUReactionImpls.h
        include/actions/SCPUReactionUtils.h
        include/actions/SCPUTriggerReaction.h

        include/model/ObservableData.h
        include/model/SCPUNeighborList.h
        include/model/SCPUParticleData.h
        include/model/topologies/SCPUTopologyActionFactory.h
        include/model/topologies/SCPUTopologyActions.h

        include/observables/SCPUAggregators.h
        include/observables/SCPUObservableFactory.h
        include/observables/SCPUObservables.h
)

# IDE organization
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} PREFIX "Source" FILES ${SINGLECPU_SOURCES})
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} PREFIX "Header" FILES ${SINGLECPU_HEADERS})

# Define the shared library
add_library(${PROJECT_NAME} SHARED ${SINGLECPU_SOURCES} ${SINGLECPU_HEADERS})

# Public include directories
target_include_directories(${PROJECT_NAME}
        PUBLIC
        ${READDY_INCLUDE_DIRS}
        include
)

# Link against readdy components (assumed to be Conan-provided or local targets)
target_link_libraries(${PROJECT_NAME}
        PRIVATE
        ${CMAKE_DL_LIBS}
        readdy_common
        readdy_model
)

# Output configuration
set_target_properties(${PROJECT_NAME} PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        LINK_FLAGS "${EXTRA_LINK_FLAGS}"
        COMPILE_FLAGS "${EXTRA_COMPILE_FLAGS}"
        ARCHIVE_OUTPUT_DIRECTORY "${READDY_PLUGIN_OUTPUT_DIRECTORY}"
        LIBRARY_OUTPUT_DIRECTORY "${READDY_PLUGIN_OUTPUT_DIRECTORY}"
        RUNTIME_OUTPUT_DIRECTORY "${READDY_PLUGIN_OUTPUT_DIRECTORY}"
)

# Install the shared library
install(TARGETS ${PROJECT_NAME}
        RUNTIME DESTINATION ${INSTALL_DIR}/lib/readdy_plugins
        LIBRARY DESTINATION ${INSTALL_DIR}/lib/readdy_plugins
        ARCHIVE DESTINATION ${INSTALL_DIR}/lib/readdy_plugins
)

# Install headers
install(DIRECTORY include
        DESTINATION ${INSTALL_DIR}/include/readdy/kernel/singlecpu
        FILES_MATCHING PATTERN "*.h"
)



## Tests
#if (READDY_CREATE_TEST_TARGET)
#    add_subdirectory(test)
#endif()