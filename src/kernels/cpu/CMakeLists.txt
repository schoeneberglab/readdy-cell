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
project(readdy_kernel_cpu LANGUAGES CXX)

# Source files
set(CPU_SOURCES
        src/CPUKernel.cpp
        src/CPUStateModel.cpp

        src/actions/CPUActionFactory.cpp
        src/actions/CPUCalculateForces.cpp
        src/actions/CPUEulerBDIntegrator.cpp
        src/actions/CPUEvaluateCompartments.cpp
        src/actions/CPUEvaluateTopologyReactions.cpp

        src/actions/reactions/CPUGillespie.cpp
        src/actions/reactions/CPUUncontrolledApproximation.cpp
        src/actions/reactions/Event.cpp
        src/actions/reactions/ReactionUtils.cpp

        src/actions/topologies/CPUTopologyActionFactory.cpp
        src/actions/topologies/CPUTopologyActions.cpp

        src/nl/CellLinkedList.cpp
        src/nl/ContiguousCellLinkedList.cpp

        src/observables/CPUObservableFactory.cpp
        src/observables/CPUObservables.cpp
)

# Header files
set(CPU_HEADERS
        include/CPUKernel.h
        include/CPUStateModel.h
        include/pool.h

        include/actions/CPUActionFactory.h
        include/actions/CPUBreakBonds.h
        include/actions/CPUCalculateForces.h
        include/actions/CPUCreateNeighborList.h
        include/actions/CPUEulerBDIntegrator.h
        include/actions/CPUEvaluateCompartments.h
        include/actions/CPUEvaluateTopologyReactions.h
        include/actions/CPUMiscActions.h
        include/actions/CPUTriggerReaction.h

        include/actions/reactions/CPUGillespie.h
        include/actions/reactions/CPUUncontrolledApproximation.h
        include/actions/reactions/Event.h
        include/actions/reactions/ReactionUtils.h

        include/actions/topologies/CPUTopologyActionFactory.h
        include/actions/topologies/CPUTopologyActions.h

        include/data/DataContainer.h
        include/data/DefaultDataContainer.h
        include/data/ObservableData.h

        include/nl/CellLinkedList.h
        include/nl/ContiguousCellLinkedList.h

        include/observables/CPUObservableFactory.h
        include/observables/CPUObservables.h
)

# IDE structur

# Define shared library target
add_library(
        ${PROJECT_NAME}
        SHARED
        ${CPU_SOURCES}
        ${CPU_HEADERS}
)

# Include directories
target_include_directories(
        ${PROJECT_NAME}
        PUBLIC
        ${READDY_INCLUDE_DIRS}
        include
        ../singlecpu/include # Some shared headers with singlecpu; TODO: Moving to common?
)

# Link against core ReaDDy components
target_link_libraries(
        ${PROJECT_NAME}
        PRIVATE
        ${CMAKE_DL_LIBS}
        readdy
)

# Target properties
set_target_properties(${PROJECT_NAME} PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        LINK_FLAGS "${EXTRA_LINK_FLAGS}"
        COMPILE_FLAGS "${EXTRA_COMPILE_FLAGS}"
        ARCHIVE_OUTPUT_DIRECTORY "${READDY_PLUGIN_OUTPUT_DIRECTORY}"
        LIBRARY_OUTPUT_DIRECTORY "${READDY_PLUGIN_OUTPUT_DIRECTORY}"
        RUNTIME_OUTPUT_DIRECTORY "${READDY_PLUGIN_OUTPUT_DIRECTORY}"
)

# Install plugin
install(TARGETS ${PROJECT_NAME}
        RUNTIME DESTINATION ${INSTALL_DIR}/lib/readdy_plugins
        LIBRARY DESTINATION ${INSTALL_DIR}/lib/readdy_plugins
        ARCHIVE DESTINATION ${INSTALL_DIR}/lib/readdy_plugins
)

message("(kernels/cpu) Install directory: ${INSTALL_DIR}")

## Install headers
#install(DIRECTORY include
#        DESTINATION ${INSTALL_DIR}/include/readdy/kernel/cpu
#        FILES_MATCHING PATTERN "*.h"
#)

## Enable test target
#if (READDY_CREATE_TEST_TARGET)
#    add_subdirectory(test)
#endif()