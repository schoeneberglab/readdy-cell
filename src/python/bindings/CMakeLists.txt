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

set(SOURCE_FILES
        ReaddyBinding.cpp
        common/common.h
        common/CommonModule.cpp
        common/ExportIO.cpp
        common/ReadableReactionRecord.h
        common/ReadableReactionRecord.cpp
        common/ReadableParticle.h
        common/SpdlogPythonSink.h
        common/Utils.cpp
        api/ExportObservables.h
        api/PyTopology.h
        api/ExportLoopApi.cpp
        api/ExportTopologies.cpp
        api/ExportGeometry.cpp
        api/ExportKernelContext.cpp
        api/PyFunction.h
        api/ApiModule.cpp)

message("[bindings CMakeLists.txt] READDY_INCLUDE_DIRS: ${READDY_INCLUDE_DIRS}")
message("[bindings CMakeLists.txt] READDY_DEPENDENT_LIBRARIES: ${READDY_DEPENDENT_LIBRARIES}")
message("[bindings CMakeLists.txt] PYTHON_CPP_MODULES_OUTPUT_DIR: ${PYTHON_CPP_MODULES_OUTPUT_DIR}")
message("[bindings CMakeLists.txt] PYTHON_INSTALL_DIR: ${PYTHON_INSTALL_DIR}")

pybind11_add_module(readdybinding ${SOURCE_FILES})
target_include_directories(readdybinding PUBLIC "${READDY_INCLUDE_DIRS}")
target_link_libraries(readdybinding PUBLIC ${READDY_DEPENDENT_LIBRARIES} readdy)
add_dependencies(CopyPythonSources readdybinding)

# # Set RPATH for the Python module to find dependencies
# if(UNIX AND NOT APPLE)
#     set_target_properties(readdybinding PROPERTIES
#         BUILD_RPATH_USE_ORIGIN TRUE
#         INSTALL_RPATH_USE_LINK_PATH TRUE
#         INSTALL_RPATH "$ORIGIN:$ORIGIN/../lib64:$ORIGIN/../../lib64"
#     )
# endif()

set_target_properties(readdybinding
        PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY "${PYTHON_CPP_MODULES_OUTPUT_DIR}"
        LIBRARY_OUTPUT_DIRECTORY "${PYTHON_CPP_MODULES_OUTPUT_DIR}")


message("PYTHON_INSTALL_DIR: ${PYTHON_INSTALL_DIR}")
install(TARGETS readdybinding LIBRARY DESTINATION ${PYTHON_INSTALL_DIR}/_internal)

# === Install the binding module ===
#add_subdirectory(../../plugins/test_plugin ${CMAKE_CURRENT_BINARY_DIR}/test_plugin)
#add_subdirectory(../../plugins/eabd_integrator ${CMAKE_CURRENT_BINARY_DIR}/eabd_integrator)