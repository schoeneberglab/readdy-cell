function(pybind11_stubgen target)

    # Parse arguments
    set(OPTIONS
        # Package folder containing the Python extension module, relative to the
        # installation prefix (CMAKE_INSTALL_PREFIX). This should match the
        # DESTINATION argument of the given target.
        PACKAGE
        # Destination of the stubs. Should either be an absolute path, or a path
        # relative to the PACKAGE folder.
        DESTINATION
        # The CMake installation component that the stub generation should be
        # part of.
        COMPONENT
    )
    cmake_parse_arguments(STUBGEN "" "${OPTIONS}" "" ${ARGN})
    if (NOT DEFINED STUBGEN_COMPONENT)
        set(STUBGEN_COMPONENT "python_stubs")
    endif()
    if (NOT DEFINED STUBGEN_PACKAGE)
        set(STUBGEN_PACKAGE ${PY_BUILD_CMAKE_IMPORT_NAME})
    endif()
    if (NOT DEFINED STUBGEN_DESTINATION)
        set(STUBGEN_DESTINATION "")
    endif()

    # Locate Python
    if (NOT DEFINED Python3_EXECUTABLE)
        find_package(Python3 REQUIRED COMPONENTS Interpreter)
    endif()

    # Run pybind11-stubgen in the installation prefix
    set(STUBGEN_MOD ${target})
    set(STUBGEN_DIR \"\${CMAKE_INSTALL_PREFIX}/${STUBGEN_PACKAGE}\")
    set(STUBGEN_CMD "\"${Python3_EXECUTABLE}\" -m pybind11_stubgen
        --exit-code -o \"${STUBGEN_DESTINATION}\" \"${STUBGEN_MOD}\"")
    install(CODE "
        message(STATUS \"Executing pybind11-stubgen for ${STUBGEN_MOD} \"
                \"(destination: \\\"${STUBGEN_DESTINATION}\\\")\")
        execute_process(COMMAND ${STUBGEN_CMD}
                        WORKING_DIRECTORY ${STUBGEN_DIR}
                        RESULT_VARIABLE STUBGEN_RET
                        ECHO_OUTPUT_VARIABLE  ECHO_ERROR_VARIABLE
                        COMMAND_ECHO STDERR)
        if(NOT STUBGEN_RET EQUAL 0)
            message(SEND_ERROR \"pybind11-stubgen ${STUBGEN_MOD} failed.\")
        endif()
        " EXCLUDE_FROM_ALL COMPONENT ${STUBGEN_COMPONENT})

endfunction()
