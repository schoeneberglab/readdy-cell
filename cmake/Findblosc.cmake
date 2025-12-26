#find_path(blosc_INCLUDE_DIR NAMES blosc.h HINTS "${CMAKE_PREFIX_PATH}/include" DOC "The blosc include directory")
#find_library(blosc_LIBRARY NAMES blosc HINTS "${CMAKE_PREFIX_PATH}/lib" DOC "The blosc library")
#
#include(FindPackageHandleStandardArgs)
#
#find_package_handle_standard_args(blosc REQUIRED_VARS blosc_LIBRARY blosc_INCLUDE_DIR)
#if(blosc_FOUND)
#    set(blosc_LIBRARIES ${blosc_LIBRARY})
#    set(blosc_INCLUDE_DIRS ${blosc_INCLUDE_DIR})
#    if(NOT TARGET blosc::blosc)
#        add_library(blosc::blosc UNKNOWN IMPORTED)
#        set_target_properties(blosc::blosc PROPERTIES
#                INTERFACE_INCLUDE_DIRECTORIES "${blosc_INCLUDE_DIR}"
#                IMPORTED_LOCATION "${blosc_LIBRARY}")
#    endif()
#endif()
#
#mark_as_advanced(blosc_INCLUDE_DIR blosc_LIBRARY)

## Try to find Blosc 1 first
#find_path(blosc_INCLUDE_DIR NAMES blosc.h HINTS "${CMAKE_PREFIX_PATH}/include" DOC "The blosc include directory")
#find_library(blosc_LIBRARY NAMES blosc HINTS "${CMAKE_PREFIX_PATH}/lib" DOC "The blosc library")
#
## If Blosc 1 is not found, try Blosc 2
#if(NOT blosc_INCLUDE_DIR OR NOT blosc_LIBRARY)
#    message(STATUS "Blosc not found, trying Blosc2...")
#    find_path(blosc_INCLUDE_DIR NAMES blosc2.h HINTS "${CMAKE_PREFIX_PATH}/include" DOC "The blosc2 include directory")
#    find_library(blosc_LIBRARY NAMES blosc2 HINTS "${CMAKE_PREFIX_PATH}/lib" DOC "The blosc2 library")
#endif()
#
## Handle package standard args
#include(FindPackageHandleStandardArgs)
#find_package_handle_standard_args(blosc REQUIRED_VARS blosc_LIBRARY blosc_INCLUDE_DIR)
#
## If found, set up targets
#if(blosc_FOUND)
#    set(blosc_LIBRARIES ${blosc_LIBRARY})
#    set(blosc_INCLUDE_DIRS ${blosc_INCLUDE_DIR})
#    if(NOT TARGET blosc::blosc)
#        add_library(blosc::blosc UNKNOWN IMPORTED)
#        set_target_properties(blosc::blosc PROPERTIES
#                INTERFACE_INCLUDE_DIRECTORIES "${blosc_INCLUDE_DIR}"
#                IMPORTED_LOCATION "${blosc_LIBRARY}")
#    endif()
#endif()
#
#mark_as_advanced(blosc_INCLUDE_DIR blosc_LIBRARY)

# Detect paths relative to the Python interpreter (for venv support)
if(Python_EXECUTABLE)
    get_filename_component(_python_bin_dir "${Python_EXECUTABLE}" DIRECTORY)
    get_filename_component(_python_prefix "${_python_bin_dir}" DIRECTORY)

    # Prefer system hints from Python if not already set
    if(NOT blosc_INCLUDE_DIR)
        list(APPEND blosc_INCLUDE_HINTS "${_python_prefix}/include" "${_python_prefix}/include/blosc")
    endif()

    if(NOT blosc_LIBRARY)
        list(APPEND blosc_LIBRARY_HINTS "${_python_prefix}/lib")
        list(APPEND blosc_LIBRARY_HINTS "${_python_prefix}/lib64")
    endif()
endif()

# Fallback to standard CMAKE_PREFIX_PATH
list(APPEND blosc_INCLUDE_HINTS "${CMAKE_PREFIX_PATH}/include")
list(APPEND blosc_LIBRARY_HINTS "${CMAKE_PREFIX_PATH}/lib")
list(APPEND blosc_LIBRARY_HINTS "${CMAKE_PREFIX_PATH}/lib64")

# Try to find Blosc 1 first
find_path(blosc_INCLUDE_DIR NAMES blosc.h HINTS ${blosc_INCLUDE_HINTS} DOC "The blosc include directory")
find_library(blosc_LIBRARY NAMES blosc HINTS ${blosc_LIBRARY_HINTS} DOC "The blosc library")

## If Blosc 1 is not found, try Blosc 2
#if(NOT blosc_INCLUDE_DIR OR NOT blosc_LIBRARY)
#    message(STATUS "Blosc not found, trying Blosc2...")
#    find_path(blosc_INCLUDE_DIR NAMES blosc2.h HINTS ${blosc_INCLUDE_HINTS} DOC "The blosc2 include directory")
#    find_library(blosc_LIBRARY NAMES blosc2 HINTS ${blosc_LIBRARY_HINTS} DOC "The blosc2 library")
#endif()

# Standard package handle
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(blosc REQUIRED_VARS blosc_LIBRARY blosc_INCLUDE_DIR)

# If found, define imported target
if(blosc_FOUND)
    set(blosc_LIBRARIES ${blosc_LIBRARY})
    set(blosc_INCLUDE_DIRS ${blosc_INCLUDE_DIR})
    if(NOT TARGET blosc::blosc)
        add_library(blosc::blosc UNKNOWN IMPORTED)
        set_target_properties(blosc::blosc PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${blosc_INCLUDE_DIR}"
                IMPORTED_LOCATION "${blosc_LIBRARY}"
        )
    endif()
endif()

message("[Findblosc.cmake] BLOSC_INCLUDE_DIR: ${blosc_INCLUDE_DIR}")
message("[Findblosc.cmake] BLOSC_LIBRARY: ${blosc_LIBRARY}")


mark_as_advanced(blosc_INCLUDE_DIR blosc_LIBRARY)