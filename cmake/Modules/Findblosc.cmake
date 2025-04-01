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
# Try to find Blosc 1 first
find_path(blosc_INCLUDE_DIR NAMES blosc.h HINTS "${CMAKE_PREFIX_PATH}/include" DOC "The blosc include directory")
find_library(blosc_LIBRARY NAMES blosc HINTS "${CMAKE_PREFIX_PATH}/lib" DOC "The blosc library")

# If Blosc 1 is not found, try Blosc 2
if(NOT blosc_INCLUDE_DIR OR NOT blosc_LIBRARY)
    message(STATUS "Blosc not found, trying Blosc2...")
    find_path(blosc_INCLUDE_DIR NAMES blosc2.h HINTS "${CMAKE_PREFIX_PATH}/include" DOC "The blosc2 include directory")
    find_library(blosc_LIBRARY NAMES blosc2 HINTS "${CMAKE_PREFIX_PATH}/lib" DOC "The blosc2 library")
endif()

# Handle package standard args
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(blosc REQUIRED_VARS blosc_LIBRARY blosc_INCLUDE_DIR)

# If found, set up targets
if(blosc_FOUND)
    set(blosc_LIBRARIES ${blosc_LIBRARY})
    set(blosc_INCLUDE_DIRS ${blosc_INCLUDE_DIR})
    if(NOT TARGET blosc::blosc)
        add_library(blosc::blosc UNKNOWN IMPORTED)
        set_target_properties(blosc::blosc PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${blosc_INCLUDE_DIR}"
                IMPORTED_LOCATION "${blosc_LIBRARY}")
    endif()
endif()

mark_as_advanced(blosc_INCLUDE_DIR blosc_LIBRARY)