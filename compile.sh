#!/bin/bash

# Configuration
PREFIX=~/miniforge3/envs/readdy-dev
PROJECT_ROOT=$(pwd)
BUILD_TYPE="Debug"
PY3K=1
PY_VER="3.10"
RDY_VER="2.0.13"
RUN_UNIT_TESTS=false
RUN_TEST_SIM=false

BUILD_DIR="build"
CONAN_GEN_DIR="$BUILD_DIR/$BUILD_TYPE/generators"
SITE_PACKAGES_DIR="$PREFIX/lib/python$PY_VER/site-packages"
PYTHON="${PREFIX}/bin/python"

#########################################################
#                                                       #
# CMake configuration flags                             #
#                                                       #
#########################################################

# Use an array for CMake flags for better readability
CMAKE_FLAGS=(
  "-DCMAKE_INSTALL_PREFIX=${PREFIX}"
  "-DCMAKE_PREFIX_PATH=${PREFIX}"
  "-DCMAKE_OSX_SYSROOT=${CONDA_BUILD_SYSROOT}"
  "-DCMAKE_BUILD_TYPE=${BUILD_TYPE}"
  "-DPYTHON_EXECUTABLE=${PYTHON}"
  "-DPYTHON_PREFIX=${PREFIX}"
  "-DREADDY_LOG_CMAKE_CONFIGURATION:BOOL=ON"
  "-DREADDY_CREATE_TEST_TARGET:BOOL=ON"
  "-DREADDY_INSTALL_UNIT_TEST_EXECUTABLE:BOOL=OFF"
  "-DREADDY_VERSION=${PKG_VERSION}"
  "-DREADDY_GENERATE_DOCUMENTATION_TARGET:BOOL=OFF"
  "-DREADDY_BUILD_SHARED_COMBINED:BOOL=OFF"
  "-DSP_DIR=${SITE_PACKAGES_DIR}" # Site-packages directory
  "-DCMAKE_CXX_FLAGS_RELEASE=-03" # C++ optimization flags
  "-DCMAKE_C_FLAGS_RELEASE=-03"   # C optimization flags
#  "-DCMAKE_CXX_FLAGS_RELEASE=-O3 -march=native -ffast-math" # C++ optimization flags
#  "-DCMAKE_C_FLAGS_RELEASE=-O3 -march=native -ffast-math"   # C optimization flags
)

# Note:
# -O3: Highest optimization level
# -march=native: Optimizes code for current CPU architecture
# -ffast-math: Allows the compiler to use non-IEEE-compliant optimizations

# Uncomment to use clang instead of gcc
if [ "$1" = "clang" ]; then
    CMAKE_FLAGS+=("-DCMAKE_C_COMPILER=/usr/bin/clang")
    CMAKE_FLAGS+=("-DCMAKE_CXX_COMPILER=/usr/bin/clang++")
fi

export HDF5_ROOT=${PREFIX}
export PYTHON_INCLUDE_DIR=$("$PREFIX/bin/python" -c "import sysconfig; print(sysconfig.get_path('include'))")

# Attempt to feed the correct Python library to FindPythonLibs
lib_path=""
case "$(uname)" in
  "Darwin")
    if [ "$PY3K" -ne 1 ]; then
      cd "${PREFIX}/lib" || exit
      ln -s "libpython${PY_VER}m.dylib" "libpython${PY_VER}.dylib"
      cd - || exit
    fi
    lib_path="$PREFIX/lib/libpython${PY_VER}.dylib"
    ;;
  "Linux")
    if [ "$PY3K" -eq 1 ]; then
      lib_path="$PREFIX/lib/libpython${PY_VER}m.so"
    else
      lib_path="$PREFIX/lib/libpython${PY_VER}.so"
    fi
    ;;
esac
CMAKE_FLAGS+=("-DPYTHON_LIBRARY:FILEPATH=${lib_path}")

## Set CPU count for Travis CI
#if [ "$TRAVIS" == "true" ]; then
#  CPU_COUNT=2;
#else
#  CPU_COUNT=$(nproc --ignore=1);
#fi
CPU_COUNT=4

# Set up directories (creating containing directories if necessary)
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR" || exit

# Run conan to get dependencies and set up the build environment
conan install "$PROJECT_ROOT" --build=missing

source ./$BUILD_TYPE/generators/conanbuild.sh

# Set compiler paths if not already set in the environment
: "${CC:=/usr/bin/gcc}"
: "${CXX:=/usr/bin/g++}"
CMAKE_FLAGS+=("-DCMAKE_C_COMPILER=${CC}")
CMAKE_FLAGS+=("-DCMAKE_CXX_COMPILER=${CXX}")
CMAKE_FLAGS+=("-DCMAKE_TOOLCHAIN_FILE=${CONAN_GEN_DIR}/conan_toolchain.cmake")

# Call CMake
echo "calling cmake with flags: "
for flag in "${CMAKE_FLAGS[@]}"; do
    echo "   $flag"
done

#cmake --clean-first "$PROJECT_ROOT" "${CMAKE_FLAGS[@]}"
cmake "$PROJECT_ROOT" "${CMAKE_FLAGS[@]}"
#cmake --build . --config "$BUILD_TYPE" --parallel "$CPU_COUNT"
cmake --build . --config "$BUILD_TYPE" --parallel "$CPU_COUNT" --target install

# Run Tests
if $RUN_UNIT_TESTS; then
  export READDY_N_CORES=2
  export READDY_PLUGIN_DIR="${PREFIX}/lib/readdy_plugins"

  err_code=0
  ret_code=0

  echo "Calling C++ Core Unit Tests..."
  ./readdy/test/runUnitTests --durations yes
  err_code=$?
  if [ ${err_code} -ne 0 ]; then
     ret_code=${err_code}
     echo "core unit tests failed with ${ret_code}"
  fi

  # echo "calling c++ integration tests"
  # ./readdy/test/runUnitTests --durations yes [integration]
  # err_code=$?
  # if [ ${err_code} -ne 0 ]; then
  #    ret_code=${err_code}
  #    echo "core unit tests failed with ${ret_code}"
  # fi

  echo "Calling C++ SingleCPU Unit Tests..."
  ./kernels/singlecpu/test/runUnitTests_singlecpu --durations yes
  err_code=$?
  if [ ${err_code} -ne 0 ]; then
     ret_code=${err_code}
     echo "singlecpu unit tests failed with ${ret_code}"
  fi

  echo "Calling C++ CPU Unit Tests..."
  ./kernels/cpu/test/runUnitTests_cpu --durations yes
  err_code=$?
  if [ ${err_code} -ne 0 ]; then
    ret_code=${err_code}
    echo "cpu unit tests failed with ${ret_code}"
  fi

  exit ${ret_code}
fi

# Resetting environment
#source ./$BUILD_TYPE/generators/deactivate_conanbuild.sh

# Test Simulation
#cd ..
#if [ $RUN_TEST_SIM ]; then
#  echo "Running test simulation"
#  python ./rxn_test_1.py
#fi

#if [ $1 ]; then
#  TEST_INDEX=$1
#  echo "Running test simulation"
#  python ./rxn_test_$TEST_INDEX.py
#fi

# Done