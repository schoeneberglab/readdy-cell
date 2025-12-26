#!/bin/bash

# Configuration
PREFIX="${CONDA_PREFIX}"
PROJECT_ROOT=$(pwd)
BUILD_TYPE="Release"
PY3K=1
PY_VER="3.11"
RDY_VER="3.0.1"
RUN_UNIT_TESTS=true
RUN_TEST_SIM=false
CPU_COUNT=6
BUILD_DIR="build"
CONAN_GEN_DIR="$BUILD_DIR/$BUILD_TYPE/generators"
SITE_PACKAGES_DIR="$PREFIX/lib/python$PY_VER/site-packages"
PYTHON="${PREFIX}/bin/python"

# Command-line arguments
for arg in "$@"; do
  case $arg in
    --run-tests)
      RUN_UNIT_TESTS=true
      shift
      ;;
  esac
done

CMAKE_FLAGS=(
  "-DCMAKE_INSTALL_PREFIX=${PREFIX}"
  "-DCMAKE_PREFIX_PATH=${PREFIX}"
  "-DCMAKE_OSX_SYSROOT=${CONDA_BUILD_SYSROOT}"
  "-DCMAKE_BUILD_TYPE=${BUILD_TYPE}"
  "-DREADDY_LOG_CMAKE_CONFIGURATION:BOOL=ON"
  "-DREADDY_CREATE_TEST_TARGET:BOOL=ON"
  "-DREADDY_INSTALL_UNIT_TEST_EXECUTABLE:BOOL=OFF"
  "-DREADDY_VERSION=${PKG_VERSION}"
  "-DREADDY_GENERATE_DOCUMENTATION_TARGET:BOOL=OFF"
  "-DSP_DIR=${SITE_PACKAGES_DIR}"
  "-DCMAKE_CXX_FLAGS_RELEASE=-O3"
  "-DCMAKE_C_FLAGS_RELEASE=-O3"
)

if [ "$1" = "clang" ]; then
   CMAKE_FLAGS+=("-DCMAKE_C_COMPILER=/usr/bin/clang")
   CMAKE_FLAGS+=("-DCMAKE_CXX_COMPILER=/usr/bin/clang++")
fi

export HDF5_ROOT="${PREFIX}"
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

# Set up directories (creating containing directories if necessary)
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR" || exit

# Print all environment variables, flags, and paths for debugging
echo "========================="
echo "Environment Variables:"
env
echo ""
echo "Variables:"
echo "   PREFIX: $PREFIX"
echo "   PROJECT_ROOT: $PROJECT_ROOT"
echo "   BUILD_DIR: $BUILD_DIR"
echo "   BUILD_TYPE: $BUILD_TYPE"
echo "   CONAN_GEN_DIR: $CONAN_GEN_DIR"
echo "   SITE_PACKAGES_DIR: $SITE_PACKAGES_DIR"
echo "   PYTHON: $PYTHON"
echo "   PYTHON_INCLUDE_DIR: $PYTHON_INCLUDE_DIR"
echo "   PYTHON_LIBRARY: $lib_path"
echo "   HDF5_ROOT: $HDF5_ROOT"
echo "   HDF5_PLUGIN_PATH: $HDF5_PLUGIN_PATH"
echo "========================="
echo "========================="
echo "CMake Flags:"
for flag in "${CMAKE_FLAGS[@]}"; do
    echo "   $flag"
done

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

cmake "$PROJECT_ROOT" "${CMAKE_FLAGS[@]}"
cmake --build . --config "$BUILD_TYPE" --parallel "$CPU_COUNT" --target install
 
# Run Tests
if $RUN_UNIT_TESTS; then
  export READDY_N_CORES=2
  export READDY_PLUGIN_DIR="${PREFIX}/lib/readdy_plugins"

  err_code=0
  ret_code=0

  echo "Calling C++ Core Unit Tests..."
  ./tests/core/runUnitTests --durations yes
  err_code=$?
  if [ ${err_code} -ne 0 ]; then
     ret_code=${err_code}
     echo "core unit tests failed with ${ret_code}"
  fi

  echo "Calling C++ SingleCPU Unit Tests..."
./tests/kernels/singlecpu/runUnitTests_singlecpu --durations yes
  err_code=$?
  if [ ${err_code} -ne 0 ]; then
     ret_code=${err_code}
     echo "singlecpu unit tests failed with ${ret_code}"
  fi

  echo "Calling C++ CPU Unit Tests..."
./tests/kernels/cpu/runUnitTests_cpu --durations yes
  err_code=$?
  if [ ${err_code} -ne 0 ]; then
    ret_code=${err_code}
    echo "cpu unit tests failed with ${ret_code}"
  fi

  exit ${ret_code}
fi
