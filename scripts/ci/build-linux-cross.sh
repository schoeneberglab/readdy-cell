#!/usr/bin/env bash
cd "$( dirname "${BASH_SOURCE[0]}" )"/../..
set -ex

# Select Python version
build_python_version="$(python3 --version | cut -d' ' -f2)"
python_version="${1:-${build_python_version}}"
python_majmin="$(echo "$python_version" | cut -d'.' -f1,2)"
python_majmin_nodot="${python_majmin//./}"

# Select architecture
triple="${2:-x86_64-bionic-linux-gnu}"
case "$triple" in
    x86_64-bionic-*) plat_tag=manylinux_2_27_x86_64 ;;
    aarch64-rpi3-*) plat_tag=manylinux_2_27_aarch64 ;;
    armv7-neon-*) plat_tag=manylinux_2_27_armv7l ;;
    armv6-rpi-*) plat_tag=linux_armv6l ;;
    *) echo "Unknown platform ${triple}"; exit 1 ;;
esac

# Package and output directories
pkg_dir="${3:-.}"
out_dir="${4:-dist}"

# Initial cleanup
rm -rf "$pkg_dir"/build/{generators,CMakeCache.txt}

# Create Conan profile to inject the appropriate Python development files
python_profile="$PWD/conan-python.cross.profile"
cat << EOF > "$python_profile"
include($PWD/scripts/ci/profiles/$triple.profile)
[conf]
tools.build:skip_test=true
[options]
&:with_conan_python=True
[replace_requires]
tttapa-python-dev/*: tttapa-python-dev/[~$python_majmin]
EOF

# Install dependencies using Conan
conan install "$pkg_dir" --build=missing -pr "$python_profile"

# Create a py-build-cmake configuration file for cross-compilation
pbc_config="$PWD/$triple.py-build-cmake.cross.pbc"
cat << EOF > "$pbc_config"
os=linux
implementation=cp
version="$python_majmin_nodot"
abi="cp$python_majmin_nodot"
arch="$plat_tag"
cmake.options.CMAKE_C_COMPILER_LAUNCHER=sccache
cmake.options.CMAKE_CXX_COMPILER_LAUNCHER=sccache
cmake.build_args+=["--verbose"]
EOF

# Build the Python packages
python3 -m build -w "$pkg_dir" -o "$out_dir" -C cross="$pbc_config"
