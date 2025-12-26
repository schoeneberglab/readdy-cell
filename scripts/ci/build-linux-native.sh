#!/usr/bin/env bash
cd "$( dirname "${BASH_SOURCE[0]}" )"/../..
set -ex

# Package and output directories
pkg_dir="${1:-.}"
out_dir="${2:-dist}"
install_stubs_dir="$3"

# Initial cleanup
rm -rf "$pkg_dir"/build/{generators,CMakeCache.txt}

# Create a Conan profile
python_profile="$PWD/native-conan-python.local.profile"
cat << EOF > "$python_profile"
include(default)
include($PWD/scripts/ci/profiles/linux.profile)
[conf]
tools.build:skip_test=true
EOF

# Install dependencies using Conan
conan install "$pkg_dir" --build=missing -pr "$python_profile"

# Create a py-build-cmake config file
pbc_config="$PWD/native-py-build-cmake.local.pbc"
cat << EOF > "$pbc_config"
cmake.options.CMAKE_C_COMPILER_LAUNCHER=sccache
cmake.options.CMAKE_CXX_COMPILER_LAUNCHER=sccache
cmake.build_args+=["--verbose"]
EOF

# Build the Python package
python3 -m build -w "$pkg_dir" -o "$out_dir" -C local="$pbc_config"

# Install the Python stubs
if [ -n "$install_stubs_dir" ]; then
    cd "$pkg_dir"
    # We install the Python modules and stubs into the given directory
    py-build-cmake --local="$pbc_config" \
        configure
    py-build-cmake --local="$pbc_config" \
        install --component python_modules -- --prefix "$install_stubs_dir"
    py-build-cmake --local="$pbc_config" \
        install --component python_stubs -- --prefix "$install_stubs_dir"
    # Then we remove the binary Python modules (sdist is source only)
    while IFS= read -r f || [ -n "$f" ]; do rm -f "$f"
    done < build/install_manifest_python_modules.txt
fi
