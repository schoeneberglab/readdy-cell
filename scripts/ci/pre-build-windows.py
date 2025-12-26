import configparser
import os
from pathlib import Path
from subprocess import run
import sysconfig

os.chdir(Path(__file__).parent.parent.parent)

native_platform = sysconfig.get_platform()
plat_name = ""
dist_extra_conf = os.getenv("DIST_EXTRA_CONFIG")
if dist_extra_conf is not None:
    distcfg = configparser.ConfigParser()
    distcfg.read(dist_extra_conf)
    plat_name = distcfg.get("build_ext", "plat_name", fallback="")

print("build_ext.plat_name:", plat_name)
print("native platform:    ", native_platform)

if not plat_name:
    plat_name = native_platform

can_run = (plat_name, native_platform) in {
    ("win32", "win32"),
    ("win32", "win-amd64"),
    ("win-amd64", "win-amd64"),
    ("win-arm32", "win-arm32"),
    ("win-arm64", "win-arm64"),
}
cmake_processor = {
    "win32": "x86",
    "win-amd64": "AMD64",
    "win-arm32": "ARM",
    "win-arm64": "ARM64",
}[plat_name]
conan_arch = {
    "win32": "x86",
    "win-amd64": "x86_64",
    "win-arm32": "armv7",
    "win-arm64": "armv8",
}[plat_name]
cpu_flags = {
    "win32": ["/arch:SSE2"],
    "win-amd64": ["/arch:AVX2"],
    "win-arm32": [],
    "win-arm64": [],
}[plat_name]

module_linker_flags = {  # https://github.com/conan-io/conan/issues/17539
    f"CMAKE_MODULE_LINKER_FLAGS{c}_INIT": f"${{CMAKE_SHARED_LINKER_FLAGS{c}_INIT}}"
    for c in ("", "_DEBUG", "_RELEASE", "_RELWITHDEBINFO")
}

native_profile = f"""\
include(default)
[settings]
arch={conan_arch}
build_type=Release
[conf]
tools.build:skip_test=True
tools.cmake.cmaketoolchain:generator=Ninja Multi-Config
tools.build:cflags+={cpu_flags}
tools.build:cxxflags+={cpu_flags}
tools.cmake.cmaketoolchain:extra_variables*={repr(module_linker_flags)}
"""
cross_profile = native_profile
cross_profile += f"""\
tools.build.cross_building:can_run={can_run}
tools.cmake.cmaketoolchain:system_name=Windows
tools.cmake.cmaketoolchain:system_processor={cmake_processor}
"""

cross = not can_run
profile = cross_profile if cross else native_profile
print(profile)
Path("cibw.profile").write_text(profile)

opts = dict(shell=True, check=True)
run("conan install . -pr:h ./cibw.profile --build=missing", **opts)
