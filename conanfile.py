from conan import ConanFile
from conan.tools.cmake import CMake, cmake_layout, CMakeToolchain, CMakeDeps


class Recipe(ConanFile):
    name = "readdy"
    version = "0.0"

    # Optional metadata
    url = "https://github.com/readdy"
    description = "ReaDDy - Simulation software for particle-based reaction-diffusion systems"
    topics = ("reaction-diffusion", "mesoscale", "simulation", "molecular biology", "iPRD")

    # Binary configuration
    settings = "os", "compiler", "build_type", "arch"


    exports_sources = (
        "CMakeLists.txt",
        "src/*",
        "include/*",
        "wrappers/*",
        "kernels/*",
        "cmake/Modules/*",
        "cmake/sources/*",
        "cmake/sources/kernels/*",
    )

    default_options = {
        "spdlog/*:header_only": True,  # Set default option for spdlog to header_only
        "fmt/*:header_only": True ,     # Set default option for fmt to header_only
    }

    def layout(self):
        cmake_layout(self)

    def requirements(self):
        self.requires("pybind11/2.13.6")
        self.requires("catch2/3.8.0")
        self.requires("nlohmann_json/3.11.3")
        self.requires("zlib/1.3.1")
        self.requires("hdf5/1.14.3", options={"hl": True, "shared": True})
        self.requires("fmt/11.0.2")
        self.requires("spdlog/1.15.0")

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()

    def generate(self):
        tc = CMakeToolchain(self)
        tc.generate()
        deps = CMakeDeps(self)
        deps.generate()

