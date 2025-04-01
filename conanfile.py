from conan import ConanFile
from conan.tools.cmake import CMakeToolchain, CMake, cmake_layout, CMakeDeps

class ReaDDyRecipe(ConanFile):
    name = "readdy_main"
    version = "0.0"
    package_type = "application"

    # Optional metadata
    license = "MIT"
    author = "Computational Molecular Biology Group, Freie Universität Berlin"
    url = "https://github.com/readdy"
    description = "ReaDDy - Simulation software for particle-based reaction-diffusion systems"
    topics = ("reaction-diffusion", "mesoscale", "simulation", "molecular biology")

    # Binary configuration
    settings = "os", "compiler", "build_type", "arch"

    # Sources
    exports_sources = (
        "CMakeLists.txt",
        "src/*",
        "contrib/*",
        "include/*",
        "wrappers/*",
        "kernels/*",
        "cmake/Modules/*",
        "cmake/sources/*",
        "cmake/sources/kernels/*"
    )

    # Options for the user
    options = {
        "build_python_wrapper": [True, False],
        "build_mpi_kernel": [True, False],
        "install_headers": [True, False],
        "create_test_target": [True, False],
        "generate_documentation_target": [True, False],
        "log_cmake_configuration": [True, False],  # Added option for logging cmake configuration
        "build_shared_combined": [True, False],  # Added option to match `READDY_BUILD_SHARED_COMBINED`
    }

    default_options = {
        "build_python_wrapper": True,
        "build_mpi_kernel": False,
        "install_headers": True,
        "create_test_target": True,
        "generate_documentation_target": True,  # Set to ON
        "log_cmake_configuration": True,  # Set to ON
        "build_shared_combined": False,  # Set to OFF for building separate libs
        "spdlog/*:header_only": True,  # Set default option for spdlog to header_only
        "fmt/*:header_only": True ,     # Set default option for fmt to header_only
    }

    def layout(self):
        cmake_layout(self)

    def requirements(self):
        self.requires("spdlog/1.10.0")
        self.requires("nlohmann_json/3.10.3")
        self.requires("fmt/8.1.1")
        # self.requires("fmt/11.0.2")
        # self.requires("c-blosc/1.21.0")
        self.requires("c-blosc2/2.17.0")
        self.requires("hdf5/1.14.5")
        # self.requires("zlib/1.2.11")
        # self.test_requires("nose/1.3.7")

    def configure(self):
        # Apply the header_only option to spdlog and fmt only
        self.options["spdlog/*"].header_only = True
        self.options["fmt/*"].header_only = True

    def generate(self):
        deps = CMakeDeps(self)
        deps.generate()
        tc = CMakeToolchain(self)
        # Explicitly set the architecture to x86_64
        # tc.variables["CMAKE_SYSTEM_PROCESSOR"] = "x86_64"
        tc.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()

    def package_info(self):
        self.cpp_info.libs = ["readdy"]
