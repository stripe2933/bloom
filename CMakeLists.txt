cmake_minimum_required(VERSION 3.28)
project(bloom)

set(CMAKE_CXX_STANDARD 23)

# ----------------
# Enable standard library module.
# ----------------

file(DOWNLOAD https://raw.githubusercontent.com/stripe2933/CppStandardLibraryModule/main/cmake/EnableStandardLibraryModule.cmake
    ${PROJECT_BINARY_DIR}/EnableStandardLibraryModule.cmake
)
include(${PROJECT_BINARY_DIR}/EnableStandardLibraryModule.cmake)

# ----------------
# Install CPM.cmake.
# ----------------

file(DOWNLOAD https://github.com/cpm-cmake/CPM.cmake/releases/latest/download/cpm.cmake
    ${PROJECT_BINARY_DIR}/CPM.cmake
)
include(${PROJECT_BINARY_DIR}/CPM.cmake)

# ----------------
# External dependencies.
# ----------------

set(VKUTIL_VK_VERSION 1001000)
CPMAddPackage("gh:stripe2933/vkutil#main")
CPMAddPackage(
    NAME glm
    URL https://github.com/g-truc/glm/archive/refs/tags/1.0.1.tar.gz
)

find_package(Stb REQUIRED)

# ----------------
# Module configuration for external dependencies.
# ----------------

file(DOWNLOAD https://raw.githubusercontent.com/stripe2933/Cpp20Module/main/glm.cmake
    ${PROJECT_BINARY_DIR}/glm.cmake
)
include(${PROJECT_BINARY_DIR}/glm.cmake)

# ----------------
# Project executable.
# ----------------

add_executable(bloom main.cpp impl.cpp)
target_include_directories(bloom PRIVATE
    ${Stb_INCLUDE_DIR}
)
target_sources(bloom PRIVATE
    FILE_SET CXX_MODULES
    FILES
        interface/pipeline/average_luminance_computer.cppm
        interface/pipeline/bloom_downsample_computer.cppm
        interface/pipeline/bloom_upsample_computer.cppm
        interface/pipeline/diamond_image_computer.cppm
        interface/pipeline/pipeline.cppm
        interface/pipeline/tone_mapping_computer.cppm
        interface/utils/ranges.cppm
        interface/utils/utils.cppm
        interface/app.cppm
        interface/bloom.cppm
)
target_link_libraries(bloom PRIVATE
    glm::module
    vkutil
)

# ----------------
# Shader compilations.
# ----------------

compile_shader(bloom
    FORMAT spv
    SOURCES
        shaders/average_luminance.comp
        shaders/bloom_downsample.comp
        shaders/bloom_upsample.comp
        shaders/diamond_image.comp
        shaders/tone_mapping.comp
)