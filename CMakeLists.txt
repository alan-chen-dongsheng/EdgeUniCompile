cmake_minimum_required(VERSION 3.20)

# Set CMAKE_CXX_FLAGS before project() to ensure consistent libc++ usage
set(CMAKE_CXX_FLAGS_INIT "-stdlib=libc++" CACHE STRING "" FORCE)
set(CMAKE_EXE_LINKER_FLAGS_INIT "-stdlib=libc++" CACHE STRING "" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_INIT "-stdlib=libc++" CACHE STRING "" FORCE)

project(EdgeUniCompile VERSION 0.1.0 LANGUAGES CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default compiler to Clang
if(NOT CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER "clang++")
endif()

# Compiler flags (global, minus -fno-rtti which breaks googletest)
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -Wno-unused-parameter)
    add_link_options(-stdlib=libc++)
endif()

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

# Output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Include directories for project code
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

# CMake modules
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

# =============================================================================
# Dependencies (all managed via git submodules)
# Run: git submodule update --init --recursive
# =============================================================================

# FlatBuffers
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/third_party/flatbuffers/CMakeLists.txt")
    message(FATAL_ERROR "FlatBuffers submodule not found. Run: git submodule update --init --recursive")
endif()

# Configure FlatBuffers options
set(FLATBUFFERS_BUILD_TESTS OFF CACHE BOOL "Build FlatBuffers tests" FORCE)
set(FLATBUFFERS_BUILD_FLATLIB ON CACHE BOOL "Build flatbuffers library" FORCE)
add_subdirectory(third_party/flatbuffers)

# pybind11 (optional, for Python bindings)
set(PYTHON_BINDINGS_AVAILABLE OFF)
option(BUILD_PYTHON_BINDINGS "Build pybind11 Python bindings" OFF)

if(BUILD_PYTHON_BINDINGS)
    if(NOT EXISTS "${CMAKE_SOURCE_DIR}/third_party/pybind11/CMakeLists.txt")
        message(FATAL_ERROR "pybind11 submodule not found. Run: git submodule update --init --recursive")
    endif()
    add_subdirectory(third_party/pybind11)

    # Python
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

    # Check if Python headers exist
    if(EXISTS "${Python_INCLUDE_DIRS}/Python.h")
        message(STATUS "Python found: ${Python_VERSION} (headers: ${Python_INCLUDE_DIRS})")
        set(PYTHON_BINDINGS_AVAILABLE ON)
    else()
        message(FATAL_ERROR "Python headers not found at ${Python_INCLUDE_DIRS}")
    endif()
else()
    message(STATUS "Python bindings disabled (set BUILD_PYTHON_BINDINGS=ON to enable)")
endif()

# GTest (for testing)
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/third_party/googletest/CMakeLists.txt")
    message(FATAL_ERROR "Googletest submodule not found. Run: git submodule update --init --recursive")
endif()

# Configure googletest options - disable gmock and tests
set(INSTALL_GTEST OFF CACHE BOOL "Install gtest" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "Build gmock" FORCE)
set(gtest_build_tests OFF CACHE BOOL "Build gtest tests" FORCE)

# Build googletest with C++17 and without the strict compiler flags
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)  # Required for some C++ library features
add_subdirectory(third_party/googletest)
# Restore strict settings for our code
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(MLIR)
if(NOT MLIR_FOUND)
    message(STATUS "MLIR not found, will install via Python later")
endif()

# Subdirectories
add_subdirectory(src)
add_subdirectory(tests/cpp)
