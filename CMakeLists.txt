cmake_minimum_required(VERSION 3.20)

# Set CMAKE_CXX_FLAGS before project() to ensure FetchContent packages use libc++
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

# Compiler flags
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -Werror -Wno-unused-parameter)
    add_compile_options(-fno-rtti)  # For MLIR compatibility
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

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

# CMake modules
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

# Find packages
find_package(FlatBuffers)
if(NOT FLATBUFFERS_FOUND)
  message(STATUS "FlatBuffers not found, will build from source")
  include(FetchContent)
  FetchContent_Declare(
    flatbuffers
    GIT_REPOSITORY https://github.com/google/flatbuffers.git
    GIT_TAG v24.3.25
    GIT_SHALLOW ON
  )
  FetchContent_MakeAvailable(flatbuffers)
endif()

# Find or fetch pybind11 for Python bindings (optional)
set(PYTHON_BINDINGS_AVAILABLE OFF)

# Check if we should build Python bindings
option(BUILD_PYTHON_BINDINGS "Build pybind11 Python bindings" OFF)

if(BUILD_PYTHON_BINDINGS)
  find_package(pybind11 CONFIG)
  if(NOT pybind11_FOUND)
    message(STATUS "pybind11 not found, will build from source")
    include(FetchContent)
    FetchContent_Declare(
      pybind11
      GIT_REPOSITORY https://github.com/pybind/pybind11.git
      GIT_TAG v2.11.1
      GIT_SHALLOW ON
    )
    FetchContent_MakeAvailable(pybind11)
  else()
    message(STATUS "pybind11 found: ${pybind11_VERSION}")
  endif()

  # Try to find Python development files
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

  # Check if Python headers actually exist
  if(EXISTS "${Python_INCLUDE_DIRS}/Python.h")
    message(STATUS "Python found: ${Python_VERSION} (headers: ${Python_INCLUDE_DIRS})")
    set(PYTHON_BINDINGS_AVAILABLE ON)
  else()
    message(FATAL_ERROR "Python headers not found at ${Python_INCLUDE_DIRS}")
  endif()
else()
  message(STATUS "Python bindings disabled (set BUILD_PYTHON_BINDINGS=ON to enable)")
endif()

find_package(MLIR)
if(NOT MLIR_FOUND)
  message(STATUS "MLIR not found, will install via Python later")
endif()

# Subdirectories
add_subdirectory(src)
add_subdirectory(tests/cpp)

