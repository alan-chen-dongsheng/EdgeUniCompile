cmake_minimum_required(VERSION 3.20)
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
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -stdlib=libc++")
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

find_package(MLIR)
if(NOT MLIR_FOUND)
  message(STATUS "MLIR not found, will install via Python later")
endif()

# Subdirectories
add_subdirectory(src)
add_subdirectory(tests/cpp)

