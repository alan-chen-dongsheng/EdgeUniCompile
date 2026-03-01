# Source directory CMake file

# Main library
add_library(edgeunic SHARED
    core/types.cpp
    core/context.cpp
    ir/graph.cpp
    ir/node.cpp
    ir/tensor.cpp
)

# Include directories
target_include_directories(edgeunic
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# C++17 standard
target_compile_features(edgeunic PRIVATE cxx_std_17)

# Compiler flags
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(edgeunic PRIVATE -Wall -Wextra -Wpedantic -Werror -Wno-unused-parameter -fno-rtti)
    target_link_options(edgeunic PRIVATE -stdlib=libc++)
endif()

# Link dependencies
if(TARGET flatbuffers)
    target_link_libraries(edgeunic PRIVATE flatbuffers)
elseif(TARGET flatbuffers::flatbuffers)
    target_link_libraries(edgeunic PRIVATE flatbuffers::flatbuffers)
else()
    message(FATAL_ERROR "FlatBuffers library not found")
endif()

# Public headers
set_target_properties(edgeunic PROPERTIES
    PUBLIC_HEADER
        "${CMAKE_SOURCE_DIR}/include/edgeunicompile/core/types.h;\
         ${CMAKE_SOURCE_DIR}/include/edgeunicompile/core/context.h;\
         ${CMAKE_SOURCE_DIR}/include/edgeunicompile/ir/graph.h;\
         ${CMAKE_SOURCE_DIR}/include/edgeunicompile/ir/node.h;\
         ${CMAKE_SOURCE_DIR}/include/edgeunicompile/ir/tensor.h"
)
