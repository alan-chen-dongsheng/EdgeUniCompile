# Source directory CMake file

# Generate FlatBuffer code from schema
set(FLATBUF_SCHEMA ${CMAKE_SOURCE_DIR}/include/edgeunicompile/flatbuf/edgeunicompile.fbs)
set(FLATBUF_GENERATED_HEADER ${CMAKE_BINARY_DIR}/include/edgeunicompile/flatbuf/edgeunicompile_generated.h)

# Find flatc compiler
find_program(FLATC_EXECUTABLE flatc)

if(FLATC_EXECUTABLE)
    # Create directory for generated files
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/include/edgeunicompile/flatbuf)

    # Generate C++ code from schema
    add_custom_command(
        OUTPUT ${FLATBUF_GENERATED_HEADER}
        COMMAND ${FLATC_EXECUTABLE}
        ARGS --cpp --gen-mutable -o ${CMAKE_BINARY_DIR}/include/edgeunicompile/flatbuf ${FLATBUF_SCHEMA}
        DEPENDS ${FLATBUF_SCHEMA}
        COMMENT "Generating FlatBuffer code from schema"
        VERBATIM
    )

    # Generate Python code from schema
    add_custom_command(
        OUTPUT ${CMAKE_SOURCE_DIR}/python/edgeunicompile/flatbuf/edgeunicompile_generated.py
        COMMAND ${FLATC_EXECUTABLE}
        ARGS --python --gen-mutable -o ${CMAKE_SOURCE_DIR}/python/edgeunicompile/flatbuf ${FLATBUF_SCHEMA}
        DEPENDS ${FLATBUF_SCHEMA}
        COMMENT "Generating Python FlatBuffer code from schema"
        VERBATIM
    )

    # Custom target to generate FlatBuffer code
    add_custom_target(generate_flatbuffers DEPENDS ${FLATBUF_GENERATED_HEADER})

    # Add define for FlatBuffer code availability
    add_compile_definitions(FLATBUFFER_CODE_AVAILABLE)
    message(STATUS "FlatBuffer code generation enabled (flatc found: ${FLATC_EXECUTABLE})")
else()
    message(WARNING "flatc compiler not found, FlatBuffer code generation disabled")
endif()

# Main library
add_library(edgeunic SHARED
    core/types.cpp
    core/context.cpp
    ir/graph.cpp
    ir/node.cpp
    ir/tensor.cpp
    passes/pass_base.cpp
    passes/pass_manager.cpp
    passes/constant_folding_pass.cpp
    passes/print_node_names_pass.cpp
    passes/memory_allocation_pass.cpp
    flatbuf/flatbuf_converter.cpp
    ffi/ffi_bindings.cpp
    codegen/instruction.cpp
    codegen/instruction_scheduler.cpp
)

# Add dependency on generated FlatBuffer code
if(FLATC_EXECUTABLE)
    add_dependencies(edgeunic generate_flatbuffers)
    target_include_directories(edgeunic PRIVATE ${CMAKE_BINARY_DIR}/include)
endif()

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
    target_compile_options(edgeunic PRIVATE -Wall -Wextra -Wpedantic -Werror -Wno-unused-parameter -fno-rtti -stdlib=libc++)
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
         ${CMAKE_SOURCE_DIR}/include/edgeunicompile/ir/tensor.h;\
         ${CMAKE_SOURCE_DIR}/include/edgeunicompile/passes/pass_base.h;\
         ${CMAKE_SOURCE_DIR}/include/edgeunicompile/passes/pass_manager.h;\
         ${CMAKE_SOURCE_DIR}/include/edgeunicompile/passes/constant_folding_pass.h;\
         ${CMAKE_SOURCE_DIR}/include/edgeunicompile/passes/print_node_names_pass.h;\
         ${CMAKE_SOURCE_DIR}/include/edgeunicompile/passes/memory_allocation_pass.h;\
         ${CMAKE_SOURCE_DIR}/include/edgeunicompile/flatbuf/flatbuf_converter.h;\
         ${CMAKE_SOURCE_DIR}/include/edgeunicompile/ffi/ffi_bindings.h;\
         ${CMAKE_SOURCE_DIR}/include/edgeunicompile/codegen/instruction.h;\
         ${CMAKE_SOURCE_DIR}/include/edgeunicompile/codegen/instruction_scheduler.h"
)
