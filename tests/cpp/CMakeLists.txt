# Tests CMake file

include(CTest)

# Use manually built GTest
set(GTEST_INCLUDE_DIR "/tmp/googletest/install/include")
set(GTEST_LIBRARY "/tmp/googletest/install/lib/libgtest.a")
set(GTEST_MAIN_LIBRARY "/tmp/googletest/install/lib/libgtest_main.a")
include_directories(${GTEST_INCLUDE_DIR})

# Add test executable
add_executable(euc_test
    core/context_test.cpp
    core/types_test.cpp
    ir/graph_test.cpp
    ir/node_test.cpp
    ir/tensor_test.cpp
    passes/pass_test.cpp
)

# Link libraries
target_link_libraries(euc_test PRIVATE
    /tmp/googletest/install/lib/libgtest.a
    /tmp/googletest/install/lib/libgtest_main.a
    /tmp/googletest/install/lib/libgmock.a
    edgeunic
    /tmp/flatbuffers/install/lib/libflatbuffers.a
    pthread
)

# Include directories
target_include_directories(euc_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Add tests to CTest
include(GoogleTest)
gtest_discover_tests(euc_test)

# Compiler flags
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(euc_test PRIVATE -Wall -Wextra -Wpedantic -Werror -Wno-unused-parameter -fno-rtti -stdlib=libc++)
    target_link_options(euc_test PRIVATE -stdlib=libc++)
endif()

# Set output directory
set_target_properties(euc_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests
)

# Installation
install(TARGETS euc_test
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Add custom test command for convenience
add_custom_target(run_tests
    COMMAND ${CMAKE_BINARY_DIR}/tests/euc_test
    DEPENDS euc_test
    COMMENT "Running tests"
)
