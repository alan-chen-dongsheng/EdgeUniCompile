if(NOT FLATBUFFERS_FOUND)
  find_path(FLATBUFFERS_INCLUDE_DIR
    NAMES flatbuffers/flatbuffers.h
    PATHS ${CMAKE_SOURCE_DIR}/third_party/flatbuffers/include
          /usr/local/include
          /usr/include
    PATH_SUFFIXES flatbuffers
  )

  find_library(FLATBUFFERS_LIBRARY
    NAMES flatbuffers libflatbuffers
    PATHS ${CMAKE_SOURCE_DIR}/third_party/flatbuffers/build
          /usr/local/lib
          /usr/lib
    PATH_SUFFIXES lib
  )

  find_program(FLATC_EXECUTABLE
    NAMES flatc
    PATHS ${CMAKE_SOURCE_DIR}/third_party/flatbuffers/build
          /usr/local/bin
          /usr/bin
  )

  if(FLATBUFFERS_INCLUDE_DIR AND FLATBUFFERS_LIBRARY)
    set(FLATBUFFERS_FOUND TRUE)
    set(FLATBUFFERS_INCLUDE_DIRS ${FLATBUFFERS_INCLUDE_DIR})
    set(FLATBUFFERS_LIBRARIES ${FLATBUFFERS_LIBRARY})
  endif()

  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(FlatBuffers
    REQUIRED_VARS FLATBUFFERS_INCLUDE_DIR FLATBUFFERS_LIBRARY
    VERSION_VAR FLATBUFFERS_VERSION
  )

  if(FLATBUFFERS_FOUND)
    add_library(flatbuffers::flatbuffers SHARED IMPORTED)
    set_target_properties(flatbuffers::flatbuffers PROPERTIES
      IMPORTED_LOCATION ${FLATBUFFERS_LIBRARY}
      INTERFACE_INCLUDE_DIRECTORIES ${FLATBUFFERS_INCLUDE_DIR}
    )
  else()
    message(WARNING "FlatBuffers not found. Downloading from GitHub...")
    include(FetchContent)
    FetchContent_Declare(
      flatbuffers
      GIT_REPOSITORY https://github.com/google/flatbuffers.git
      GIT_TAG v24.3.25
      GIT_SHALLOW ON
    )
    FetchContent_MakeAvailable(flatbuffers)
  endif()
endif()
