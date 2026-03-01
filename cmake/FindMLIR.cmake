if(NOT MLIR_FOUND)
  find_path(MLIR_INCLUDE_DIR
    NAMES mlir/IR/MLIRContext.h
    PATHS ${CMAKE_SOURCE_DIR}/third_party/mlir/include
          /usr/local/include
          /usr/include
    PATH_SUFFIXES mlir
  )

  find_library(MLIR_LIBRARY
    NAMES mlir libmlir
    PATHS ${CMAKE_SOURCE_DIR}/third_party/mlir/build/lib
          /usr/local/lib
          /usr/lib
    PATH_SUFFIXES lib
  )

  find_program(MLIR_OPT_EXECUTABLE
    NAMES mlir-opt
    PATHS ${CMAKE_SOURCE_DIR}/third_party/mlir/build/bin
          /usr/local/bin
          /usr/bin
  )

  find_program(MLIR_TRANSLATE_EXECUTABLE
    NAMES mlir-translate
    PATHS ${CMAKE_SOURCE_DIR}/third_party/mlir/build/bin
          /usr/local/bin
          /usr/bin
  )

  if(MLIR_INCLUDE_DIR AND MLIR_LIBRARY)
    set(MLIR_FOUND TRUE)
    set(MLIR_INCLUDE_DIRS ${MLIR_INCLUDE_DIR})
    set(MLIR_LIBRARIES ${MLIR_LIBRARY})
  endif()

  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(MLIR
    REQUIRED_VARS MLIR_INCLUDE_DIR MLIR_LIBRARY
    VERSION_VAR MLIR_VERSION
  )

  if(MLIR_FOUND)
    add_library(mlir::mlir SHARED IMPORTED)
    set_target_properties(mlir::mlir PROPERTIES
      IMPORTED_LOCATION ${MLIR_LIBRARY}
      INTERFACE_INCLUDE_DIRECTORIES ${MLIR_INCLUDE_DIR}
    )
  else()
    message(WARNING "MLIR not found. Will install via Python later.")
  endif()
endif()
