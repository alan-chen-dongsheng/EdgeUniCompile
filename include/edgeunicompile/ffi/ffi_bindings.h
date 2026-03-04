#pragma once

// FFI Bindings for EdgeUniCompile
// This file provides C-compatible exports for calling C++ passes from Python

#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle to a computation graph
typedef void* EUC_GraphHandle;

// Opaque handle to a pass result
typedef void* EUC_PassResultHandle;

// Status codes matching edgeunic::StatusCode
typedef enum {
    EUC_OK = 0,
    EUC_ERROR = 1,
    EUC_INVALID_ARGUMENT = 2,
    EUC_NOT_FOUND = 3,
    EUC_NOT_IMPLEMENTED = 4,
    EUC_INTERNAL = 5,
    EUC_RESOURCE_EXHAUSTED = 6,
} EUC_StatusCode;

// Result of an FFI operation
typedef struct {
    EUC_StatusCode code;
    char* message;  // Caller must free with EUC_FreeString
} EUC_Result;

/**
 * Run the Print Node Names pass on a serialized graph.
 *
 * @param data Pointer to FlatBuffer data.
 * @param size Size of the data in bytes.
 * @param verbose If non-zero, print additional details.
 * @return EUC_Result containing status and optionally modified graph data.
 */
EUC_Result EUC_RunPrintNodeNamesPass(
    const uint8_t* data,
    size_t size,
    int verbose
);

/**
 * Run the Memory Allocation pass on a serialized graph.
 *
 * @param data Pointer to FlatBuffer data.
 * @param size Size of the data in bytes.
 * @param sram_base SRAM base address.
 * @param sram_max_size SRAM maximum size in bytes.
 * @param dram_base DRAM base address.
 * @param dram_max_size DRAM maximum size in bytes.
 * @return EUC_Result containing status and modified graph data.
 */
EUC_Result EUC_RunMemoryAllocationPass(
    const uint8_t* data,
    size_t size,
    uint64_t sram_base,
    uint64_t sram_max_size,
    uint64_t dram_base,
    uint64_t dram_max_size
);

/**
 * Run the Constant Folding pass on a serialized graph.
 *
 * @param data Pointer to FlatBuffer data.
 * @param size Size of the data in bytes.
 * @return EUC_Result containing status and modified graph data.
 */
EUC_Result EUC_RunConstantFoldingPass(
    const uint8_t* data,
    size_t size
);

/**
 * Free a string allocated by the FFI library.
 *
 * @param str String to free.
 */
void EUC_FreeString(char* str);

/**
 * Free graph data allocated by the FFI library.
 *
 * @param data Data to free.
 */
void EUC_FreeGraphData(uint8_t* data);

#ifdef __cplusplus
}
#endif
