#pragma once

#include <memory>
#include <string>
#include <vector>
#include <cstdint>
#include "edgeunicompile/core/types.h"
#include "edgeunicompile/ir/graph.h"

// Forward declare flatbuffers types
namespace flatbuffers {
    class FlatBufferBuilder;
    class Verifier;
}

namespace edgeunic {
namespace flatbuf {

/**
 * FlatBuffer converter for EdgeUniCompile Graph.
 *
 * This class provides functionality to:
 * 1. Serialize EdgeUniCompile Graph to FlatBuffer
 * 2. Deserialize FlatBuffer to EdgeUniCompile Graph
 *
 * The FlatBuffer schema is located at:
 *   include/edgeunicompile/flatbuf/edgeunicompile.fbs
 *
 * To generate C++ code from the schema:
 *   flatc -c --gen-mutable include/edgeunicompile/flatbuf/edgeunicompile.fbs
 */
class FlatBufferConverter {
public:
    FlatBufferConverter() = default;
    ~FlatBufferConverter() = default;

    /**
     * Serialize a graph to FlatBuffer format.
     *
     * @param graph The graph to serialize.
     * @return Serialized FlatBuffer bytes.
     */
    static std::vector<uint8_t> Serialize(const GraphPtr& graph);

    /**
     * Deserialize FlatBuffer to a graph.
     *
     * @param data Pointer to FlatBuffer data.
     * @param size Size of the data in bytes.
     * @return Deserialized graph.
     */
    static GraphPtr Deserialize(const uint8_t* data, size_t size);

    /**
     * Save a graph to a FlatBuffer file.
     *
     * @param graph The graph to save.
     * @param filename Output file path.
     * @return Status indicating success or failure.
     */
    static Status SaveToFile(const GraphPtr& graph, const std::string& filename);

    /**
     * Load a graph from a FlatBuffer file.
     *
     * @param filename Input file path.
     * @param graph Output graph.
     * @return Status indicating success or failure.
     */
    static Status LoadFromFile(const std::string& filename, GraphPtr& graph);

private:
    // Internal helper functions - implemented in cpp file
};

using FlatBufferConverterPtr = std::shared_ptr<FlatBufferConverter>;

}  // namespace flatbuf
}  // namespace edgeunic
