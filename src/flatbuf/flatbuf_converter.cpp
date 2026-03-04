#include "edgeunicompile/flatbuf/flatbuf_converter.h"
#include <fstream>
#include <cstring>

// Include generated FlatBuffer header (generated from schema)
// To generate: flatc -c --gen-mutable include/edgeunicompile/flatbuf/edgeunicompile.fbs
#ifdef FLATBUFFER_CODE_AVAILABLE
#include "edgeunicompile/flatbuf/edgeunicompile_generated.h"
#endif

namespace edgeunic {
namespace flatbuf {

std::vector<uint8_t> FlatBufferConverter::Serialize(const GraphPtr& graph) {
#ifndef FLATBUFFER_CODE_AVAILABLE
    // Fallback: return empty vector if FlatBuffer code is not available
    return {};
#else
    if (!graph) {
        return {};
    }

    flatbuffers::FlatBufferBuilder builder(1024);

    // Serialize tensors
    std::vector<flatbuffers::Offset<edgeunicompile::Tensor>> tensor_offsets;
    std::unordered_map<TensorPtr, uint32_t> tensor_id_map;

    for (const auto& tensor : graph->GetTensors()) {
        uint32_t tensor_id = static_cast<uint32_t>(tensor_offsets.size());
        tensor_id_map[tensor] = tensor_id;

        // Create shape
        std::vector<int64_t> dims;
        for (const auto& dim : tensor->GetShape().dims) {
            dims.push_back(dim);
        }
        auto shape_offset = edgeunicompile::CreateShapeDirect(builder, &dims);

        // Create data vector
        flatbuffers::Offset<flatbuffers::Vector<uint8_t>> data_offset;
        const auto& data = tensor->GetData();
        if (!data.empty()) {
            data_offset = builder.CreateVector(data);
        }

        // Create name string
        auto name_offset = builder.CreateString(tensor->GetName());

        // Create tensor
        auto tensor_offset = edgeunicompile::CreateTensor(
            builder,
            tensor_id,
            name_offset,
            static_cast<edgeunicompile::DataType>(tensor->GetDataType()),
            shape_offset,
            data_offset,
            tensor->IsConstant()
        );
        tensor_offsets.push_back(tensor_offset);
    }

    // Serialize nodes
    std::vector<flatbuffers::Offset<edgeunicompile::Node>> node_offsets;

    for (const auto& node : graph->GetNodes()) {
        // Create input tensor IDs
        std::vector<uint32_t> input_ids;
        for (const auto& input : node->GetInputs()) {
            auto it = tensor_id_map.find(input);
            if (it != tensor_id_map.end()) {
                input_ids.push_back(it->second);
            }
        }

        // Create output tensor IDs
        std::vector<uint32_t> output_ids;
        for (const auto& output : node->GetOutputs()) {
            auto it = tensor_id_map.find(output);
            if (it != tensor_id_map.end()) {
                output_ids.push_back(it->second);
            }
        }

        auto inputs_offset = builder.CreateVector(input_ids);
        auto outputs_offset = builder.CreateVector(output_ids);
        auto name_offset = builder.CreateString(node->GetName());

        // Map OpType to FlatBuffer OpType
        edgeunicompile::OpType fb_op_type = edgeunicompile::OpType::OpType_Unknown;
        // TODO: Add proper OpType mapping

        auto node_offset = edgeunicompile::CreateNode(
            builder,
            name_offset,
            fb_op_type,
            inputs_offset,
            outputs_offset
        );
        node_offsets.push_back(node_offset);
    }

    // Create input/output tensor ID lists
    std::vector<uint32_t> input_tensor_ids;
    for (const auto& tensor : graph->GetInputTensors()) {
        auto it = tensor_id_map.find(tensor);
        if (it != tensor_id_map.end()) {
            input_tensor_ids.push_back(it->second);
        }
    }

    std::vector<uint32_t> output_tensor_ids;
    for (const auto& tensor : graph->GetOutputTensors()) {
        auto it = tensor_id_map.find(tensor);
        if (it != tensor_id_map.end()) {
            output_tensor_ids.push_back(it->second);
        }
    }

    auto input_ids_offset = builder.CreateVector(input_tensor_ids);
    auto output_ids_offset = builder.CreateVector(output_tensor_ids);
    auto graph_name_offset = builder.CreateString(graph->GetName());
    auto nodes_offset = builder.CreateVector(node_offsets);
    auto tensors_offset = builder.CreateVector(tensor_offsets);

    // Create the Graph table
    auto graph_offset = edgeunicompile::CreateGraph(
        builder,
        builder.CreateString("0.1.0"),
        graph_name_offset,
        nodes_offset,
        tensors_offset,
        input_ids_offset,
        output_ids_offset
    );

    // Finish the buffer
    builder.Finish(graph_offset);

    // Copy to output vector
    const uint8_t* buffer = builder.GetBufferPointer();
    size_t size = builder.GetSize();

    return std::vector<uint8_t>(buffer, buffer + size);
#endif
}

GraphPtr FlatBufferConverter::Deserialize(const uint8_t* data, size_t size) {
#ifndef FLATBUFFER_CODE_AVAILABLE
    return nullptr;
#else
    if (!data || size == 0) {
        return nullptr;
    }

    // Verify the FlatBuffer
    flatbuffers::Verifier verifier(data, size);
    if (!edgeunicompile::VerifyGraphBuffer(verifier)) {
        return nullptr;
    }

    // Get root as Graph
    const auto* fb_graph = edgeunicompile::GetGraph(data);
    if (!fb_graph) {
        return nullptr;
    }

    auto graph = std::make_shared<Graph>(fb_graph->name()->str());

    // Deserialize tensors
    std::unordered_map<uint32_t, TensorPtr> id_to_tensor;

    for (uint32_t i = 0; i < fb_graph->tensors()->size(); ++i) {
        const auto* fb_tensor = fb_graph->tensors()->Get(i);

        // Get shape
        Shape shape;
        if (fb_tensor->shape() && fb_tensor->shape()->dims()) {
            for (uint32_t j = 0; j < fb_tensor->shape()->dims()->size(); ++j) {
                shape.dims.push_back(fb_tensor->shape()->dims()->Get(j));
            }
        }

        // Create tensor
        auto tensor = std::make_shared<Tensor>(
            fb_tensor->name()->str(),
            static_cast<DataType>(fb_tensor->dtype()),
            shape
        );

        // Get data
        if (fb_tensor->data() && fb_tensor->data()->size() > 0) {
            std::vector<uint8_t> data(fb_tensor->data()->begin(),
                                       fb_tensor->data()->end());
            tensor->SetData(data);
            tensor->SetIsConstant(true);
        }

        graph->AddTensor(tensor);
        id_to_tensor[i] = tensor;
    }

    // Deserialize nodes
    for (uint32_t i = 0; i < fb_graph->nodes()->size(); ++i) {
        const auto* fb_node = fb_graph->nodes()->Get(i);

        // Map FlatBuffer OpType to internal OpType
        OpType op_type = OpType::kUnknown;
        // TODO: Add proper OpType mapping

        auto node = std::make_shared<Node>(fb_node->name()->str(), op_type);

        // Add inputs
        if (fb_node->inputs()) {
            for (uint32_t id : *fb_node->inputs()) {
                auto it = id_to_tensor.find(id);
                if (it != id_to_tensor.end()) {
                    node->AddInput(it->second);
                }
            }
        }

        // Add outputs
        if (fb_node->outputs()) {
            for (uint32_t id : *fb_node->outputs()) {
                auto it = id_to_tensor.find(id);
                if (it != id_to_tensor.end()) {
                    node->AddOutput(it->second);
                }
            }
        }

        graph->AddNode(node);
    }

    // Add input tensors
    if (fb_graph->input_tensor_ids()) {
        for (uint32_t id : *fb_graph->input_tensor_ids()) {
            auto it = id_to_tensor.find(id);
            if (it != id_to_tensor.end()) {
                graph->AddInputTensor(it->second);
            }
        }
    }

    // Add output tensors
    if (fb_graph->output_tensor_ids()) {
        for (uint32_t id : *fb_graph->output_tensor_ids()) {
            auto it = id_to_tensor.find(id);
            if (it != id_to_tensor.end()) {
                graph->AddOutputTensor(it->second);
            }
        }
    }

    return graph;
#endif
}

Status FlatBufferConverter::SaveToFile(const GraphPtr& graph, const std::string& filename) {
    if (!graph) {
        return Status::InvalidArgument("Graph cannot be null");
    }

    auto data = Serialize(graph);
    if (data.empty()) {
        return Status::Internal("Failed to serialize graph or FlatBuffer code not available");
    }

    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return Status::NotFound("Failed to open file: " + filename);
    }

    file.write(reinterpret_cast<const char*>(data.data()), data.size());
    file.close();

    if (file.fail()) {
        return Status::Internal("Failed to write to file: " + filename);
    }

    return Status::Ok();
}

Status FlatBufferConverter::LoadFromFile(const std::string& filename, GraphPtr& graph) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        return Status::NotFound("Failed to open file: " + filename);
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> buffer(static_cast<size_t>(size));
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        return Status::Internal("Failed to read file: " + filename);
    }

    graph = Deserialize(buffer.data(), buffer.size());
    if (!graph) {
        return Status::Internal("Failed to deserialize graph");
    }

    return Status::Ok();
}

}  // namespace flatbuf
}  // namespace edgeunic
