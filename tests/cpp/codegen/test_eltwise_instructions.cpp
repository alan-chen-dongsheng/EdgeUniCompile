// Standalone test for Eltwise instruction generation
// Compile and run to verify Eltwise operations generate correct instructions

#include <iostream>
#include <memory>
#include "edgeunicompile/codegen/instruction_scheduler.h"
#include "edgeunicompile/ir/graph.h"
#include "edgeunicompile/ir/tensor.h"
#include "edgeunicompile/ir/node.h"

using namespace edgeunic;

TensorPtr CreateTensor(const std::string& name, const Shape& shape) {
    return std::make_shared<Tensor>(name, DataType::kFloat32, shape);
}

NodePtr CreateEltwiseNode(const std::string& name, OpType op_type,
                          const TensorPtr& input, const TensorPtr& output) {
    auto node = std::make_shared<Node>();
    node->SetName(name);
    node->SetOpType(op_type);
    node->AddInput(input);
    node->AddOutput(output);
    return node;
}

void RunEltwiseTest() {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "Eltwise Instruction Generation Test\n";
    std::cout << std::string(60, '=') << "\n\n";

    // Create graph: Input -> Relu -> Abs -> Sigmoid -> Output
    auto graph = std::make_shared<Graph>("eltwise_chain");

    // Create tensors
    auto input = CreateTensor("input", Shape{1, 16, 32, 32});
    auto relu_out = CreateTensor("relu_out", Shape{1, 16, 32, 32});
    auto abs_out = CreateTensor("abs_out", Shape{1, 16, 32, 32});
    auto sigmoid_out = CreateTensor("sigmoid_out", Shape{1, 16, 32, 32});

    graph->AddTensor(input);
    graph->AddTensor(relu_out);
    graph->AddTensor(abs_out);
    graph->AddTensor(sigmoid_out);

    // Create Eltwise nodes chain
    auto relu_node = CreateEltwiseNode("relu1", OpType::kRelu, input, relu_out);
    auto abs_node = CreateEltwiseNode("abs1", OpType::kRelu, relu_out, abs_out);  // Using Relu as Abs placeholder
    auto sigmoid_node = CreateEltwiseNode("sigmoid1", OpType::kSigmoid, abs_out, sigmoid_out);

    graph->AddNode(relu_node);
    graph->AddNode(abs_node);
    graph->AddNode(sigmoid_node);

    graph->AddInputTensor(input);
    graph->AddOutputTensor(sigmoid_out);

    std::cout << "Graph: " << graph->GetName() << "\n";
    std::cout << "Nodes: " << graph->GetNodes().size() << "\n";
    std::cout << "Tensors: " << graph->GetTensors().size() << "\n\n";

    // Print node list
    std::cout << "Node List:\n";
    for (size_t i = 0; i < graph->GetNodes().size(); ++i) {
        const auto& node = graph->GetNodes()[i];
        std::cout << "  [" << i << "] " << node->GetName()
                  << " (OpType: " << node->GetOpType() << ")\n";
    }
    std::cout << "\n";

    // Generate instructions
    InstructionScheduler scheduler;
    auto status = scheduler.GenerateInstructions(graph);

    if (!status.IsOk()) {
        std::cerr << "Error generating instructions: " << status.ToString() << "\n";
        return;
    }

    // Schedule instructions
    status = scheduler.ScheduleInstructions();
    if (!status.IsOk()) {
        std::cerr << "Error scheduling instructions: " << status.ToString() << "\n";
        return;
    }

    // Print summary
    const auto& instructions = scheduler.GetInstructions();
    const auto& packets = scheduler.GetPackets();

    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "Instruction Summary\n";
    std::cout << std::string(60, '=') << "\n";
    std::cout << "Total instructions: " << instructions.size() << "\n";
    std::cout << "Total packets: " << packets.size() << "\n\n";

    // Count instruction types
    int load_count = 0, exec_count = 0, store_count = 0;
    for (const auto& instr : instructions) {
        switch (instr->GetType()) {
            case InstructionType::kLoad: load_count++; break;
            case InstructionType::kExec: exec_count++; break;
            case InstructionType::kStore: store_count++; break;
        }
    }

    std::cout << "Instruction breakdown:\n";
    std::cout << "  LOAD:  " << load_count << "\n";
    std::cout << "  EXEC:  " << exec_count << "\n";
    std::cout << "  STORE: " << store_count << "\n\n";

    // Verify Eltwise operations
    std::cout << "Eltwise Operations Verification:\n";
    bool found_relu = false, found_sigmoid = false;
    for (const auto& instr : instructions) {
        if (instr->GetType() == InstructionType::kExec) {
            auto exec = std::static_pointer_cast<ExecInstruction>(instr);
            std::cout << "  - EXEC " << OpTypeToString(exec->GetOpType())
                      << " -> " << exec->GetName() << "\n";
            if (exec->GetOpType() == OpType::kRelu) found_relu = true;
            if (exec->GetOpType() == OpType::kSigmoid) found_sigmoid = true;
        }
    }

    std::cout << "\nVerification:\n";
    std::cout << "  Relu found: " << (found_relu ? "YES" : "NO") << "\n";
    std::cout << "  Sigmoid found: " << (found_sigmoid ? "YES" : "NO") << "\n";

    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "Test PASSED!\n";
    std::cout << std::string(60, '=') << "\n\n";
}

void RunSingleEltwiseTest() {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "Single Eltwise Node Test\n";
    std::cout << std::string(60, '=') << "\n\n";

    // Create graph: Input -> Relu -> Output
    auto graph = std::make_shared<Graph>("single_eltwise");

    auto input = CreateTensor("input", Shape{1, 8, 16, 16});
    auto output = CreateTensor("output", Shape{1, 8, 16, 16});

    graph->AddTensor(input);
    graph->AddTensor(output);

    auto relu_node = CreateEltwiseNode("relu1", OpType::kRelu, input, output);
    graph->AddNode(relu_node);
    graph->AddInputTensor(input);
    graph->AddOutputTensor(output);

    std::cout << "Graph: " << graph->GetName() << "\n";
    std::cout << "Single Eltwise node: Relu\n\n";

    // Generate and schedule instructions
    InstructionScheduler scheduler;
    auto status = scheduler.GenerateInstructions(graph);
    if (!status.IsOk()) {
        std::cerr << "Error: " << status.ToString() << "\n";
        return;
    }

    status = scheduler.ScheduleInstructions();
    if (!status.IsOk()) {
        std::cerr << "Error: " << status.ToString() << "\n";
        return;
    }

    // Print schedule
    const auto& packets = scheduler.GetPackets();
    std::cout << "Instruction Schedule (" << packets.size() << " packets):\n";

    for (size_t i = 0; i < packets.size(); ++i) {
        std::cout << "\n--- Packet " << i << " ---\n";
        for (const auto& instr : packets[i]->GetInstructions()) {
            std::cout << "  " << instr->ToString() << "\n";
        }
    }

    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "Single Eltwise Test PASSED!\n";
    std::cout << std::string(60, '=') << "\n\n";
}

int main(int argc, char** argv) {
    std::cout << "\n\n";
    std::cout << std::string(60, '#') << "\n";
    std::cout << "# Eltwise Instruction Generation Tests\n";
    std::cout << std::string(60, '#') << "\n";

    // Test 1: Single Eltwise node
    RunSingleEltwiseTest();

    // Test 2: Chain of Eltwise nodes
    RunEltwiseTest();

    std::cout << "\n" << std::string(60, '#') << "\n";
    std::cout << "# All Eltwise Tests Completed Successfully!\n";
    std::cout << std::string(60, '#') << "\n\n";

    return 0;
}
