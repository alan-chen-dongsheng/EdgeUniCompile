// pybind11 bindings for EdgeUniCompile
// This module provides Python bindings for C++ passes using pybind11

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "edgeunicompile/ir/graph.h"
#include "edgeunicompile/passes/pass_base.h"
#include "edgeunicompile/passes/pass_manager.h"
#include "edgeunicompile/passes/print_node_names_pass.h"
#include "edgeunicompile/passes/memory_allocation_pass.h"
#include "edgeunicompile/passes/constant_folding_pass.h"
#include "edgeunicompile/codegen/instruction_scheduler.h"

namespace py = pybind11;

PYBIND11_MODULE(edgeunic_cpp, m) {
    m.doc() = "EdgeUniCompile C++ Passes - Python Bindings";

    // Pass Context
    py::class_<edgeunic::PassContext, std::shared_ptr<edgeunic::PassContext>>(m, "PassContext")
        .def(py::init<>())
        .def("set_config", &edgeunic::PassContext::SetConfig)
        .def("get_config", &edgeunic::PassContext::GetConfig)
        .def("increment_counter", &edgeunic::PassContext::IncrementCounter)
        .def("get_counter", &edgeunic::PassContext::GetCounter);

    // Graph
    py::class_<edgeunic::Graph, std::shared_ptr<edgeunic::Graph>>(m, "Graph")
        .def(py::init<const std::string&>())
        .def("get_name", &edgeunic::Graph::GetName)
        .def("get_nodes", &edgeunic::Graph::GetNodes)
        .def("get_tensors", &edgeunic::Graph::GetTensors)
        .def("add_node", &edgeunic::Graph::AddNode)
        .def("add_tensor", &edgeunic::Graph::AddTensor)
        .def("add_input_tensor", &edgeunic::Graph::AddInputTensor)
        .def("add_output_tensor", &edgeunic::Graph::AddOutputTensor);

    // Node
    py::class_<edgeunic::Node, std::shared_ptr<edgeunic::Node>>(m, "Node")
        .def(py::init<>())
        .def("get_name", &edgeunic::Node::GetName)
        .def("get_op_type", &edgeunic::Node::GetOpType)
        .def("get_inputs", &edgeunic::Node::GetInputs)
        .def("get_outputs", &edgeunic::Node::GetOutputs);

    // Status
    py::class_<edgeunic::Status>(m, "Status")
        .def(py::init<>())
        .def("is_ok", &edgeunic::Status::IsOk)
        .def("to_string", &edgeunic::Status::ToString)
        .def_static("ok", &edgeunic::Status::Ok)
        .def_static("error", &edgeunic::Status::Error);

    // Pass Base
    py::class_<edgeunic::PassBase, std::shared_ptr<edgeunic::PassBase>>(m, "PassBase")
        .def("get_name", &edgeunic::PassBase::GetName)
        .def("get_description", &edgeunic::PassBase::GetDescription);

    // Print Node Names Pass
    py::class_<edgeunic::PrintNodeNamesPass, edgeunic::PassBase,
               std::shared_ptr<edgeunic::PrintNodeNamesPass>>(m, "PrintNodeNamesPass")
        .def(py::init<bool>(), py::arg("verbose") = false)
        .def("run", &edgeunic::PrintNodeNamesPass::Run,
             py::arg("graph"), py::arg("context") = nullptr);

    // Memory Allocation Pass
    py::class_<edgeunic::MemoryAllocationPass, edgeunic::PassBase,
               std::shared_ptr<edgeunic::MemoryAllocationPass>>(m, "MemoryAllocationPass")
        .def(py::init<uint64_t, uint64_t, uint64_t, uint64_t>(),
             py::arg("sram_base") = 0,
             py::arg("sram_max_size") = 3 * 1024 * 1024,
             py::arg("dram_base") = 0,
             py::arg("dram_max_size") = 5ULL * 1024 * 1024 * 1024)
        .def("run", &edgeunic::MemoryAllocationPass::Run,
             py::arg("graph"), py::arg("context") = nullptr);

    // Constant Folding Pass
    py::class_<edgeunic::ConstantFoldingPass, edgeunic::PassBase,
               std::shared_ptr<edgeunic::ConstantFoldingPass>>(m, "ConstantFoldingPass")
        .def(py::init<>())
        .def("run", &edgeunic::ConstantFoldingPass::Run,
             py::arg("graph"), py::arg("context") = nullptr);

    // Instruction Scheduler
    py::class_<edgeunic::InstructionScheduler, std::shared_ptr<edgeunic::InstructionScheduler>>(m, "InstructionScheduler")
        .def(py::init<>())
        .def("generate_instructions", &edgeunic::InstructionScheduler::GenerateInstructions,
             py::arg("graph"), py::arg("context") = nullptr)
        .def("schedule_instructions", &edgeunic::InstructionScheduler::ScheduleInstructions)
        .def("get_packets", &edgeunic::InstructionScheduler::GetPackets)
        .def("get_node_order", &edgeunic::InstructionScheduler::GetNodeOrder)
        .def("print_schedule", &edgeunic::InstructionScheduler::PrintSchedule);

    // Instruction Packet
    py::class_<edgeunic::InstructionPacket, std::shared_ptr<edgeunic::InstructionPacket>>(m, "InstructionPacket")
        .def("num_instructions", &edgeunic::InstructionPacket::NumInstructions)
        .def("get_instructions", &edgeunic::InstructionPacket::GetInstructions);

    // Module-level convenience functions
    m.def("run_print_node_names",
          [](std::shared_ptr<edgeunic::Graph> graph, bool verbose = false) {
              edgeunic::PrintNodeNamesPass pass(verbose);
              auto context = std::make_shared<edgeunic::PassContext>();
              return pass.Run(graph, context);
          },
          "Run PrintNodeNamesPass on a graph",
          py::arg("graph"), py::arg("verbose") = false);

    m.def("run_memory_allocation",
          [](std::shared_ptr<edgeunic::Graph> graph,
             uint64_t sram_base = 0,
             uint64_t sram_max_size = 3 * 1024 * 1024,
             uint64_t dram_base = 0,
             uint64_t dram_max_size = 5ULL * 1024 * 1024 * 1024) {
              edgeunic::MemoryAllocationPass pass(sram_base, sram_max_size, dram_base, dram_max_size);
              auto context = std::make_shared<edgeunic::PassContext>();
              return pass.Run(graph, context);
          },
          "Run MemoryAllocationPass on a graph",
          py::arg("graph"),
          py::arg("sram_base") = 0,
          py::arg("sram_max_size") = 3 * 1024 * 1024,
          py::arg("dram_base") = 0,
          py::arg("dram_max_size") = 5ULL * 1024 * 1024 * 1024);

    m.def("run_constant_folding",
          [](std::shared_ptr<edgeunic::Graph> graph) {
              edgeunic::ConstantFoldingPass pass;
              auto context = std::make_shared<edgeunic::PassContext>();
              return pass.Run(graph, context);
          },
          "Run ConstantFoldingPass on a graph",
          py::arg("graph"));

    m.def("generate_instructions",
          [](std::shared_ptr<edgeunic::Graph> graph) {
              edgeunic::InstructionScheduler scheduler;
              auto context = std::make_shared<edgeunic::PassContext>();
              auto status = scheduler.GenerateInstructions(graph, context);
              if (!status.IsOk()) {
                  throw std::runtime_error("Failed to generate instructions: " + status.ToString());
              }
              status = scheduler.ScheduleInstructions();
              if (!status.IsOk()) {
                  throw std::runtime_error("Failed to schedule instructions: " + status.ToString());
              }
              return scheduler;
          },
          "Generate and schedule instructions for a graph",
          py::arg("graph"));
}
