#include <torch/extension.h>

#include "parallel_controllers.h"

namespace py = pybind11;
using rlpx4controller_torch::ParallelAttiControl;
using rlpx4controller_torch::ParallelPosControl;
using rlpx4controller_torch::ParallelRateControl;
using rlpx4controller_torch::ParallelVelControl;

PYBIND11_MODULE(_parallel_control, module) {
    module.doc() = "Torch-native parallel PX4-like controller bindings";

    py::class_<ParallelPosControl>(module, "ParallelPosControl")
        .def(py::init<std::int64_t, std::string>(), py::arg("envs_num"), py::arg("device"))
        .def("set_status", &ParallelPosControl::set_status)
        .def("update", &ParallelPosControl::update);

    py::class_<ParallelVelControl>(module, "ParallelVelControl")
        .def(py::init<std::int64_t, std::string>(), py::arg("envs_num"), py::arg("device"))
        .def("set_status", &ParallelVelControl::set_status)
        .def("update", &ParallelVelControl::update);

    py::class_<ParallelAttiControl>(module, "ParallelAttiControl")
        .def(py::init<std::int64_t, std::string>(), py::arg("envs_num"), py::arg("device"))
        .def("set_status", &ParallelAttiControl::set_status)
        .def("update", &ParallelAttiControl::update);

    py::class_<ParallelRateControl>(module, "ParallelRateControl")
        .def(py::init<std::int64_t, std::string>(), py::arg("envs_num"), py::arg("device"))
        .def("set_q_world", &ParallelRateControl::set_q_world)
        .def("update", &ParallelRateControl::update);
}
