#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "src/runtime/runtime_engine.h"

namespace py = pybind11;
using namespace seecpp::runtime;

PYBIND11_MODULE(seecpp, m) {
    // Submodule for runtime
    py::module_ runtime_m = m.def_submodule("runtime", "SeeC++ Execution VM");

    py::class_<RuntimeEngine>(runtime_m, "RuntimeEngine")
        .def(py::init<>())
        
        // Wrap Load (Translate std::expected to Python Exceptions)
        .def("load", [](RuntimeEngine& self, std::string_view path) {
            auto result = self.Load(path);
            if (!result) throw std::runtime_error(result.error().message);
        })

        // Wrap SetInput (Accept a numpy array)
        .def("set_input", [](RuntimeEngine& self, py::array_t<float> input_array) {
            py::buffer_info buf = input_array.request();
            auto result = self.SetInput(static_cast<const float*>(buf.ptr), buf.size);
            if (!result) throw std::runtime_error(result.error().message);
        })

        // Wrap Invoke
        .def("invoke", [](RuntimeEngine& self) {
            auto result = self.Invoke();
            if (!result) throw std::runtime_error(result.error().message);
        })

        // Wrap GetOutput (Return a zero-copy numpy array pointing to the arena)
        .def("get_output", [](const RuntimeEngine& self, size_t offset, std::vector<size_t> shape) {
            const float* out_ptr = self.GetOutput(offset);
            if (!out_ptr) throw std::runtime_error("Invalid offset or arena not mapped.");
            
            // Create a numpy array that does NOT own the memory (zero-copy view)
            return py::array_t<float>(shape, out_ptr, py::cast(self));
        });
}
