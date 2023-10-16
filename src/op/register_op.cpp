#include "op_declare.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

    m.def("calculate_force_forward", &calculate_force_forward, "calculate force forward");
    m.def("calculate_back_forward", &calculate_force_backward, "calculate force backward");

}

TORCH_LIBRARY(op, m) {
    m.def("calculate_force_forward", calculate_force_forward);
    m.def("calculate_force_backward", calculate_force_backward);

}