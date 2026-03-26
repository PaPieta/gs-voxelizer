#include "bindings.h"
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("optim_to_render_forward", &optim_to_render_forward_torch);
    m.def("optim_to_render_backward", &optim_to_render_backward_torch);
    m.def("bin_and_sort_gaussians", &bin_and_sort_gaussians_torch);
    m.def("voxelize_forward", &voxelize_forward_torch);
    m.def("voxelize_backward", &voxelize_backward_torch);
    m.def("voxelize_backward_per_gaussian", &voxelize_backward_per_gaussian_torch);
}
