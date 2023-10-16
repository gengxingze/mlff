#include <torch/extension.h>

void calculate_force_forward(torch::Tensor &nblist,
                        const torch::Tensor &dE_Rid,
                        int64_t batch_size,
                        int64_t natoms,
                        int64_t neigh_num,
                        torch::Tensor &force
);

void calculate_force_backward(torch::Tensor &nblist,
                       const torch::Tensor &grad_output,
                       int64_t batch_size,
                       int64_t natoms,
                       int64_t neigh_num,
                       torch::Tensor &grad
);
