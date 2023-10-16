#include <torch/extension.h>
#include "calculate_force.h"


void launch_calculate_force_forward(
    const int64_t* nblist,
    const double* dE_Rid,
    int64_t batch_size,
    int64_t natoms,
    int64_t neigh_num,
    double* force)
{
    int64_t ji_id = 0;
    for (int64_t bb = 0; bb < batch_size; bb++)
    {
        for (int64_t ii = 0; ii < natoms; ii++)
        {
            for (int64_t jj = 0; jj < neigh_num; jj++)
            {
                ji_id = nblist[bb * natoms * neigh_num + ii * neigh_num + jj];
                if (ji_id > 0)
                {
                    for (int64_t xyz = 0; xyz < 3; xyz++)
                    {
                        *(force + bb * natoms * 3 + ji_id * 3 + xyz) += *(dE_Rid + bb * natoms * neigh_num * 3 + ii * neigh_num * 3 + jj * 3 + xyz);
                    }
                }
            }
        }
    }
}

void calculate_force_forward(torch::Tensor& nblist,
    const torch::Tensor& dE_Rid,
    int64_t batch_size,
    int64_t natoms,
    int64_t neigh_num,
    torch::Tensor& force
)
{
    auto dtype = dE_Rid.dtype();
    assert(force.dtype() == dtype);
    std::cout << force.dtype() <<"Force dtype: " << dtype << std::endl;
    if (dtype == torch::kFloat64)
    {
        launch_calculate_force_forward(
            (const int64_t*)nblist.data_ptr(),
            (const double*)dE_Rid.data_ptr(),
            batch_size, natoms, neigh_num,
            (double*)force.data_ptr()
        );

    }
    //else if (dtype == torch::kfloat64)
    //{
    //    launch_calculate_force<double>(
    //        (const int*)nblist.data_ptr(),
    //        (const double*)de.data_ptr(),
    //        (const double*)ri_d.data_ptr(),
    //        batch_size, natoms, neigh_num,
    //        (double*)force.data_ptr()
    //    );
    //}
    //else
    //    std::cout << "rrrrrrrrr" << std::endl;
}

void launch_calculate_force_backward(
    const int64_t* nblist,
    double* grad_output,
    int64_t batch_size,
    int64_t natoms,
    int64_t neigh_num,
    double* grad)
{
    int64_t ji_id = 0;
    for (int64_t bb = 0; bb < batch_size; bb++)
    {
        for (int64_t ii = 0; ii < natoms; ii++)
        {
            for (int64_t jj = 0; jj < neigh_num; jj++)
            {
                ji_id = nblist[bb * natoms * neigh_num + ii * neigh_num + jj];
                if (ji_id > 0)
                {
                    for (int64_t xyz = 0; xyz < 3; xyz++)
                    {
                        *(grad + bb * natoms * neigh_num * 3 + ii * neigh_num * 3 + jj * 3 + xyz) += *(grad_output + bb * natoms * 3 + ji_id * 3 + xyz);
                    }
                }
            }
        }
    }
}

void calculate_force_backward(torch::Tensor& nblist,
    const torch::Tensor& grad_output,
    int64_t batch_size,
    int64_t natoms,
    int64_t neigh_num,
    torch::Tensor& grad
)
{
    auto dtype = grad_output.dtype();
    assert(grad_output.dtype() == dtype);
    std::cout << grad_output.dtype() <<"Force dtype back: " << dtype << std::endl;
    if (dtype == torch::kFloat64)
    {
        launch_calculate_force_backward(
            (const int64_t*)nblist.data_ptr(),
            (double*)grad_output.data_ptr(),
            batch_size, natoms, neigh_num,
            (double*)grad.data_ptr()
        );
        //std::cout << "Force size 000: "  << std::endl;
    }
    //else if (dtype == torch::kfloat64)
    //{
    //    launch_calculate_force<double>(
    //        (const int*)nblist.data_ptr(),
    //        (const double*)de.data_ptr(),
    //        (const double*)ri_d.data_ptr(),
    //        batch_size, natoms, neigh_num,
    //        (double*)force.data_ptr()
    //    );
    //}
    //else
    //    std::cout << "rrrrrrrrr" << std::endl;
}