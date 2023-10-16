
void launch_calculate_force_forward(
    const int64_t* nblist,
    const double* dE_Rid,
    int64_t batch_size,
    int64_t natoms,
    int64_t neigh_num,
    double* force);
    

void launch_calculate_force_backward(
    const int64_t* nblist,
    double* grad_output,
    int64_t batch_size,
    int64_t natoms,
    int64_t neigh_num,
    double* grad);
