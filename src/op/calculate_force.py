from torch import nn
import torch
from torch.autograd import Function
import op


class CalculateForce(Function):
    @staticmethod
    def forward(ctx, neighbor_list, dE_Rid, F):
        dims = neighbor_list.shape
        batch_size = dims[0]
        natoms = dims[1]
        neigh_num = dims[2]
        ctx.save_for_backward(neighbor_list, dE_Rid)
        print('calculate foward 0')
        op.calculate_force_forward(neighbor_list, dE_Rid, batch_size, natoms, neigh_num, F)
        print('calculate foward 0')
        return F

    @staticmethod
    def backward(ctx, grad_output):
        inputs = ctx.saved_tensors
        neighbor_list = inputs[0]
        dE_Rid = inputs[1]
        dims = neighbor_list.shape
        batch_size = dims[0]
        natoms = dims[1]
        neigh_num = dims[2]
        grad = torch.zeros_like(dE_Rid)
        print('grad_output',grad_output.shape)
        op.calculate_force_backward(neighbor_list, grad_output, batch_size, natoms, neigh_num, grad)
        return (None, grad, None)