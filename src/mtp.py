from typing import List, Optional, Dict
import time
import torch
import torch.nn as nn
import numpy as np


@torch.jit.interface
class ModuleInterface(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        tensor = torch.Tensor(40, 5)
        nn.init.normal_(tensor, mean=0.1, std=0.2)
        self.a = nn.Tanh()
        self.n = nn.Parameter(tensor, requires_grad=True)
        self.l1 = nn.Linear(25, 25, bias=True)
        self.l2 = nn.Linear(25, 25, bias=True)

    def forward(self, x: torch.Tensor, Imagetype: torch.Tensor, neighbor_list: torch.Tensor) -> torch.Tensor:
        batch = neighbor_list.shape[0]
        natom = neighbor_list.shape[1]
        max_neighbor = neighbor_list.shape[2]

        # Convert atom ids in the nearest-neighbour table to their corresponding atom types
        # t1 = time.time_ns()
        neighbor_type = torch.zeros_like(neighbor_list)
        for i in range(0, batch):
            aa = torch.cat((torch.zeros(1), Imagetype[i]))
            bb = neighbor_list[i].int()
            neighbor_type[i] = aa[bb]
        # t2 = time.time_ns()
        # aa = torch.cat((torch.zeros(batch, 1), Imagetype), dim=-1)
        # combined = aa.unsqueeze(-1).expand(batch, natom + 1, max_neighbor)
        # neighbor_type = torch.gather(combined, -2, neighbor_list.to(torch.int64)).int()
        # t3 = time.time_ns()

        iitype = torch.zeros(batch, natom, 5)
        jjtype = torch.zeros(batch, natom, max_neighbor, 5)
        for bb in range(0, batch):
            iitemp = Imagetype[bb]
            iitype[bb] = self.n[iitemp]
            for nn in range(0, natom):
                jjtemp = neighbor_type[bb, nn].int()
                jjtype[bb, nn] = self.n[jjtemp]
        embed = torch.matmul(iitype.reshape(batch,natom,1,5,1) ,torch.unsqueeze(jjtype, dim=-2))
        embed =embed.reshape(batch,natom,max_neighbor,1,25)

        x = torch.matmul(x.unsqueeze(-1), embed)
        # print(t3-t2, t2-t1)
        return x


class FittingNet(nn.Module):
    def __init__(self):
        super(FittingNet, self).__init__()

        if True:
            self.activate = nn.Tanh()
        self.linears = nn.ModuleList()
        self.resnet = nn.ParameterList()
        tensor = torch.Tensor(1, 10)
        nn.init.normal_(tensor, mean=0.1, std=0.1)
        self.n = nn.Parameter(tensor, requires_grad=True)
        self.l1 = nn.Linear(625, 100, bias=True)
        self.l2 = nn.Linear(100, 100, bias=True)
        self.l3 = nn.Linear(100, 100, bias=True)
        self.l4 = nn.Linear(100, 1, bias=True)
        nn.init.normal_(self.l4.bias, mean=3.3, std=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1=self.l1(x)
        x2=self.activate(x1)
        x3=self.l2(x2) + x1
        x4=self.activate(x3)
        x5=self.l3(x4) + x3

        x=self.activate(x5)
        x=self.l4(x)

        return x


class MTP(nn.Module):
    def __init__(self,
                 fit_config: dict,
                 embed_config: list[dict],
                 type_map: list[int],
                 neighbor: int,
                 energy_shift: list[float],
                 stdv: torch.Tensor,
                 ):
        super(MTP, self).__init__()
        self.type_map = type_map
        self.max_neighbor = neighbor
        self.stdv = stdv

        self.net = FittingNet()
        self.em = EmbeddingNet()

    def forward(self, Imagetype: torch.Tensor, neighbor_list: torch.Tensor, ImagedR: torch.Tensor):
        # t1 = time.time()
        batch = neighbor_list.shape[0]
        natom = neighbor_list.shape[1]
        max_neighbor = neighbor_list.shape[2]
        # Must be set to int64 or it will cause a type mismatch when run in c++.
        type_map_temp: torch.Tensor = torch.unique(Imagetype).to(torch.int64)
        type_map: List[int] = type_map_temp.tolist()
        ntype = len(type_map)
        feat, dfeat = self.calculate_moment_tensor(Imagetype, neighbor_list, ImagedR,6.0,0.5,25)
        # t2 = time.time()
        G = self.em(feat, Imagetype, neighbor_list)
        G = G.sum(dim=2).reshape(batch, natom, 25*25)
        Ei = self.net(G)
        Etot = torch.sum(Ei, dim=1)
        mask: List[Optional[torch.Tensor]] = [torch.ones_like(Etot)]
        dE = torch.autograd.grad([Etot], [feat], grad_outputs=mask, retain_graph=True, create_graph=True)[0]
        # The data type obtained by torch.autograd.grad [0] is Optional[Tensor], and when exported as a torchscripy
        # model, if the variable is subsequently used, it needs to be changed to tensor by the following statement
        assert dE is not None
        # t3 = time.time()
        dE = dE.unsqueeze(3)
        dE_Rid = torch.matmul(dE, dfeat).squeeze(dim=3)
        Force = -1 * dE_Rid.sum(dim=2)
        for bb in range(0, batch):
            for ii in range(1, natom + 1):
                Force[bb, ii - 1] = Force[bb, ii - 1] + dE_Rid[bb][neighbor_list[bb] == ii].sum(dim=0)
        # t4 = time.time()
        # print(t4-t3, t3-t2, t2-t1, t4-t1)
        return Etot, Ei, Force

    def calculate_moment_tensor(self, 
                                Imagetype: torch.Tensor, 
                                neighbor_list: torch.Tensor, 
                                ImagedR: torch.Tensor,
                                Rmax: float,
                                Rmin: float,
                                n_basis: int) -> tuple[torch.Tensor, torch.Tensor]:
        pi = 3.141592653589
        batch = neighbor_list.shape[0]
        natom = neighbor_list.shape[1]
        max_neighbor = neighbor_list.shape[2]

        xij = ImagedR[:, :, :, 1]
        yij = ImagedR[:, :, :, 2]
        zij = ImagedR[:, :, :, 3]
        rij = ImagedR[:, :, :, 0]
        rs = torch.linspace(Rmin, Rmax, n_basis)
        r_temp = torch.repeat_interleave(rij, n_basis, dim=2).reshape(batch, natom, max_neighbor, n_basis)
        r_temp.requires_grad_()
        fc = torch.where((r_temp < Rmax) & (r_temp > 0),
                         torch.exp(- 1.0 * (r_temp - rs) ** 2) * (0.5 * torch.cos(pi * r_temp) + 0.5), 0)
        # dfc = torch.where((r_temp < Rmax) & (r_temp > 0), (-0.5 * pi * torch.sin(pi * r_temp))-(2 * (r_temp - rs) * (0.5 * torch.cos(pi * r_temp) + 0.5))
        #                   * torch.exp(- 1.0 * (r_temp - rs) ** 2), 0)
        mesk: List[Optional[torch.Tensor]] = [torch.ones_like(r_temp)]
        dfc = torch.autograd.grad([fc], [r_temp], grad_outputs=mesk, retain_graph=True, create_graph=True)[0]
        assert dfc is not None
        feat = fc
        dfeat = torch.zeros(batch, natom, max_neighbor, n_basis, 3)

        dfeat[:, :, :, :, 0] = torch.unsqueeze(torch.where(rij != 0, xij / rij, 0), dim=-1) * dfc
        dfeat[:, :, :, :, 1] = torch.unsqueeze(torch.where(rij != 0, yij / rij, 0), dim=-1) * dfc
        dfeat[:, :, :, :, 2] = torch.unsqueeze(torch.where(rij != 0, zij / rij, 0), dim=-1) * dfc
        return feat, dfeat
