from typing import List, Optional
import torch
import torch.nn as nn
import time
import numpy as np

@torch.jit.interface
class ModuleInterface(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class NET(nn.Module):
    def __init__(self,
                 network_size: list[int],
                 activate: str = 'tanh',
                 bias: bool = True,
                 resnet_dt: bool = False,
                 energy_shift: float = 0.0):
        super(NET, self).__init__()
        self.network_size = [50, 100, 100, 100, 1]  # [input, 25, 50, 100, 1]
        self.bias = bias
        self.resnet_dt = resnet_dt
        if activate == 'tanh':
            self.activate = nn.Tanh()
        self.linears = nn.ModuleList()
        self.resnet = nn.ParameterList()
        for i in range(len(self.network_size) - 1):
            if i == (len(self.network_size) - 2):
                self.linears.append(nn.Linear(in_features=self.network_size[i],
                                              out_features=self.network_size[i + 1], bias=True))
                nn.init.normal_(self.linears[i].bias, mean=energy_shift, std=1.0)
            else:
                self.linears.append(nn.Linear(in_features=self.network_size[i],
                                              out_features=self.network_size[i + 1], bias=self.bias))
                if self.bias:
                    nn.init.normal_(self.linears[i].bias, mean=0.0, std=1.0)
            if self.network_size[i] == self.network_size[i + 1] and self.resnet_dt:
                resnet_tensor = torch.Tensor(1, self.network_size[i + 1])
                nn.init.normal_(resnet_tensor, mean=0.1, std=0.001)
                self.resnet.append(nn.Parameter(resnet_tensor, requires_grad=True))
            nn.init.normal_(self.linears[i].weight, mean=0.0,
                            std=(1.0 / (self.network_size[i] + self.network_size[i + 1]) ** 0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = 0
        for i, linear in enumerate(self.linears):
            if i == (len(self.network_size) - 2):
                hiden = linear(x)
                x = hiden
            else:
                hiden = linear(x)
                hiden = self.activate(hiden)
                if self.network_size[i] == self.network_size[i + 1] and self.resnet_dt:
                    for ii, resnet in enumerate(self.resnet):
                        if ii == m:
                            x = hiden * resnet + x
                    m = m + 1
                elif self.network_size[i] == self.network_size[i + 1] and (not self.resnet_dt):
                    x = hiden + x
                elif self.network_size[i] * 2 == self.network_size[i + 1] and self.resnet_dt:
                    for ii, resnet in enumerate(self.resnet):
                        if ii == m:
                            x = hiden * resnet + torch.cat((x, x), dim=-1)
                    m = m + 1
                elif self.network_size[i] * 2 == self.network_size[i + 1] and (not self.resnet_dt):
                    x = hiden + torch.cat((x, x), dim=-1)
                else:
                    x = hiden
        return x


class MLP(nn.Module):
    def __init__(self,
                 fit_config: dict,
                 embed_config: list[dict],
                 type_map: list[int],
                 neighbor: int,
                 energy_shift: list[float],
                 stdv: torch.Tensor,
                 ):
        super(MLP, self).__init__()
        self.type_map = type_map
        self.max_neighbor = neighbor
        self.stdv = stdv

        network_size = [25, 100, 100, 100]
        self.fitting_net = nn.ModuleList()
        self.fitting_net_index = []
        for i, itype in enumerate(self.type_map):
            self.fitting_net.append(NET(network_size=network_size,
                                        activate=fit_config["activate_funtion"],
                                        bias=fit_config["bias"],
                                        resnet_dt=fit_config["resnet_dt"],
                                        energy_shift=energy_shift[i]))
            self.fitting_net_index.append(str(itype))

    def forward(self, Imagetype: torch.Tensor, neighbor_list: torch.Tensor, ImagedR: torch.Tensor):
        # t1 = time.time()
        batch = neighbor_list.shape[0]
        natom = neighbor_list.shape[1]
        max_neighbor = neighbor_list.shape[2]
        # Must be set to int64 or it will cause a type mismatch when run in c++.
        type_map_temp: torch.Tensor = torch.unique(Imagetype).to(torch.int64)
        type_map: List[int] = type_map_temp.tolist()
        ntype = len(type_map)
        feat, dfeat = self.body2fun(Imagetype, neighbor_list, ImagedR, 6.0, 0.5, 25)
        feat.requires_grad_()
        # t2 = time.time()
        Ei = torch.zeros(batch, natom, 1)
        for i, itype in enumerate(type_map):
            mesk_itype = (Imagetype == itype)
            iifeat = feat[mesk_itype].reshape(batch, -1, feat.shape[-1])
            ii = self.fitting_net_index.index(str(itype))
            fitting_net: ModuleInterface = self.fitting_net[ii]
            Ei_itype = fitting_net.forward(iifeat)
            Ei[mesk_itype] = Ei_itype.reshape(-1, 1)
        Etot = torch.sum(Ei, dim=1)
        mask: List[Optional[torch.Tensor]] = [torch.ones_like(Etot)]
        dE = torch.autograd.grad([Etot], [feat], grad_outputs=mask, retain_graph=True, create_graph=True)[0]
        # t3 = time.time()
        # The data type obtained by torch.autograd.grad [0] is Optional[Tensor], and when exported as a torchscripy
        # model, if the variable is subsequently used, it needs to be changed to tensor by the following statement
        assert dE is not None
        dE = dE.unsqueeze(-2).unsqueeze(-2)
        dE_Rid = torch.matmul(dE, dfeat).squeeze(dim=3)
        Force = -1 * dE_Rid.sum(dim=2)
        for bb in range(0, batch):
            for ii in range(1, natom + 1):
                Force[bb, ii - 1] = Force[bb, ii - 1] + dE_Rid[bb][neighbor_list[bb] == ii].sum(dim=0)
        # t4 = time.time()
        # print(t4-t3, t3-t2, t2-t1, t4-t1)
        return Etot, Ei, Force

    def body2fun(self,
                 Imagetype: torch.Tensor,
                 neighbor_list: torch.Tensor,
                 ImagedR: torch.Tensor,
                 Rmax: float,
                 Rmin: float,
                 n_basis: int) -> tuple[torch.Tensor, torch.Tensor]:
        # t1 = time.time()
        pi = 3.1415926
        batch = neighbor_list.shape[0]
        natom = neighbor_list.shape[1]
        max_neighbor = neighbor_list.shape[2]
        # Convert atom ids in the nearest-neighbour table to their corresponding atom types
        neighbor_type = torch.zeros_like(neighbor_list)
        for i in range(0, batch):
            aa = torch.cat((torch.zeros(1), Imagetype[i]))
            bb = neighbor_list[i].int()
            neighbor_type[i] = aa[bb]
        # t2 = time.time()
        xij = ImagedR[:, :, :, 1]
        yij = ImagedR[:, :, :, 2]
        zij = ImagedR[:, :, :, 3]
        rij = ImagedR[:, :, :, 0]
        rs = torch.linspace(Rmin, Rmax, n_basis)
        # rij [batch,natom,max_neighbor] ->rij  Expansion of rij by n_basis ->r_temp[batch,natom,max_neighbor,n_basis]
        r_temp = torch.repeat_interleave(rij, n_basis, dim=2).reshape(batch, natom, max_neighbor, n_basis)
        r_temp.requires_grad_()
        fc = torch.where((r_temp < Rmax) & (r_temp > 0), torch.exp(- 1.0 * (r_temp - rs) ** 2)*(0.5 * torch.cos(pi * r_temp)+ 0.5), 0)
        # dfc = torch.where((r_temp < Rmax) & (r_temp > 0), (-0.5 * pi * torch.sin(pi * r_temp))-(2 * (r_temp - rs) * (0.5 * torch.cos(pi * r_temp) + 0.5))
        #                   * torch.exp(- 1.0 * (r_temp - rs) ** 2), 0)
        # t3 = time.time()
        mesk: List[Optional[torch.Tensor]] = [torch.ones_like(r_temp)]
        dfc = torch.autograd.grad([fc], [r_temp], grad_outputs=mesk, retain_graph=True, create_graph=True)[0]
        assert dfc is not None
        # t4 = time.time()
        feat = torch.zeros(batch, natom, int(n_basis * len(self.type_map)))
        dfeat = torch.zeros(batch, natom, max_neighbor, n_basis * len(self.type_map), 3)
        for i, itype in enumerate(self.type_map):
            fc_temp = fc.clone()
            fc_temp[neighbor_type != itype] = 0
            feat[:, :,  n_basis * i:n_basis * (i + 1)] = torch.sum(fc_temp, dim=-2)

            dfc_temp = dfc.clone()
            dfc_temp[neighbor_type != itype] = 0

            dfeat[:, :, :, n_basis * i:n_basis * (i + 1), 0] = torch.unsqueeze(
                torch.where(rij != 0, xij / rij, 0), dim=-1) * dfc_temp
            dfeat[:, :, :, n_basis * i:n_basis * (i + 1), 1] = torch.unsqueeze(
                torch.where(rij != 0, yij / rij, 0), dim=-1) * dfc_temp
            dfeat[:, :, :, n_basis * i:n_basis * (i + 1), 2] = torch.unsqueeze(
                torch.where(rij != 0, zij / rij, 0), dim=-1) * dfc_temp
        # t5 = time.time()
        # print(t5-t4, t4-t3, t3-t2, t2-t1, t4-t1)
        return feat, dfeat

