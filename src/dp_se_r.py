from typing import List, Optional, Dict
import time
import torch
import torch.nn as nn


@torch.jit.interface
class ModuleInterface(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class EmbeddingNet(nn.Module):
    def __init__(self,
                 network_size: list[int],
                 activate: str = 'tanh',
                 bias: bool = True,
                 resnet_dt: bool = False):
        super(EmbeddingNet, self).__init__()
        self.network_size = [1] + network_size  # [1, 25, 50, 100]
        self.bias = bias
        self.resnet_dt = resnet_dt
        if activate == 'tanh':
            self.activate = nn.Tanh()
        self.linears = nn.ModuleList()
        self.resnet = nn.ParameterList()
        for i in range(len(self.network_size) - 1):
            self.linears.append(nn.Linear(in_features=self.network_size[i],
                                          out_features=self.network_size[i + 1], bias=self.bias))
            if self.bias:
                nn.init.normal_(self.linears[i].bias, mean=0.0, std=1.0)
            if self.network_size[i] == self.network_size[i+1] or self.network_size[i] * 2 == self.network_size[i+1]:
                resnet_tensor = torch.Tensor(1, self.network_size[i + 1])
                nn.init.normal_(resnet_tensor, mean=0.1, std=0.001)
                self.resnet.append(nn.Parameter(resnet_tensor, requires_grad=True))
            nn.init.normal_(self.linears[i].weight, mean=0.0,
                            std=(1.0 / (self.network_size[i] + self.network_size[i + 1]) ** 0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = 0
        for i, linear in enumerate(self.linears):
            hiden = linear(x)
            hiden = self.activate(hiden)
            if self.network_size[i] == self.network_size[i+1] and self.resnet_dt:
                for ii, resnet in enumerate(self.resnet):
                    if ii == m:
                        x = hiden * resnet + x
                m = m + 1
            elif self.network_size[i] == self.network_size[i+1] and (not self.resnet_dt):
                x = hiden + x
            elif self.network_size[i] * 2 == self.network_size[i+1] and self.resnet_dt:
                for ii, resnet in enumerate(self.resnet):
                    if ii == m:
                        x = hiden * resnet + torch.cat((x, x), dim=-1)
                m = m + 1
            elif self.network_size[i] * 2 == self.network_size[i+1] and (not self.resnet_dt):
                x = hiden + torch.cat((x, x), dim=-1)
            else:
                x = hiden
        return x


class FittingNet(nn.Module):
    def __init__(self,
                 network_size: list[int],
                 activate: str = 'tanh',
                 bias: bool = True,
                 resnet_dt: bool = False,
                 energy_shift: float = 0.0):
        super(FittingNet, self).__init__()
        self.network_size = network_size + [1]  # [input, 25, 50, 100, 1]
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
            if self.network_size[i] == self.network_size[i+1] and self.resnet_dt:
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
                if self.network_size[i] == self.network_size[i+1] and self.resnet_dt:
                    for ii, resnet in enumerate(self.resnet):
                        if ii == m:
                            x = hiden * resnet + x
                    m = m + 1
                elif self.network_size[i] == self.network_size[i+1] and (not self.resnet_dt):
                    x = hiden + x
                elif self.network_size[i] * 2 == self.network_size[i+1] and self.resnet_dt:
                    for ii, resnet in enumerate(self.resnet):
                        if ii == m:
                            x = hiden * resnet + torch.cat((x, x), dim=-1)
                    m = m + 1
                elif self.network_size[i] * 2 == self.network_size[i+1] and (not self.resnet_dt):
                    x = hiden + torch.cat((x, x), dim=-1)
                else:
                    x = hiden
        return x


class DpSe(nn.Module):
    def __init__(self,
                 fit_config: dict,
                 embed_config: dict,
                 type_map: list[int],
                 neighbor: int,
                 energy_shift: list[float],
                 stdv: list[torch.Tensor]):
        super(DpSe, self).__init__()
        self.type_map = type_map
        self.max_neighbor = neighbor
        self.M2 = embed_config["M2"]
        self.Rmax = embed_config["Rmax"]
        self.Rmin = embed_config["Rmin"]

        self.embedding_net = nn.ModuleList()
        self.fitting_net = nn.ModuleList()
        self.embedding_net_index = []
        self.fitting_net_index = []
        self.embedding_size = embed_config["network_size"]
        self.stdv = stdv

        for i, itype in enumerate(self.type_map):
            for j, jtype in enumerate(self.type_map):
                self.embedding_net.append(EmbeddingNet(network_size=embed_config["network_size"],
                                                       activate=embed_config["activate_funtion"],
                                                       bias=embed_config["bias"],
                                                       resnet_dt=embed_config["resnet_dt"]))
                self.embedding_net_index.append(str(itype) + "_" + str(jtype))
            fit_network_size = [embed_config["network_size"][-1] * embed_config["M2"]] + fit_config[
                "network_size"]
            self.fitting_net.append(FittingNet(network_size=fit_network_size,
                                               activate=fit_config["activate_funtion"],
                                               bias=fit_config["bias"],
                                               resnet_dt=fit_config["resnet_dt"],
                                               energy_shift=energy_shift[i]))
            self.fitting_net_index.append(str(itype))

    def forward(self,
                Imagetype: torch.Tensor,
                neighbor_list: torch.Tensor,
                neighbor_type: torch.Tensor,
                ImagedR: torch.Tensor,
                nghost: int):
        # time1 = time.time()
        batch = neighbor_list.shape[0]
        natom = neighbor_list.shape[1]
        max_neighbor = neighbor_list.shape[2]
        # Must be set to int64 or it will cause a type mismatch when run in c++.
        type_map_temp: torch.Tensor = torch.unique(Imagetype).to(torch.int64)
        type_map: List[int] = type_map_temp.tolist()
        ntype = len(type_map)
        # Ri[batch, natom, max_neighbor,1] 1-->(srij) For se_r consider only Srij
        # Ri_d[batch, natom, max_neighbor, 1, 3] 3--->dx,dy,dz
        Ri, Ri_d = self.calculate_Ri(type_map, Imagetype, neighbor_list, neighbor_type, ImagedR)
        # time2 = time.time()
        Ri.requires_grad_()
        # time3 = time.time()
        # Ei[batch, natom, 1]
        Ei = torch.zeros(batch, natom, 1)
        for i, itype in enumerate(type_map):
            mesk_itype = (Imagetype == itype)
            i_Ri = Ri[mesk_itype].reshape(batch, -1, max_neighbor * ntype, 1)
            xyz_scater_a = torch.zeros(batch, i_Ri.shape[1], 1, self.embedding_size[-1])
            for j, jtype in enumerate(type_map):
                # srij[batch, natom of itype, neighbor of jtype(100), 1]
                srij = i_Ri[:, :, j * max_neighbor:(j+1) * max_neighbor, 0].unsqueeze(-1)
                ij = self.embedding_net_index.index(str(itype) + "_" + str(jtype))
                embedding_net: ModuleInterface = self.embedding_net[ij]
                # G[batch, natom of itype, max_neighbor, embeding_net_size[-1]]
                G = embedding_net.forward(srij)
                temp_a = i_Ri[:, :, j * max_neighbor:(j+1) * max_neighbor, :].transpose(-2, -1)
                temp_b = torch.matmul(temp_a, G)
                xyz_scater_a = xyz_scater_a + temp_b
            xyz_scater_a = xyz_scater_a / (self.max_neighbor * len(self.type_map))
            xyz_scater_b = xyz_scater_a[:, :, :, :16]
            # DR_itype[batch, natom of itype, M2 * embeding_net_size[-1]]
            DR_itype = torch.matmul(xyz_scater_a.transpose(-2, -1), xyz_scater_b).reshape(batch, i_Ri.shape[1], -1)
            ii = self.fitting_net_index.index(str(itype))
            fitting_net: ModuleInterface = self.fitting_net[ii]
            # Ei_itype[batch, natom of itype, 1]
            Ei_itype = fitting_net.forward(DR_itype)
            Ei[mesk_itype] = Ei_itype.reshape(-1, 1)
        # Etot[batch, 1]
        Etot = torch.sum(Ei, dim=1)
        mask: List[Optional[torch.Tensor]] = [torch.ones_like(Etot)]
        dE = torch.autograd.grad([Etot], [Ri], grad_outputs=mask, retain_graph=True, create_graph=True)[0]
        # The data type obtained by torch.autograd.grad [0] is Optional[Tensor], and when exported as a torchscripy
        # model, if the variable is subsequently used, it needs to be changed to tensor by the following statement
        assert dE is not None
        # time4 = time.time()
        dE = torch.unsqueeze(dE, dim=-1)
        # dE * Ri_d [batch,natom,max_neighbor*len(type_map),1,1] * [batch, natom, max_neighbor * len(type_map), 1, 3]
        # dE_Rid [batch, natom, max_neighbor * len(type_map), 3]
        dE_Rid = torch.mul(dE, Ri_d).sum(dim=-2)
        # Force[batch, natom, 3]
        Force = torch.zeros(batch, natom + nghost + 1, 3)
        Force[:, 1:natom + 1, :] = -1 * dE_Rid.sum(dim=-2)
        # Here only the action force F_ij = -1 * sum(de_ij * dr_ij) is considered, and the reaction force
        # -F_ji = sum(de_ji * dr_ji) should also be considered and subtracted from.
        # Finally, F_ij = - 1*sum(de_ij * dr_ij) + sum(de_ji * dr_ji)
        # for bb in range(0, batch):
        #     for ii in range(1, natom + nghost):
        #         for tt in range(0, ntype):
        #             Force[bb, ii - 1] = Force[bb, ii - 1] + dE_Rid[bb, :, tt * max_neighbor:(tt + 1) * max_neighbor][
        #                 neighbor_list[bb] == ii].sum(dim=0)
        for bb in range(0, batch):
            indice = neighbor_list[bb].squeeze(dim=0).to(torch.int64).view(-1).unsqueeze(-1).expand(-1, 3)
            values = dE_Rid[bb].squeeze(dim=0).view(-1, 3)
            Force[bb] = Force[bb].scatter_add(0, indice, values).view(natom + nghost + 1, 3)
        Force = Force[:, 1:, :]
        # time5 = time.time()
        # print(time5-time4, time4-time3, time3-time2, time2-time1)
        return Etot, Ei, Force

    def calculate_Ri(self,
                     type_map: List[int],
                     Imagetype: torch.Tensor,
                     neighbor_list: torch.Tensor,
                     neighbor_type: torch.Tensor,
                     ImagedR: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pi = 3.1415926535
        batch = neighbor_list.shape[0]
        natom = neighbor_list.shape[1]
        max_neighbor = neighbor_list.shape[2]

        R = ImagedR[:, :, :, 0]
        R.requires_grad_()
        Srij = torch.zeros(batch, natom, max_neighbor)

        Srij[(R > 0) & (R < self.Rmin)] = 1 / R[(R > 0) & (R < self.Rmin)]
        r = R[(R > self.Rmin) & (R < self.Rmax)]
        Srij[(R > self.Rmin) & (R < self.Rmax)] = 1 / r * (((r - self.Rmin) / (self.Rmax - self.Rmin)) ** 3 * (
                    -6 * ((r - self.Rmin) / (self.Rmax - self.Rmin)) ** 2 + 15 * (
                        (r - self.Rmin) / (self.Rmax - self.Rmin)) - 10) + 1)
        # Ri[(R > self.Rmin) & (R < self.Rmax)] = 1 / r * (
        #             0.5 * torch.cos(pi * (r - self.Rmin) / (self.Rmax - self.Rmin)) + 0.5)

        # d_Srij[batch, natom, max_neighbor]  d_Srij = dSrij/drij
        mesk: List[Optional[torch.Tensor]] = [torch.ones_like(R)]
        d_Srij = torch.autograd.grad([Srij], [R], grad_outputs=mesk, retain_graph=True, create_graph=True)[0]
        assert d_Srij is not None
        # feat [batch, natom, max_neighbor * len(type_map),1]
        # dfeat [batch, natom, max_neighbor * len(type_map), 1, 3] 3-->[dx, dy, dz]
        feat = torch.zeros(batch, natom, max_neighbor * len(type_map), 1)
        dfeat = torch.zeros(batch, natom, max_neighbor * len(type_map), 1, 3)

        for i, itype in enumerate(type_map):
            Srij_temp = Srij.clone()
            Srij_temp[neighbor_type != itype] = 0
            feat[:, :, max_neighbor * i:max_neighbor * (i + 1), 0] = Srij_temp
            d_Srij_temp = d_Srij.clone()
            d_Srij_temp[neighbor_type != itype] = 0

            mask = (d_Srij_temp.abs() > 1e-5)

            common = d_Srij_temp[mask] / R[mask]
            # dfeat[:,:,:,0]-->Srij
            # dSrij / dxij = d_Srij * xij / rij (x-->x,y,z)
            dfeat[:, :, max_neighbor * i:max_neighbor * (i + 1), 0, 0][mask] = ImagedR[:, :, :, 1][mask] / common
            dfeat[:, :, max_neighbor * i:max_neighbor * (i + 1), 0, 1][mask] = ImagedR[:, :, :, 2][mask] / common
            dfeat[:, :, max_neighbor * i:max_neighbor * (i + 1), 0, 2][mask] = ImagedR[:, :, :, 3][mask] / common

        # davg_res = torch.zeros(0)
        # dstd_res = torch.zeros(0)
        # for i, element in enumerate(type_map):
        #     indice = self.type_map.index(element)
        #     davg_res = self.stdv[0][indice]
        #     dstd_res = self.stdv[1][indice]
        #     mesk_res = (Imagetype == element)
        #     feat[mesk_res] = (feat[mesk_res] - 0) / dstd_res[0]
        #     dfeat[mesk_res] = dfeat[mesk_res] / dstd_res[0]
        return feat, dfeat

