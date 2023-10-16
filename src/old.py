Ri, Ri_d = self.calculate_R_i(torch.squeeze(Imagetype, dim=0), torch.squeeze(neighbor_list, dim=0),
                              torch.squeeze(ImagedR, dim=0))
Ri.requires_grad_()
ntype = [29]
Ei = torch.zeros(1)
# Ei = Optional
for ii, net in enumerate(self.embedding_net):
    for i, itype in enumerate(ntype):
        # xyz_scater_a = None
        xyz_scater_a = torch.zeros(1)
        srij = torch.zeros(1)
        for j, jtype in enumerate(ntype):
            srij = Ri[torch.squeeze(Imagetype, dim=0) == itype][:,
                   j * ImagedR.shape[2]:(j + 1) * ImagedR.shape[2]].unsqueeze(-1)
            temp_a = srij.transpose(-2, -1)
            if ii == i * len(ntype) + j:
                G = net(srij)
                temp_b = torch.matmul(temp_a, G)
                if j == 0:
                    xyz_scater_a = temp_b
                else:
                    xyz_scater_a += temp_b
            else:
                G = torch.zeros(1)

            # xyz_scater_a = temp_b if xyz_scater_a is None else xyz_scater_a + temp_b

        xyz_scater_a = xyz_scater_a / (self.max_neighbor * len(self.type_map))
        xyz_scater_b = xyz_scater_a[:, :, :16]
        DR_ntype = torch.matmul(xyz_scater_a.transpose(-2, -1), xyz_scater_b)
        DR_ntype = DR_ntype.reshape(srij.shape[0], -1)

        for iii, fnet in enumerate(self.fitting_net):
            Ei = torch.zeros(1)
            if iii == i:
                Ei_ntype = fnet(DR_ntype)
                if i == 0:
                    Ei = Ei_ntype
                else:
                    Ei = torch.concat((Ei, Ei_ntype), dim=1)
            else:
                Ei_ntype = torch.zeros(1)

        # Ei = Ei_ntype if Ei is Optional else torch.concat((Ei, Ei_ntype), dim=1)