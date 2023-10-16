import ase.io
import numpy as np
from ase.io import vasp
from ase.neighborlist import NeighborList


def struct_cal(atoms, cutoff):
    nl = NeighborList([cutoff / 2] * len(atoms), skin=0, self_interaction=False, bothways=True)
    nl.update(atoms)
    atoms_num = len(atoms)
    imagedR = []
    neigh_list = []
    # 打印最近邻表
    for i in range(atoms_num):
        indices, offsets = nl.get_neighbors(i)

        temp_a = atoms.positions[indices] + np.dot(offsets, atoms.get_cell())
        temp = atoms.positions[i] - temp_a
        dxyz = np.pad(temp, ((0, 100-temp.shape[0]), (0, 0)), mode='constant', constant_values=(0, 0))
        dr = np.sqrt(np.sum(dxyz ** 2, axis=1))
        atomdR = np.column_stack((dr, dxyz))
        atom_neigh = np.pad(indices+1, (0, 100-len(indices)), mode='constant', constant_values=(0, 0))
        imagedR.append(atomdR)
        neigh_list.append(atom_neigh)
    imagedR = np.array(imagedR)
    neigh_list = np.array(neigh_list)

    imagetype = np.insert(atoms.numbers, 0, 0)
    neigh_list_type = imagetype[neigh_list]

    imagedR[imagedR == 0] = 100.0
    sortarray, indices = np.sort(imagedR[:, :, 0], axis=-1), np.argsort(imagedR[:, :, 0], axis=-1)
    temp = np.zeros((atoms_num, 100, 4))
    for i in range(4):
        temp[:, :, i] = np.take_along_axis(imagedR[:, :, i], indices, axis=-1)
    temp[temp == 100] = 0
    imagedR = temp
    neigh_list_type = np.take_along_axis(neigh_list_type, indices, axis=-1)
    neigh_list = np.take_along_axis(neigh_list, indices, axis=-1)
    return atoms.numbers, neigh_list, neigh_list_type, imagedR
