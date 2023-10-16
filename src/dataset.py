from torch.utils.data import Dataset
import torch


class MlffDataset(Dataset):
    def __init__(self, nframes: int,
                 natoms: int,
                 atom_type: torch.Tensor,
                 force: torch.Tensor,
                 energy: torch.Tensor,
                 virial: torch.Tensor,
                 neighbor_list: torch.Tensor,
                 neighbor_type: torch.Tensor,
                 imagedR: torch.Tensor):
        super(MlffDataset, self).__init__()
        self.nframes = nframes
        self.natoms = natoms
        self.atom_type = atom_type
        self.force = force
        self.energy = energy
        self.virial = virial
        self.neighbor_list = neighbor_list
        self.neighbor_type = neighbor_type
        self.imagedR = imagedR

    def __len__(self):
        return self.nframes

    def __getitem__(self, index):
        dic = {
            'natoms': self.natoms,
            'Imagetype': self.atom_type,
            'force': self.force[index],
            'energy': self.energy[index],
            'virial': self.virial[index],
            'neighbor_list': self.neighbor_list[index],
            'neighbor_type': self.neighbor_type[index],
            'ImagedR': self.imagedR[index],
        }
        return dic






