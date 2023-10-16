import dpdata
import numpy as np
import neighbor
import torch
from ase import io
import dataset


def get_data(trainfile: list, vaildfile: list, logger):
    logger.info("################################  train data file ###################################")
    logger.info("########## file name ############# Atom number ##### nframes ###### Element #########")

    train_data = read_data(trainfile, cutoff=6.0, logger=logger)

    logger.info("################################  vaild data file ###################################")
    logger.info("########## file name ############# Atom number ##### nframes ###### Element #########")
    vaild_data = read_data(vaildfile, cutoff=6.0, logger=logger)

    return train_data, vaild_data


def read_data(filelist: list, cutoff: float, logger):
    datalist = []
    nimage = 0
    for ii, file in enumerate(filelist):
        dp = dpdata.LabeledSystem(file, fmt="vasp/xml")
        # nframes = dp.get_nframes()
        nframes = 500
        nimage = nimage + nframes
        atom = io.read(file, index='0:' + str(nframes))
        neighbor_list = []
        neighbor_list_type = []
        ImagedR = []
        for i in range(0, nframes):
            atom_imagetype, atom_neigh, atom_neigh_type,atom_imagedr = neighbor.struct_cal(atom[i], cutoff=cutoff)
            ImagedR.append(atom_imagedr)
            neighbor_list.append(atom_neigh)
            neighbor_list_type.append(atom_neigh_type)
        ImagedR = torch.from_numpy(np.array(ImagedR)).to(torch.float32)
        neighbor_list = torch.from_numpy(np.array(neighbor_list)).to(torch.int)
        neighbor_list_type = torch.from_numpy(np.array(neighbor_list_type)).to(torch.int)
        force = torch.from_numpy(dp['forces']).to(torch.float32)
        Etot = torch.from_numpy(np.reshape(dp['energies'], (-1, 1))).to(torch.float32)
        Imagetype = torch.from_numpy(atom[0].numbers)
        iidata = dataset.MlffDataset(nframes, dp.get_natoms(), Imagetype, force, Etot, dp['virials'], neighbor_list,
                                     neighbor_list_type, ImagedR)
        datalist.append(iidata)
        logger.info(
            "%-30s       %5d         %5d          %s" % (file, dp.get_natoms(), nframes, ', '.join(dp.get_atom_names())))
    logger.info(
            "Number of files found: %8d, number of image : %8d" % (len(filelist), nimage))
    logger.info("#####################################################################################")
    return datalist

