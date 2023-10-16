import os
import unittest
import json
import logger_utils
import dpdata
from ase import io
import neighbor
import torch
import numpy as np
import dataset
import mlp
import dp_se_r
import torch.nn as nn
import train
import time


def calc_stat(type_map, Imagetype, feat, dfeat):
    Imagetype = Imagetype
    feat = torch.squeeze(feat[0], dim=0)
    dfeat = torch.squeeze(dfeat[0], dim=0)
    davg = torch.zeros(len(type_map))
    dstd = torch.zeros(len(type_map), 2)
    for i, itype in enumerate(type_map):
        iRi = feat[Imagetype == itype, :]
        iRi2 = iRi * iRi
        sum_Ri = iRi.sum(axis=0)
        sum2_Ri = iRi2.sum(axis=0)
        idRi = dfeat[0, Imagetype == itype, :]
        idRi2 = idRi * idRi

        davg_unit = (sum_Ri / (iRi.shape[0] + 1e-10)).sum() / 50
        dstd_unit = [torch.sqrt(sum2_Ri / feat.shape[0]).sum() / 50, torch.sqrt(idRi2.sum(axis=0) / feat.shape[0]).sum() / 50]
        dstd_unit = torch.tensor(dstd_unit)
        davg[i] = davg_unit.detach()
        dstd[i] = dstd_unit.detach()
    return [davg, dstd]


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.train_load = None
        self.net = None
        self.logger = None
        self.config = None
        self.workspace = None
        self.read_data()
        self.read_config()
        pass

    def tearDown(self) -> None:
        pass

    def read_config(self, fileName):
        self.logger = logger_utils.setup_logger()
        with open('log', 'w') as fp:
            fp.write('################################   mlff log file  ###################################\n')
        with open(fileName) as fp:
            self.config = json.load(fp)
        print(self.config)
        print(self.config['data_setting']['training_data']['file'])

    def read_data(self):
        logger = logger_utils.setup_logger()
        with open('log', 'w') as fp:
            fp.write('################################   mlff log file  ###################################\n')
        with open("input.json") as fp:
            config = json.load(fp)
        print(config)
        print(config['data_setting']['training_data']['file'])
        read_data = True
        if read_data:
            datalist = []
            logger.info("################################  train data file ###################################")
            logger.info("########## file name ############# Atom number ##### nframes ###### Element #########")
            for file in config['data_setting']['training_data']['file']:
                dp = dpdata.LabeledSystem(file, fmt="vasp/xml")
                nframes = dp.get_nframes()
                # nframes = 100
                atom = io.read(file, index='0:' + str(nframes))
                neighbor_list = []
                ImagedR = []
                for i in range(0, nframes):
                    atom_imagetype, atom_neigh, atom_imagedr = neighbor.struct_cal(atom[i], cutoff=6.0)
                    ImagedR.append(atom_imagedr)
                    neighbor_list.append(atom_neigh)
                ImagedR = torch.from_numpy(np.array(ImagedR)).to(torch.float32)
                neighbor_list = torch.from_numpy(np.array(neighbor_list)).to(torch.int)
                force = torch.from_numpy(dp['forces']).to(torch.float32)
                Etot = torch.from_numpy(np.reshape(dp['energies'], (-1, 1))).to(torch.float32)
                Imagetype = torch.from_numpy(atom[0].numbers)
                train_data = dataset.MlffDataset(nframes, dp['atom_numbs'][0], Imagetype, force, Etot, dp['virials'],
                                                 neighbor_list, ImagedR)
                datalist.append(train_data)
                logger.info("%s          %d          %d          %s" % (
                    file, dp.get_natoms(), nframes, ', '.join(dp.get_atom_names())))
            logger.info("#####################################################################################")
            all_data = torch.utils.data.ConcatDataset(datalist)
            self.train_load = torch.utils.data.DataLoader(all_data, batch_size=2, shuffle=False, num_workers=2)

    def load_MLP_net(self):
        self.net = mlp.MLP(self.config["model"]["fit_net"], self.config["model"]["feature"],
                           self.config["model"]["type_map"],
                           energy_shift=-3.37, stdv=None)
        self.net = self.net.to(torch.float64)

    def load_dpser_net(self):
        Imagetype = None
        neighbor_list = None
        ImagedR = None
        net = dp_se_r.DPER2(self.config["model"]["fit_net"], self.config["model"]["feature"],
                            self.config["model"]["type_map"],
                            energy_shift=-3.3086, stdv=None)
        feat, dfeat = net.calculate_R_i([29], Imagetype, neighbor_list[0:1], ImagedR[0:1])
        stdv = calc_stat([29], Imagetype, feat, dfeat)
        self.net = dp_se_r.DPER2(self.config["model"]["fit_net"], self.config["model"]["feature"],
                                 self.config["model"]["type_map"],
                                 energy_shift=-3.3086, stdv=stdv)
        self.net = self.net.to(torch.float64)

    def train(self):
        epochs = 50
        device = torch.device("cpu")
        loss_fun = nn.MSELoss()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001)
        for epoch in range(1, epochs + 1):
            start = time.time()
            losses, loss_etot, loss_force, loss_ei, loss_virial = train.train_loop(self.train_load, self.net, loss_fun,
                                                                                   optimizer, epoch, device, self.config,
                                                                                   self.logger)
            end = time.time()
            with open('output/lc.out', 'a') as fp:
                fp.write("%d  %e %e %e %e %e %e\n" % (
                    epoch, losses, loss_etot, loss_force, loss_ei, loss_virial, optimizer.param_groups[0]['lr']))
            torch.save({"epoch": epoch, "net": self.net.state_dict(), "optimizer": optimizer.state_dict()},
                       "output/checkpoint.pt")
        torch.save({"state_dict": self.net.state_dict()}, "output/latest.pt")

    def generating(self):
        ms = torch.jit.script(self.net)
        ms.save('output/script.pt')
        ms = torch.jit.load('output/script.pt')

    def test_main(self):
        logger = logger_utils.setup_logger()
        with open('log', 'w') as fp:
            fp.write('################################   mlff log file  ###################################\n')
        with open("input.json") as fp:
            config = json.load(fp)
        print(config)
        print(config['data_setting']['training_data']['file'])
        read_data = True
        if read_data:
            datalist = []
            logger.info("################################  train data file ###################################")
            logger.info("########## file name ############# Atom number ##### nframes ###### Element #########")
            for file in config['data_setting']['training_data']['file']:
                dp = dpdata.LabeledSystem(file, fmt="vasp/xml")
                nframes = dp.get_nframes()
                # nframes = 100
                atom = io.read(file, index='0:' + str(nframes))
                neighbor_list = []
                ImagedR = []
                for i in range(0, nframes):
                    atom_imagetype, atom_neigh, atom_imagedr = neighbor.struct_cal(atom[i], cutoff=6.0)
                    ImagedR.append(atom_imagedr)
                    neighbor_list.append(atom_neigh)
                ImagedR = torch.from_numpy(np.array(ImagedR)).to(torch.float32)
                neighbor_list = torch.from_numpy(np.array(neighbor_list)).to(torch.int)
                force = torch.from_numpy(dp['forces']).to(torch.float32)
                Etot = torch.from_numpy(np.reshape(dp['energies'], (-1, 1))).to(torch.float32)
                Imagetype = torch.from_numpy(atom[0].numbers)
                train_data = dataset.MlffDataset(nframes, dp['atom_numbs'][0], Imagetype, force, Etot, dp['virials'],
                                                 neighbor_list, ImagedR)
                datalist.append(train_data)
                logger.info("%s          %d          %d          %s" % (
                file, dp.get_natoms(), nframes, ', '.join(dp.get_atom_names())))
            logger.info("#####################################################################################")
            all_data = torch.utils.data.ConcatDataset(datalist)
            train_load = torch.utils.data.DataLoader(all_data, batch_size=2, shuffle=False, num_workers=2)
        model_name = 'MLP'
        if model_name == 'MLP':
            net = mlp.MLP(config["model"]["fit_net"], config["model"]["feature"], config["model"]["type_map"],
                          energy_shift=-3.37, stdv=None)
        if model_name == 'dpser':
            net = dp_se_r.DPER2(config["model"]["fit_net"], config["model"]["feature"], config["model"]["type_map"],
                                energy_shift=-3.3086, stdv=None)
            feat, dfeat = net.calculate_R_i([29], Imagetype, neighbor_list[0:1], ImagedR[0:1])
            stdv = calc_stat([29], Imagetype, feat, dfeat)
            net = dp_se_r.DPER2(config["model"]["fit_net"], config["model"]["feature"], config["model"]["type_map"],
                                energy_shift=-3.3086, stdv=stdv)
        net = net.to(torch.float64)
        torch.set_default_dtype(torch.float64)
        device = torch.device("cpu")
        read_modle = False
        if read_modle:
            net.load_state_dict(torch.load('output/latest.pt')["state_dict"])

        is_train = True
        epochs = 50
        loss_fun = nn.MSELoss()
        if is_train:
            optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
            for epoch in range(1, epochs + 1):
                start = time.time()
                losses, loss_etot, loss_force, loss_ei, loss_virial = train.train_loop(train_load, net, loss_fun,
                                                                                       optimizer, epoch, device, config,
                                                                                       logger)
                end = time.time()
                with open('output/lc.out', 'a') as fp:
                    fp.write("%d  %e %e %e %e %e %e\n" % (
                    epoch, losses, loss_etot, loss_force, loss_ei, loss_virial, optimizer.param_groups[0]['lr']))
                torch.save({"epoch": epoch, "net": net.state_dict(), "optimizer": optimizer.state_dict()},
                           "output/checkpoint.pt")
            torch.save({"state_dict": net.state_dict()}, "output/latest.pt")

        is_test = False
        if is_test:
            losses, loss_etot, loss_force, loss_ei, loss_virial = train.vaild_loop(train_load, net, loss_fun, None,
                                                                                   device, config)
        is_generating = True
        if is_generating:
            ms = torch.jit.script(net)
            ms.save('output/script.pt')
        ms = torch.jit.load('output/script.pt')

        print('0000000')

    def test_MLP(self):
        self.workspace = "./MLP/lr_0.01"
        os.chdir(self.workspace)
        self.read_data()
        self.load_MLP_net()
        self.train()
        self.generating()

    def test_DPSER(self):
        self.workspace = "./DPSER/lr_0.01"
        os.chdir(self.workspace)
        self.read_data()
        self.load_dpser_net()
        self.train()
        self.generating()

    def test_run(self):
        import subprocess
        nProcess = 16
        self.workspace = "./lmp"
        os.chdir(self.workspace)
        command = "salloc -p normal -w g1 -n 28"
        proc = subprocess.Popen(command)
        os.system(f"mpirun -np {nProcess} lmp -in in.lmp")

    def test_clean(self):
        # os.system("scancel -u gengxingze")
        ID = 0
        os.system(f"scancel {ID}")
