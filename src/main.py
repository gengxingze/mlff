import json
import numpy as np
import torch
import train
import mlp
import dp_se_a as dp_se_a
import time
import torch.nn as nn
import logger_utils
import read_data
import mtp
import argparse
import os


os.chdir(r"D:\work\Pycharm\mlff\test")


def calc_stat(type_map, traindata, net):
    for j, data in enumerate(traindata):
        if all(np.in1d(type_map, np.array(data[0]['Imagetype']))):
            if True:
                Imagetype = torch.unsqueeze(data[0]['Imagetype'], dim=0)
                neighbor_list = torch.unsqueeze(data[0]['neighbor_list'], dim=0)
                neighbor_type = torch.unsqueeze(data[0]['neighbor_type'], dim=0)
                ImagedR = torch.unsqueeze(data[0]['ImagedR'], dim=0)
                feat, dfeat = net.calculate_Ri(type_map, Imagetype, neighbor_list, neighbor_type, ImagedR)
                davg = torch.zeros(len(type_map))
                dstd = torch.zeros(len(type_map), 2)
                for i, itype in enumerate(type_map):
                    iRi = feat[Imagetype == itype, :]
                    iRi2 = iRi * iRi
                    sum_Ri = iRi.sum()
                    sum2_Ri = iRi2.sum()
                    idRi = dfeat[Imagetype == 29]
                    idRi2 = idRi * idRi

                    davg_unit = sum_Ri / torch.nonzero(idRi).shape[0]
                    dstd_unit = [torch.sqrt(sum2_Ri/torch.nonzero(iRi2).shape[0]), torch.sqrt(idRi2.sum()/torch.nonzero(idRi2).shape[0])]
                    dstd_unit = torch.tensor(dstd_unit)
                    davg[i] = davg_unit.detach()
                    dstd[i] = dstd_unit.detach()
    return [davg, dstd]


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    logger = logger_utils.setup_logger()
    with open('log', 'w') as fp:
        fp.write('################################   mlff log file  ###################################\n')
    with open("input.json") as fp:
        config = json.load(fp)
    print(config)

    num_threads = torch.get_num_threads()
    num_cores = torch.get_num_interop_threads()
    # torch.set_num_threads(1)
    logger.info("num_threads : %d, num_cores: %d" % (num_threads, num_cores))

    is_read_data = True
    if is_read_data:
        training_data_list, vaild_data_list = read_data.get_data(config['data_setting']['train_file'], config['data_setting']['valid_file'], logger)
        all_data = torch.utils.data.ConcatDataset(training_data_list)
        vaild_data = torch.utils.data.ConcatDataset(vaild_data_list)
        train_load = torch.utils.data.DataLoader(all_data, batch_size=2, shuffle=True, num_workers=1)
    #model_name = config['model']['model_name']
    model_name = 'deepmd'
    if model_name == 'nn':
        net = mlp.MLP(config["model"]["fitting_net"],
                      config["model"]["descriptor"],
                      config["model"]["type_map"],
                      config["model"]["neighbor"],
                      energy_shift=[-3.37, -4.0], stdv=None)
    if model_name == 'deepmd':
        # net = dp_se_a.DpSe(config["model"]["fitting_net"],
        #                    config["model"]["descriptor"],
        #                    config["model"]["type_map"],
        #                    config["model"]["neighbor"],
        #                    energy_shift=[-3.37, -4.0], stdv=None)
        # stdv = calc_stat(config["model"]["type_map"], training_data_list, net)
        net = dp_se_a.DpSe(config["model"]["fitting_net"],
                           config["model"]["descriptor"],
                           config["model"]["type_map"],
                           config["model"]["neighbor"],
                           energy_shift=[-3.37,-4.0], stdv=[0])
    if model_name == 'mtp':
        net = mtp.MTP(config["model"]["fitting_net"],
                      config["model"]["descriptor"],
                      config["model"]["type_map"],
                      config["model"]["neighbor"],
                      energy_shift=[-3.37, -4.0], stdv=None)
    net.double()
    torch.set_default_dtype(torch.float64)
    device = torch.device("cpu")
    read_modle = False
    if read_modle:
        # net.load_state_dict(torch.load('output/latest.pt')["state_dict"])
        net = torch.jit.load('output/script.pt')
    is_train = True
    epochs = 1
    loss_fun = nn.MSELoss()
    predict_load = torch.utils.data.DataLoader(vaild_data, batch_size=1, shuffle=False, num_workers=1)
    if is_train:
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        for epoch in range(1, epochs + 1):
            start = time.time()
            losses, loss_etot, loss_force, loss_ei, loss_virial = train.train_loop(train_load, net, loss_fun, optimizer, epoch, device, config, logger)
            end = time.time()
            with open('train.out', 'a') as fp:
                fp.write("%d  %e %e %e %e %e %e\n" % (epoch,  losses, loss_etot, loss_force, loss_ei, loss_virial, optimizer.param_groups[0]['lr']))
            torch.save({"epoch": epoch, "net": net.state_dict(), "optimizer": optimizer.state_dict()}, "checkpoint.pt")
            is_test = False
            if is_test:
                losses, loss_etot, loss_force, loss_ei, loss_virial = train.vaild_loop(predict_load, net, loss_fun, device, config, logger)
                with open("vaild.out", 'a') as fp:
                    fp.write("%d  %e %e %e %e %e %e\n" % (
                    epoch, losses, loss_etot, loss_force, loss_ei, loss_virial, optimizer.param_groups[0]['lr']))
        torch.save({"state_dict": net.state_dict()}, "latest.pt")
    is_generating = True
    if is_generating:
        ms = torch.jit.script(net.eval())
        ms.save('script.pt')
        #mf = torch.jit.optimize_for_inference(ms)
        #mf.save('script_f.pt')
        #tt = torch.jit.trace(net,(torch.full((1, 128), 29), torch.full((1, 128, 100), 29), torch.full((1, 128, 100), 29), torch.zeros(1, 128, 100, 4),0))
        #frozen_mod = torch.jit.optimize_for_inference(torch.jit.script(net.eval()))
        #frozen_mod.save('output/script1.pt')
    ms = torch.jit.load('script.pt')
    ispredict = False
    if ispredict:
        losses, loss_etot, loss_force, loss_ei, loss_virial = train.vaild_loop(predict_load, net, loss_fun, device,
                                                                                config, logger)
        Ep, Eb, Fp, Fb = train.predict(predict_load, ms, loss_fun, device, config)
        Ep, Eb, Fp, Fb = np.array(Ep), np.array(Eb), np.array(Fp), np.array(Fb)
        import matplotlib.pyplot as plt
        plt.figure(dpi=300, figsize=(20, 16))
        plt.subplot(221)
        plt.scatter(Eb, Ep, s=5)
        Emin = min(Eb.min(), Ep.min())
        Emax = max(Eb.max(), Ep.max())
        plt.plot([Emin, Emax], [Emin, Emax], color='black', linestyle='--')
        plt.xlabel('enregy DFT')
        plt.ylabel('energy predict')
        plt.xlim(Emin, Emax)
        plt.ylim(Emin, Emax)

        plt.subplot(222)
        plt.plot(np.arange(1, Ep.shape[0]+1), Ep, label='E_predict')
        plt.plot(np.arange(1, Ep.shape[0]+1), Eb, label='E_dft')
        plt.xlabel('step')
        plt.ylabel('energy')
        plt.legend()

        plt.subplot(223)
        plt.scatter(Fb.reshape(-1), Fp.reshape(-1), s=5)
        Fmin = min(Fb.min(), Fp.min())
        Fmax = max(Fb.max(), Fp.max())
        plt.plot([Fmin, Fmax], [Fmin, Fmax], color='black', linestyle='--')
        plt.xlabel('force DFT')
        plt.ylabel('force predict')
        plt.xlim(Fmin, Fmax)
        plt.ylim(Fmin, Fmax)

        plt.subplot(224)
        plt.plot(np.arange(1, Ep.shape[0]+1), ((np.array(Fp)-np.array(Fb))**2/(3*128)).sum(axis=1).sum(axis=1)**0.5, label='force rmse')
        plt.xlabel('step')
        plt.ylabel('force rmse')
        plt.legend()
        plt.savefig('ff.png', dpi=128)
    for i in range(100):
        aa = torch.rand(1, 128, 100, 4) * 3
        t1 = time.time()
        a, b, c = ms(torch.full((1, 128), 29), torch.full((1, 128, 100), 29), torch.full((1, 128, 100), 29), aa, 0)
        print('time',time.time() - t1)
    print('0000000')

