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

logger = logger_utils.setup_logger()
with open('log', 'w') as fp:
    fp.write('################################   mlff log file  ###################################\n')
with open("input.json") as fp:
    config = json.load(fp)
print(config)
torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
num_threads = torch.get_num_threads()
num_cores = torch.get_num_interop_threads()
# torch.set_num_threads(1)
logger.info("num_threads : %d, num_cores: %d" % (num_threads, num_cores))

is_read_data = True
if is_read_data:
    training_data_list, vaild_data_list = read_data.get_data(config['data_setting']['train_file'],
                                                             config['data_setting']['valid_file'], logger)
    all_data = torch.utils.data.ConcatDataset(training_data_list)
    vaild_data = torch.utils.data.ConcatDataset(vaild_data_list)
    train_load = torch.utils.data.DataLoader(all_data, batch_size=1, shuffle=True, num_workers=1)
# model_name = config['model']['model_name']
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
                       energy_shift=[-3.37, -4.0], stdv=[0])
if model_name == 'mtp':
    net = mtp.MTP(config["model"]["fitting_net"],
                  config["model"]["descriptor"],
                  config["model"]["type_map"],
                  config["model"]["neighbor"],
                  energy_shift=[-3.37, -4.0], stdv=None)

net.to(device)
net.double()
read_modle = False
if read_modle:
    # net.load_state_dict(torch.load('output/latest.pt')["state_dict"])
    net = torch.jit.load('output/script.pt')
is_train = True
epochs = 1
loss_fun = nn.MSELoss()
predict_load = torch.utils.data.DataLoader(vaild_data, batch_size=1, shuffle=False, num_workers=1)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)


def train_loop(train_load, model, criterion, optimizer, device, config):
    model.train()
    train_config = config['training']
    batch_size = train_config["batch_size"]
    for i, image_batch in enumerate(train_load):
        start = time.time()
        global_step = (epoch - 1) * len(train_load) + i
        if train_config['dtype'] == "float32":
            ImagedR = image_batch['ImagedR'].float().to(device)
            etot_label = image_batch['energy'].float().to(device)
            force_label = image_batch['force'].float().to(device)
            # ei_label = image_batch['ImagedR'].float()
        elif train_config['dtype'] == "float64":
            ImagedR = image_batch['ImagedR'].double().to(device)
            etot_label = image_batch['energy'].double().to(device)
            force_label = image_batch['force'].double().to(device)
            # ei_label = image_batch['ImagedR'].float()
        else:
            raise Exception("Error! Please specify floating point type:float32 or float64 by the parameter --datatype!")

        Imagetype = image_batch["Imagetype"].int().to(device)
        neighbor_list = image_batch['neighbor_list'].int().to(device)
        neighbor_type = image_batch['neighbor_type'].int().to(device)
        etot_predict, ei_predict, force_predict = model(Imagetype, neighbor_list, neighbor_type, ImagedR, 0)
        loss_etot_val = criterion(etot_label, etot_predict)
        loss_force_val = criterion(force_label, force_predict)
        loss_virial_val = torch.tensor([0.0])
        loss_ei_val = torch.tensor([0.0])

        loss = 1.0 * loss_etot_val / 128 + 10 * loss_force_val
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
train_loop(train_load, net, loss_fun, optimizer, device, config)