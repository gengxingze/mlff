import time
import torch
from logger_utils import AverageMeter, Summary
from mlffloss import calc_weight, adjust_lr
import numpy as np


def train_loop(train_load, model, criterion, optimizer, epoch, device, config, logger):
    train_config = config['training']
    batch_size = train_config["batch_size"]

    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e", Summary.AVERAGE)
    loss_etot = AverageMeter("Etot", ":.4e", Summary.ROOT)
    loss_force = AverageMeter("Force", ":.4e", Summary.ROOT)
    loss_ei = AverageMeter("Virial", ":.4e", Summary.ROOT)
    loss_virial = AverageMeter("Virial", ":.4e", Summary.ROOT)

    model.train()
    for i, image_batch in enumerate(train_load):
        start = time.time()
        global_step = (epoch - 1) * len(train_load) + i
        real_lr = adjust_lr(iters=global_step, config=config['learning_rate'])
        for param_group in optimizer.param_groups:
            param_group['lr'] = real_lr
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
        w_e, w_f, w_v, w_ei = calc_weight(config=config['loss_setting'], start_lr=1e-3, real_lr=real_lr)
        if w_v != 0:
            loss_virial_val = 0.0
        if w_ei != 0:
            loss_ei_val = 0.0

        loss = w_e * loss_etot_val / image_batch["natoms"][0] + w_f * loss_force_val + w_v * loss_virial_val + w_ei * loss_ei_val

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_etot_val = loss_etot_val / (image_batch["natoms"][0] ** 2)
        losses.update(loss.item(), batch_size)
        loss_etot.update(loss_etot_val.item(), batch_size)
        loss_force.update(loss_force_val.item(), batch_size)
        loss_virial.update(loss_virial_val.item(), batch_size)
        loss_ei.update(loss_ei_val.item(), batch_size)
        end = time.time()
        batch_time.update(end - start)
        if ((i + 1) * batch_size) % train_config['print_freq'] == 0:
            logger.info("epoch:[%3d][%5d],Etot: %.3e, Force: %.3e, loss: %.3e, lr: %.3e, batch_time: %.3f" % (
                epoch, ((i + 1) * batch_size), loss_etot_val ** 0.5, loss_force_val ** 0.5, losses.val,
                optimizer.param_groups[0]['lr'], batch_time.val))
    for name, param in model.named_parameters():
        if 'bias' in name:
            print(name, param.data)
    return losses.avg, loss_etot.root, loss_force.root, loss_ei.root, loss_virial.root


def vaild_loop(train_load, model, criterion, device, config, logger):
    train_config = config['training']
    batch_size = train_config["batch_size"]

    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e", Summary.AVERAGE)
    loss_etot = AverageMeter("Etot", ":.4e", Summary.ROOT)
    loss_force = AverageMeter("Force", ":.4e", Summary.ROOT)
    loss_ei = AverageMeter("Virial", ":.4e", Summary.ROOT)
    loss_virial = AverageMeter("Virial", ":.4e", Summary.ROOT)
    logger.info("##################### test #####################")
    model.eval()
    for i, image_batch in enumerate(train_load):
        start = time.time()
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

        loss = loss_etot_val / image_batch["natoms"] + loss_force_val + loss_virial_val + loss_ei_val

        loss_etot_val = loss_etot_val / (image_batch["natoms"] ** 2)
        losses.update(loss.item(), batch_size)
        loss_etot.update(loss_etot_val.item(), batch_size)
        loss_force.update(loss_force_val.item(), batch_size)
        loss_virial.update(loss_virial_val.item(), batch_size)
        loss_ei.update(loss_ei_val.item(), batch_size)
        end = time.time()
        batch_time.update(end - start)
        if (i + 1) % train_config['print_freq'] == 0:
            logger.info("epoch:[%5d][],Etot: %.3e, Force: %.3e, loss: %.3e, time: %.3f" % (
                i + 1, loss_etot.val ** 0.5, loss_force.val ** 0.5, losses.val,
                batch_time.val))

    return losses.avg, loss_etot.root, loss_force.root, loss_ei.root, loss_virial.root


def predict(train_load, model, criterion, device, config):
    train_config = config['training']
    model.eval()
    etot_lable_list, force_lable_list, virial_lable_list = [], [], []
    etot_predict_list, force_predict_list, virial_predict_list = [], [], []
    for i, image_batch in enumerate(train_load):
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
        etot_lable_list.append(etot_label.squeeze(dim=0).detach().numpy())
        etot_predict_list.append(etot_predict.squeeze(dim=0).detach().numpy())
        force_lable_list.append(force_label.squeeze(dim=0).detach().numpy())
        force_predict_list.append(force_predict.squeeze(dim=0).detach().numpy())
        print("epoch:[%5d][],Etot: %.3e, Force: %.3e " % (i + 1, loss_etot_val ** 0.5, loss_force_val ** 0.5))

    return etot_predict_list, etot_lable_list, force_predict_list, force_lable_list
