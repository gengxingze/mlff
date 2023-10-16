import math


def calc_weight(config, start_lr, real_lr):
    w_f = config['limit_force_weight'] + (
                 config['start_force_weight'] - config['limit_force_weight']) * real_lr / start_lr
    w_e = config['limit_energy_weight'] + (
                config['start_energy_weight'] - config['limit_energy_weight']) * real_lr / start_lr
    w_v = config['limit_virial_weight'] + (
                config['start_virial_weight'] - config['limit_virial_weight']) * real_lr / start_lr
    w_ei = config['limit_ei_weight'] + (
                config['start_ei_weight'] - config['limit_ei_weight']) * real_lr / start_lr
    return w_e, w_f, w_v, w_ei


def adjust_lr(iters, config):
    start_lr = config['start_lr']
    stop_lr = config['stop_lr']
    step = config['step']
    if config['type'] == "exp":
        lr = start_lr * 0.9 ** (iters//step)
        if lr <= stop_lr:
            return stop_lr
        else:
            return lr

