# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch
import numpy as np

import utils.misc as misc
import utils.lr_sched as lr_sched

import matplotlib.pyplot as plt
import librosa

def show_image(spec, cmap='jet', extent=None, vmin=-12, vmax=-0, save_path=None, title=None, is_freq=False):

    if not is_freq:
        spec = librosa.power_to_db(spec)
    data = np.flipud(spec)
    # print(data.shape)
    extent = 0, 5, 0, 200
    # plt.figure(figsize=(16, 4))
    # plt.imshow(data, cmap, extent=extent, vmin=vmin, vmax=vmax)
    plt.imshow(data, cmap, extent=extent)
    plt.title(title, fontsize=16)

    plt.axis('auto')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('time')
    plt.colorbar()
    # plt.savefig(save_path + '/{}'.format(name))
    # plt.clf()
    # plt.close()

@torch.no_grad()
def log_result(input_spec, mask, pred_spec, model, writer, epoch):
    # print(input_spec.shape, pred_spec.shape)
    x = input_spec[0]
    x = x.detach().cpu().numpy()

    y = model.unpatchify(pred_spec)
    y = y.squeeze()[0].detach().cpu().numpy()

    # h, w = model.patch_embed.patch_hw
    ma = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2)
    ma = model.unpatchify(ma)
    ma = ma.squeeze()[0].detach().cpu().numpy()

    im = np.resize(x, ma.shape)

    im_masked = im * (1 - ma)
    im_paste = im * (1 - ma) + y * ma

    fig = plt.figure(figsize=(100, 20))
    # plt.rcParams['figure.figsize'] = [100, 24]

    plt.subplot(1, 4, 1)
    show_image(x, title="original", is_freq=model.is_freq)

    plt.subplot(1, 4, 2)
    show_image(im_masked, title="masked", is_freq=model.is_freq)

    plt.subplot(1, 4, 3)
    show_image(y, title="reconstruction", is_freq=model.is_freq)

    plt.subplot(1, 4, 4)
    show_image(im_paste, title="reconstruction + visible", is_freq=model.is_freq)

    plt.tight_layout()
    writer.add_figure('Result', fig, global_step=epoch)

    # plt.show()

    print('debug')


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 200

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    # set model epoch
    model.epoch = epoch

    # for data_iter_step, (samples, _labels, _vids) in enumerate(
    #         metric_logger.log_every(data_loader, print_freq, header)):

    for data_iter_step, samples in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # print(samples.shape)# 64x3x224x224 for img, 64x1x512x128 for audio
        samples = samples.to(device, non_blocking=True, dtype=torch.float32)

        # comment out when not debugging
        # from fvcore.nn import FlopCountAnalysis, parameter_count_table
        # if data_iter_step == 1:
        #     flops = FlopCountAnalysis(model, samples)
        #     print(flops.total())
        #     print(parameter_count_table(model))

        with torch.cuda.amp.autocast(enabled=False):
            loss_a, pred, mask, _, input_spec = model(samples, mask_ratio=args.mask_ratio)

        # print(pred.shape, mask.shape, pred_spec.shape, input_spec.shape)

        loss_value = loss_a.item()
        loss_total = loss_a

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # loss /= accum_iter
        loss_total = loss_total / accum_iter
        loss_scaler(loss_total, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    if log_writer is not None:
        log_result(input_spec, mask, pred, model, log_writer, epoch)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}





