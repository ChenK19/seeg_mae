import argparse
import torch
from dataset import SeegDataset
import models_mae
import utils.misc as misc

from torch.utils.tensorboard import SummaryWriter

import timm.optim.optim_factory as optim_factory
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
import time

import json
from engine_pretrain import train_one_epoch
import os

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)

    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.8, type=float,
                        help='Masking ratio (percentage of removed patches).')  # 0.75

    # parser.add_argument('--norm_pix_loss', action='store_true',
    #                    help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.add_argument('--norm_pix_loss', type=bool, default=False,
                        help='Use (per-patch) normalized pixels as targets for computing loss')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='/home/jz/hard_disk/data/20231215_SeegMAE_model/video_mae_VitB_pretrained.pth',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=20, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # For audioset
    parser.add_argument('--audio_exp', type=bool, default=True, help='audio exp')
    # parser.add_argument("--data_train", type=str, default='/checkpoint/berniehuang/ast/egs/audioset/data/datafiles/train.json', help="training data json")
    # parser.add_argument("--data_eval", type=str, default='/checkpoint/berniehuang/ast/egs/audioset/data/datafiles/eval.json', help="validation data json")
    parser.add_argument("--data_train", type=str,
                        default='/checkpoint/berniehuang/ast/egs/audioset/data/datafiles/train_video.json',
                        help="training data json")
    parser.add_argument("--data_eval", type=str,
                        default='/checkpoint/berniehuang/ast/egs/audioset/data/datafiles/eval_video.json',
                        help="validation data json")
    parser.add_argument("--label_csv", type=str,
                        default='/checkpoint/berniehuang/ast/egs/audioset/data/class_labels_indices.csv',
                        help="csv with class labels")
    parser.add_argument('--freqm', help='frequency mask max length', type=int, default=0)  # pretraining 0
    parser.add_argument('--timem', help='time mask max length', type=int, default=0)  # pretraining 0
    parser.add_argument("--mixup", type=float, default=0,
                        help="how many (0-1) samples need to be mixup during training")
    parser.add_argument("--dataset", type=str, default="audioset", help="dataset",
                        choices=["audioset", "esc50", "speechcommands"])
    parser.add_argument("--use_fbank", type=bool, default=False)
    parser.add_argument("--fbank_dir", type=str,
                        default="/checkpoint/berniehuang/ast/egs/esc50/data/ESC-50-master/fbank", help="fbank dir")
    parser.add_argument("--alpha", type=float, default=0.0, help="contrastive loss weight")
    parser.add_argument("--omega", type=float, default=1.0, help="reconstruction loss weight")
    parser.add_argument('--mode', default=0, type=int, help='contrastive mode')
    parser.add_argument('--save_every_epoch', default=10, type=int, help='save_every_epoch')
    parser.add_argument('--use_custom_patch', type=bool, default=False,
                        help='use custom patch and override timm PatchEmbed')

    parser.add_argument("--distributed", type=bool, default=False)
    parser.add_argument('--roll_mag_aug', type=bool, default=False, help='use roll_mag_aug')
    parser.add_argument('--split_pos', type=bool, default=False, help='use splitted pos emb')
    parser.add_argument('--pos_trainable', type=bool, default=False, help='use trainable pos emb')
    parser.add_argument('--use_nce', type=bool, default=False, help='use use_nce')
    parser.add_argument('--load_video', type=bool, default=False, help='load video')
    parser.add_argument('--decoder_mode', default=1, type=int, help='decoder mode 0: global attn 1: swined local attn')
    # remove for A-MAE
    # parser.add_argument('--v_weight', default=1.0, type=float, help='reconstruction weight for the visual part')
    # parser.add_argument('--video_only', type=bool, default=False, help='video_only pre-training')
    # parser.add_argument('--cl', type=bool, default=False, help='use pre-text curriculum')
    # parser.add_argument('--n_frm', default=4, type=int,help='how many frames to encode, at least 2 as temporal kernel stride is 2')
    # parser.add_argument('--depth_av', default=3, type=int,help='depth of multimodal fusion encoder')
    parser.add_argument('--mask_t_prob', default=0.7, type=float, help='ratio of masking time')
    parser.add_argument('--mask_f_prob', default=0.3, type=float, help='ratio of masking freq')
    parser.add_argument('--mask_2d', type=bool, default=False, help='use 2d masking')
    parser.add_argument('--init_audio_with_video_mae', type=bool, default=False, help='init_audio_with_video_mae')
    parser.set_defaults(audio_exp=True)
    parser.add_argument('--no_shift', type=bool, default=False, help='no_shift')

    parser.add_argument('--data_path_folder', type=str, default='/home/jz/hard_disk/data/20231213_anes_data/5s_segment_400Hz', help='where is the data segment')
    parser.add_argument('--data_path_mean_std', type=str,
                        default='/home/jz/hard_disk/data/20231213_anes_data/mean_std_is_freq_False.pkl',
                        help='where is the data mean std')
    parser.add_argument('--sample_freq', type=int,
                        default='400',
                        help='sample_freq for the sample')
    parser.add_argument('--seg_len', type=int,
                        default='2000',
                        help='segment length for the sample')
    parser.add_argument('--batch_size', default=256, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')

    parser.add_argument('--files_num', default=-1, type=int,
                        help='used samples number, -1 for all samples')

    parser.set_defaults(norm_pix_loss=True)
    return parser

def main(args):


    seeg_conf = {
        'sample_freq': 400,
        'freqm': args.freqm,
        'timem': args.timem,
        'data_path_folder': args.data_path
    }

    dataset = SeegDataset(args) #加载所有需要训练的数据到dataset中
    sampler_train = torch.utils.data.RandomSampler(dataset)

    data_loader_train = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False
    )

    batch_size = args.batch_size
    device = torch.device(args.device)

    model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss, img_size=(201, 201), is_freq=False) #img_size 是频谱的尺寸 201,201 对应n_fft 400, hop_length10

    # model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss, img_size=(129, 126))  # img_size 是频谱的尺寸

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))

    print(optimizer)
    loss_scaler = NativeScaler()

    # misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    os.makedirs(args.log_dir, exist_ok=True)
    log_writer = SummaryWriter(log_dir=args.log_dir)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):

        # if args.distributed:
        #     data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and (epoch % args.save_every_epoch == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    # print('debug')


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    main(args)


