#!/usr/bin/env python3
import argparse
import datetime
import json
import numpy as np
import sys
import os
import time
from pathlib import Path
sys.path.append(os.path.abspath("mamba-1p1p1"))
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms

import timm
import timm.optim.optim_factory as optim_factory

from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.misc import (
    count_parameters, init_distributed_mode, get_rank, get_world_size, 
    is_main_process, load_model
)

import src.models_tor_mamba as models_tor_mamba
from engine import pretrain_one_epoch
from contextlib import suppress
from torchvision.datasets import ImageFolder, folder
from collections import Counter, defaultdict
import random

class FilteredImageFolder(ImageFolder):
    def __init__(self, root, category='obfs', transform=None, target_transform=None, 
                 loader=folder.default_loader, is_valid_file=None):
        super(FilteredImageFolder, self).__init__(root, transform, target_transform, loader, is_valid_file)
        
        if category == 'obfs':
            selected_classes = [cls for cls in self.classes if 'obfs' in cls.lower()]
        elif category == 'normal':
            selected_classes = [cls for cls in self.classes if 'obfs' not in cls.lower()]
        else:
            raise ValueError(f"Unknown category: {category}")

        if not selected_classes:
            raise ValueError(f"No classes found for category '{category}' in {root}. Available classes: {self.classes}")

        print(f"Found {category} classes: {selected_classes}")

        allowed_class_indices = {
            class_idx for class_idx, class_name in enumerate(self.classes)
            if class_name in selected_classes
        }

        filtered_samples = []
        for path, class_idx in self.samples:
            if class_idx in allowed_class_indices:
                filtered_samples.append((path, class_idx))

        self.samples = filtered_samples
        self.targets = [s[1] for s in self.samples]


        old_to_new_idx = {}
        new_classes = []
        new_class_to_idx = {}

        for old_idx, class_name in enumerate(self.classes):
            if class_name in selected_classes:
                new_idx = len(new_classes)
                old_to_new_idx[old_idx] = new_idx
                new_classes.append(class_name)
                new_class_to_idx[class_name] = new_idx

        self.classes = new_classes
        self.class_to_idx = new_class_to_idx
        self.samples = [(path, old_to_new_idx[old_idx]) for path, old_idx in self.samples]
        self.targets = [old_to_new_idx[old_idx] for old_idx in self.targets]

        print(f"Filtered dataset: {len(self.samples)} samples from {len(self.classes)} {category} classes")
        for cls in self.classes:
            cls_count = sum(1 for _, target in self.samples if self.classes[target] == cls)
            print(f"  {cls}: {cls_count} samples")

def get_args_parser():
    parser = argparse.ArgumentParser('TorMamba pre-training for obfs or normal classes', add_help=False)
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU')
    parser.add_argument('--steps', default=50000, type=int)
    parser.add_argument('--save_steps_freq', default=5000, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations')
    parser.add_argument('--category', type=str, default='obfs', choices=['obfs', 'normal'],
                        help="Choose which category to use: 'obfs' or 'normal'")
    # Model parameters
    parser.add_argument('--model', default='tor_mamba_pretrain', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=40, type=int,
                        help='images input size')
    parser.add_argument('--stride_size', default=4, type=int,
                        help='images stride size')
    parser.add_argument('--mask_ratio', default=0.90, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)
    parser.add_argument('--byte_length', default=1600, type=int,
                        help='the length of the byte sequence')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=25, metavar='N',
                        help='warmup steps')

    # Dataset parameters
    parser.add_argument('--data_path', default='Custom/dataset_sampled_normal/train', type=str,
                        help='dataset path')
    parser.add_argument('--output_dir', default='./output/pretrain_obfs',
                        help='path where to save')
    parser.add_argument('--log_dir', default='./output/pretrain_obfs',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # AMP parameters
    parser.add_argument('--no_amp', action='store_true', default=False,
                        help='Disable automatic mixed precision')

    return parser

def main(args):
    init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    mean = [0.5]
    std = [0.5]

    transform_train = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


    dataset_train = ImageFolder(args.data_path, transform=transform_train)
    print(f"Dataset: {dataset_train}")
    print(f"Classes: {dataset_train.classes}")

    num_tasks = get_world_size()
    global_rank = get_rank()


    targets = np.array(dataset_train.targets)
    class_indices = defaultdict(list)
    for idx, label in enumerate(targets):
        class_indices[label].append(idx)
    min_count = min(len(idxs) for idxs in class_indices.values())
    selected_indices = []
    for idxs in class_indices.values():
        selected_indices.extend(random.sample(idxs, min_count))
    random.shuffle(selected_indices)

    print(f"Undersampling: using {min_count} samples per class, total {len(selected_indices)} samples.")

    from torch.utils.data import Subset
    dataset_train_balanced = Subset(dataset_train, selected_indices)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train_balanced,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )


    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None


    model = models_tor_mamba.__dict__[args.model](
        norm_pix_loss=args.norm_pix_loss,
        byte_length=args.byte_length,
        stride_size=args.stride_size
    )
    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))
    trainable_params, all_param = count_parameters(model)
    print("trainable params: {:d} || all params: {:d} || trainable%: {:.4f}".format(
        trainable_params, all_param, 100 * trainable_params / all_param
    ))

    if num_tasks > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module


    eff_batch_size = args.batch_size * args.accum_iter * get_world_size()
    
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    param_groups = timm.optim.optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)


    amp_autocast = suppress
    loss_scaler = "none"
    if not args.no_amp:
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()

    load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    epochs = int(args.steps / len(data_loader_train)) + 1
    args.epochs = epochs

    print(f"Start training for {args.steps} steps ({epochs} epochs)")
    start_time = time.time()
    
    for epoch in range(0, epochs):
        if num_tasks > 1:
            data_loader_train.sampler.set_epoch(epoch)
            
        train_stats = pretrain_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler, amp_autocast,
            log_writer=log_writer,
            model_without_ddp=model_without_ddp,
            args=args
        )


        print(f"Epoch {epoch}: loss={train_stats.get('loss', 'N/A')}, acc={train_stats.get('acc', 'N/A')}")

        if args.output_dir:
            steps = len(data_loader_train) * epoch
            if steps % args.save_steps_freq == 0 or steps >= args.steps:
                checkpoint_paths = [os.path.join(args.output_dir, f'checkpoint-{steps}.pth')]
                if is_main_process():
                    for checkpoint_path in checkpoint_paths:
                        torch.save({
                            'model': model_without_ddp.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'epoch': epoch,
                            'steps': steps,
                            'scaler': loss_scaler.state_dict() if hasattr(loss_scaler, 'state_dict') else None,
                            'args': args,
                        }, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch, }

        if args.output_dir and is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        if len(data_loader_train) * epoch >= args.steps:
            break


    if log_writer is not None:
        log_writer.close()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)