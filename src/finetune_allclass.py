#!/usr/bin/env python3
import argparse
import datetime
import json
import numpy as np
import os
import shutil
import time
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import random
import sys
sys.path.append('src')
sys.path.append(os.path.abspath("mamba-1p1p1"))
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from torchvision import datasets, transforms
import util.lr_decay as lrd
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import src.models_tor_mamba as models_tor_mamba
from contextlib import suppress
from engine import train_one_epoch, evaluate

class FilteredImageFolder(datasets.ImageFolder):
    def __init__(self, root, category='obfs', transform=None, target_transform=None, 
                 loader=datasets.folder.default_loader, is_valid_file=None):
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

def detailed_evaluate(data_loader, model, device, class_names, epoch=None, save_dir=None):
    model.eval()
    
    all_preds = []
    all_targets = []
    all_probs = []
    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            

            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    

    avg_loss = total_loss / len(data_loader)
    accuracy = 100.0 * sum(np.array(all_preds) == np.array(all_targets)) / len(all_targets)
    

    precision, recall, f1, support = precision_recall_fscore_support(
        all_targets, all_preds, average=None, labels=range(len(class_names))
    )
    

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='macro'
    )
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='micro'
    )
    

    labels = list(range(len(class_names)))
    report = classification_report(
        all_targets, all_preds, 
        labels=labels,
        target_names=class_names, 
        output_dict=True,
        zero_division=0
    )
    

    cm = confusion_matrix(all_targets, all_preds)
    

    results = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'macro_precision': macro_precision * 100,
        'macro_recall': macro_recall * 100,
        'macro_f1': macro_f1 * 100,
        'micro_precision': micro_precision * 100,
        'micro_recall': micro_recall * 100,
        'micro_f1': micro_f1 * 100,
        'per_class_metrics': {},
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }
    

    for i, class_name in enumerate(class_names):
        results['per_class_metrics'][class_name] = {
            'precision': precision[i] * 100,
            'recall': recall[i] * 100,
            'f1': f1[i] * 100,
            'support': int(support[i])
        }
    

    if save_dir and epoch is not None:
        save_visualizations(cm, class_names, results, epoch, save_dir)
    
    return results

def save_visualizations(cm, class_names, results, epoch, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - Epoch {epoch}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'confusion_matrix_epoch_{epoch}.png'), dpi=300)
    plt.close()
    

    metrics = ['precision', 'recall', 'f1']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, metric in enumerate(metrics):
        values = [results['per_class_metrics'][cls][metric] for cls in class_names]
        axes[i].bar(class_names, values)
        axes[i].set_title(f'{metric.capitalize()} per Class - Epoch {epoch}')
        axes[i].set_ylabel(f'{metric.capitalize()} (%)')
        axes[i].tick_params(axis='x', rotation=45)
        

        for j, v in enumerate(values):
            axes[i].text(j, v + 1, f'{v:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'metrics_per_class_epoch_{epoch}.png'), dpi=300)
    plt.close()

def get_args_parser():
    parser = argparse.ArgumentParser('NetMamba fine-tuning for obfs with detailed metrics', add_help=False)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--accum_iter', default=1, type=int)
    parser.add_argument('--category', type=str, default='obfs', choices=['obfs', 'normal'],
                        help="Choose which category to use: 'obfs' or 'normal'")
    # Model parameters
    parser.add_argument('--model', default='tor_mamba_classifier', type=str, metavar='MODEL')
    parser.add_argument('--input_size', default=40, type=int)
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT')
    parser.add_argument('--byte_length', default=1600, type=int)
    # parser.add_argument('--stride_size', default=4, type=int,
    #                     help='images stride size')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM')
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--lr', type=float, default=None, metavar='LR')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR')
    parser.add_argument('--layer_decay', type=float, default=0.75)
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR')
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N')

    # Augmentation parameters
    parser.add_argument('--smoothing', type=float, default=0.1)

    # Dataset parameters
    parser.add_argument('--data_path', default='Custom/dataset_sampled_normal', type=str)
    parser.add_argument('--nb_classes', default=None, type=int)
    parser.add_argument('--output_dir', default='./output/finetune_obfs_metrics')
    parser.add_argument('--log_dir', default='./output/finetune_obfs_metrics')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='')

    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://')

    # AMP parameters
    parser.add_argument('--no_amp', action='store_true', default=False)

    return parser

def build_dataset(data_split, args):
    mean = [0.5]
    std = [0.5]

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    assert data_split in ["train", "test", "valid"]
    root = os.path.join(args.data_path, data_split)
    
    dataset = FilteredImageFolder(root, category=args.category, transform=transform)
    print(f"{data_split} dataset: {len(dataset)} samples, classes: {dataset.classes}")
    return dataset

def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True


    dataset_train = build_dataset(data_split="train", args=args)
    dataset_val = build_dataset(data_split="valid", args=args)
    dataset_test = build_dataset(data_split="test", args=args)


    if len(dataset_train.classes) == 0:
        print(f"Error: No {args.category} classes found")
        return


    args.nb_classes = len(dataset_train.classes)
    class_names = dataset_train.classes
    print(f"Training on {args.nb_classes} {args.category} classes: {class_names}")


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

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train_balanced,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    model = models_tor_mamba.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        byte_length=args.byte_length,
        # stride_size=args.stride_size
    )

    if args.finetune and not args.eval:
        checkpoint = torch.load(args.finetune, map_location='cpu')
        print("Load pre-trained checkpoint from: %s" % args.finetune)
        if 'model' in checkpoint:
            checkpoint_model = checkpoint['model']
        else:
            checkpoint_model = checkpoint
            

        checkpoint_model = {k: v for k, v in checkpoint_model.items() if not k.startswith('head')}
        
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)


        if hasattr(model, 'head') and hasattr(model.head, 'weight'):
            trunc_normal_(model.head.weight, std=2e-5)

    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    if num_tasks > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # build optimizer with layer-wise lr decay (lrd)
    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
        no_weight_decay_list=model_without_ddp.no_weight_decay(),
        layer_decay=args.layer_decay
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)


    amp_autocast = suppress
    loss_scaler = "none"
    if not args.no_amp:
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()

    if args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        test_results = detailed_evaluate(data_loader_test, model, device, class_names, 
                                       epoch=0, save_dir=args.output_dir)
        print("=== Test Results ===")
        print(f"Accuracy: {test_results['accuracy']:.2f}%")
        print(f"Macro F1: {test_results['macro_f1']:.2f}%")
        print(f"Micro F1: {test_results['micro_f1']:.2f}%")
        

        with open(os.path.join(args.output_dir, "test_results.json"), "w") as f:
            json.dump(test_results, f, indent=2)
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    max_f1 = 0.0
    best_results = None

    for epoch in range(args.start_epoch, args.epochs):
        if num_tasks > 1:
            data_loader_train.sampler.set_epoch(epoch)
            
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler, amp_autocast,
            args.clip_grad, None,  # no mixup
            log_writer=log_writer,
            args=args
        )
        

        val_results = detailed_evaluate(data_loader_val, model, device, class_names, 
                                      epoch=epoch, save_dir=args.output_dir)
        
        print(f"\n=== Epoch {epoch} Validation Results on VALIDATION SET: {args.data_path}/valid ===")
        print(f"Accuracy: {val_results['accuracy']:.2f}%")
        print(f"Macro Precision: {val_results['macro_precision']:.2f}%")
        print(f"Macro Recall: {val_results['macro_recall']:.2f}%")
        print(f"Macro F1: {val_results['macro_f1']:.2f}%")
        print(f"Micro F1: {val_results['micro_f1']:.2f}%")
        

        print("\nPer-class metrics:")
        for class_name in class_names:
            metrics = val_results['per_class_metrics'][class_name]
            print(f"  {class_name}: P={metrics['precision']:.1f}% R={metrics['recall']:.1f}% F1={metrics['f1']:.1f}% (n={metrics['support']})")
        

        if val_results['macro_f1'] > max_f1:
            max_f1 = val_results['macro_f1']
            max_accuracy = val_results['accuracy']
            best_results = val_results
            
            if args.output_dir:

                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch)
                

                with open(os.path.join(args.output_dir, "best_val_results.json"), "w") as f:
                    json.dump(val_results, f, indent=2)

                src_ckpt = os.path.join(args.output_dir, f"checkpoint-{epoch}.pth")
                dst_ckpt = os.path.join(args.output_dir, "checkpoint-best.pth")
                if os.path.exists(src_ckpt):
                    shutil.copy(src_ckpt, dst_ckpt)
                print(f"Best model updated and saved to {dst_ckpt}")

        print(f'Best F1: {max_f1:.2f}%, Best Accuracy: {max_accuracy:.2f}%')


        if log_writer is not None:
            log_writer.add_scalar('val/accuracy', val_results['accuracy'], epoch)
            log_writer.add_scalar('val/macro_f1', val_results['macro_f1'], epoch)
            log_writer.add_scalar('val/macro_precision', val_results['macro_precision'], epoch)
            log_writer.add_scalar('val/macro_recall', val_results['macro_recall'], epoch)
            log_writer.add_scalar('val/loss', val_results['loss'], epoch)


        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'val_{k}': v for k, v in val_results.items() if k not in ['per_class_metrics', 'confusion_matrix', 'classification_report']},
            'epoch': epoch,
            'n_parameters': n_parameters
        }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")


    print("\n=== Final Test Evaluation ===")
    model_path = os.path.join(args.output_dir, "checkpoint-best.pth")
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        print("Loaded best model for final test")
    
    test_results = detailed_evaluate(data_loader_test, model, device, class_names, 
                                   epoch="final", save_dir=args.output_dir)
    
    print(f"Final Test Accuracy: {test_results['accuracy']:.2f}%")
    print(f"Final Test Macro F1: {test_results['macro_f1']:.2f}%")
    print(f"Final Test Micro F1: {test_results['micro_f1']:.2f}%")
    
    print("\nFinal Per-class Test Results:")
    for class_name in class_names:
        metrics = test_results['per_class_metrics'][class_name]
        print(f"  {class_name}: P={metrics['precision']:.1f}% R={metrics['recall']:.1f}% F1={metrics['f1']:.1f}% (n={metrics['support']})")
    

    with open(os.path.join(args.output_dir, "final_test_results.json"), "w") as f:
        json.dump(test_results, f, indent=2)


    generate_final_report(best_results, test_results, class_names, args.output_dir)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

def generate_final_report(val_results, test_results, class_names, output_dir):
    report = {
        "model_info": {
            "classes": class_names,
            "num_classes": len(class_names)
        },
        "validation_results": val_results,
        "test_results": test_results,
        "summary": {
            "val_accuracy": val_results['accuracy'],
            "val_macro_f1": val_results['macro_f1'],
            "test_accuracy": test_results['accuracy'],
            "test_macro_f1": test_results['macro_f1'],
            "performance_drop": {
                "accuracy": val_results['accuracy'] - test_results['accuracy'],
                "f1": val_results['macro_f1'] - test_results['macro_f1']
            }
        }
    }
    
    with open(os.path.join(output_dir, "training_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    

    markdown_report = f"""# NetMamba Training Report - OBFS Classes

## Model Information
- Classes: {class_names}
- Number of classes: {len(class_names)}

## Results Summary

### Validation Results (Best Model)
- **Accuracy**: {val_results['accuracy']:.2f}%
- **Macro Precision**: {val_results['macro_precision']:.2f}%
- **Macro Recall**: {val_results['macro_recall']:.2f}%
- **Macro F1**: {val_results['macro_f1']:.2f}%

### Test Results (Final)
- **Accuracy**: {test_results['accuracy']:.2f}%
- **Macro Precision**: {test_results['macro_precision']:.2f}%
- **Macro Recall**: {test_results['macro_recall']:.2f}%
- **Macro F1**: {test_results['macro_f1']:.2f}%

## Per-Class Results

### Test Set Performance
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
"""
    
    for class_name in class_names:
        metrics = test_results['per_class_metrics'][class_name]
        markdown_report += f"| {class_name} | {metrics['precision']:.1f}% | {metrics['recall']:.1f}% | {metrics['f1']:.1f}% | {metrics['support']} |\n"
    
    with open(os.path.join(output_dir, "training_report.md"), "w") as f:
        f.write(markdown_report)
    
    print(f"\nDetailed reports saved to {output_dir}")

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)