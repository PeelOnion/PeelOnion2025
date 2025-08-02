import argparse
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from src.models_tor_mamba import net_mamba_classifier
import os
import json
from collections import defaultdict

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to test dataset')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./output/test_results', help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--input_size', type=int, default=40)
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()

def load_model(checkpoint_path, num_classes, device):
    model = net_mamba_classifier(num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    return model

def extract_subclass(filename):


    basename = os.path.basename(filename)
    parts = basename.split('_')
    if len(parts) >= 2:
        return parts[1]
    else:
        return "unknown"

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"start testing model with checkpoint: {args.checkpoint}")

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(args.data_path, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    class_names = dataset.classes
    num_classes = len(class_names)
    print(f"Test classes: {class_names}")


    model = load_model(args.checkpoint, num_classes, args.device)

    all_preds = []
    all_labels = []


    subclass_stats = defaultdict(lambda: defaultdict(int))
    subclass_total = defaultdict(int)

    with torch.no_grad():
        sample_idx = 0
        for imgs, labels in loader:
            batch_size = imgs.size(0)
            batch_paths = [dataset.samples[i][0] for i in range(sample_idx, sample_idx + batch_size)]
            sample_idx += batch_size
            imgs = imgs.to(args.device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

            for path, gt, pred in zip(batch_paths, labels.numpy(), preds):
                subclass = extract_subclass(path)
                major_class = class_names[gt]
                key = f"{major_class}-{subclass}"
                subclass_stats[key][class_names[pred]] += 1
                subclass_total[key] += 1


    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    cm = confusion_matrix(all_labels, all_preds)
    print(json.dumps(report, indent=2))


    print("Saving test report and confusion matrix...")
    with open(os.path.join(args.output_dir, 'test_report.json'), 'w') as f:
        json.dump(report, f, indent=2)


    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'confusion_matrix.png'))
    plt.close()


    print("\n=== MajorClass-Subclass Prediction Distribution ===")
    subclass_table = {}
    for key, pred_dict in subclass_stats.items():
        row = {cls: pred_dict.get(cls, 0) for cls in class_names}
        row['total'] = subclass_total[key]
        subclass_table[key] = row
        print(f"{key:20s} | " + " | ".join([f"{cls}:{row[cls]}" for cls in class_names]) + f" | total:{row['total']}")

    with open(os.path.join(args.output_dir, 'majorclass_subclass_stats.json'), 'w') as f:
        json.dump(subclass_table, f, indent=2)

    print(f"\nTest results, confusion matrix, and subclass stats saved to {args.output_dir}")

if __name__ == '__main__':
    main()