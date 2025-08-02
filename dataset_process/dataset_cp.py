import os
import shutil
from tqdm import tqdm

base_dir = "classified_array_bridge_0717_dataset_sampled"
splits = ["train", "valid"]
test_dir = os.path.join(base_dir, "test")

for split in splits:
    split_dir = os.path.join(base_dir, split)
    if not os.path.exists(split_dir):
        continue
    for class_name in os.listdir(split_dir):
        class_dir = os.path.join(split_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        test_class_dir = os.path.join(test_dir, class_name)
        os.makedirs(test_class_dir, exist_ok=True)
        for fname in tqdm(os.listdir(class_dir), desc=f"{split}/{class_name}"):
            src_file = os.path.join(class_dir, fname)
            dst_file = os.path.join(test_class_dir, fname)
            if os.path.isfile(src_file):
                shutil.copy2(src_file, dst_file)

