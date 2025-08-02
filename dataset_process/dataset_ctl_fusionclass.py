import os
from tqdm import tqdm
from dataset_common import find_files, read_5hp_list
import subprocess
import json
from PIL import Image
import random
import numpy as np
import shutil
import argparse
import multiprocessing
import concurrent.futures


def sample_pcap(minimum=200, maximum=30000, input_dir="CICIoT2022/flows/1-Power",
                output_dir="CICIoT2022/flows_sampled/1-Power", if_cic=False):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    random.seed(0)
    os.makedirs(output_dir, exist_ok=True)
    

    if not os.path.exists(input_dir):
        print(f"Input directory {input_dir} does not exist, skipping...")
        return
    
    sub_dirs = list(filter(lambda x: os.path.isdir(f"{input_dir}/{x}"), os.listdir(input_dir)))
    if if_cic:
        pcap_files = []
        for sub_dir in sub_dirs:
            pcap_files.extend(find_files(f"{input_dir}/{sub_dir}", extension=".pcap"))
        if len(pcap_files) < minimum: # Skip if less than minimum
            print(f"Skip {input_dir} due to less than {minimum} flows")
            return
        if len(pcap_files) > maximum:
            pcap_files = random.sample(pcap_files, maximum)
        for pcap_file in tqdm(pcap_files, desc=output_dir):
            compressed_name = pcap_file[len(input_dir):].replace("/", "_")
            dst_pcap_file = f"{output_dir}/{compressed_name}"
            os.makedirs(output_dir, exist_ok=True)
            subprocess.run(f"cp '{pcap_file}' '{dst_pcap_file}'", shell=True)
    else:
        for sub_dir in sub_dirs:
            pcap_files = find_files(f"{input_dir}/{sub_dir}", extension=".pcap")
            if len(pcap_files) < minimum: # Skip if less than minimum
                print(f"Skip {input_dir}/{sub_dir} due to less than {minimum} flows")
                continue
            if len(pcap_files) > maximum:
                pcap_files = random.sample(pcap_files, maximum)
            for pcap_file in tqdm(pcap_files, desc=f"{output_dir}/{sub_dir}"):
                dst_pcap_file = pcap_file.replace(input_dir, output_dir)
                os.makedirs("/".join(dst_pcap_file.split("/")[:-1]), exist_ok=True)
                subprocess.run(f"cp '{pcap_file}' '{dst_pcap_file}'", shell=True)

def sample_custom_pcap(input_dir="custom_flows", output_dir="Custom/flows_sampled", 
                      minimum=0, maximum=30000):
    if not os.path.exists(input_dir):
        print(f"Custom input directory {input_dir} does not exist, skipping...")
        return
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    random.seed(42)
    os.makedirs(output_dir, exist_ok=True)
    

    pcap_files = find_files(input_dir, extension=".pcap")
    
    if len(pcap_files) < minimum:
        print(f"Skip {input_dir} due to less than {minimum} flows ({len(pcap_files)} found)")
        return
    
    if len(pcap_files) > maximum:
        pcap_files = random.sample(pcap_files, maximum)
    
    print(f"Processing {len(pcap_files)} custom pcap files...")
    

    file_categories = {}
    for pcap_file in pcap_files:
        filename = os.path.basename(pcap_file)

        if "_" in filename:
            category = filename.split("_")[0]
        else:
            category = filename.split(".")[0]
        
        if category not in file_categories:
            file_categories[category] = []
        file_categories[category].append(pcap_file)
    

    for category, files in file_categories.items():
        category_dir = f"{output_dir}/{category}"
        os.makedirs(category_dir, exist_ok=True)
        
        for pcap_file in tqdm(files, desc=f"Processing {category}"):
            filename = os.path.basename(pcap_file)
            dst_pcap_file = f"{category_dir}/{filename}"
            subprocess.run(f"cp '{pcap_file}' '{dst_pcap_file}'", shell=True)
    
    print(f"Custom dataset processed: {file_categories}")

def sample_all_pcap():

    sample_pcap(input_dir="CICIoT2022/flows/1-Power/Audio", output_dir=f"CICIoT2022/flows_sampled/Audio",
                    minimum=200, maximum=6000, if_cic=True)
    sample_pcap(input_dir="CICIoT2022/flows/1-Power/Cameras", output_dir=f"CICIoT2022/flows_sampled/Cameras",
                    minimum=200, maximum=6000, if_cic=True)
    sample_pcap(input_dir="CICIoT2022/flows/1-Power/Home Automation", output_dir=f"CICIoT2022/flows_sampled/Home Automation",
                    minimum=200, maximum=6000, if_cic=True)
    sample_pcap(input_dir="CICIoT2022/flows/6-Attacks/1-Flood", output_dir=f"CICIoT2022/flows_sampled/Flood",
                    minimum=200, maximum=6000, if_cic=True)
    sample_pcap(input_dir="CICIoT2022/flows/6-Attacks/2-RTSP Brute Force/Hydra", 
                output_dir=f"CICIoT2022/flows_sampled/Hydra", minimum=200, maximum=6000, if_cic=True)
    sample_pcap(input_dir="CICIoT2022/flows/6-Attacks/2-RTSP Brute Force/Nmap", 
                output_dir=f"CICIoT2022/flows_sampled/Nmap", minimum=200, maximum=6000, if_cic=True)
    sample_pcap(input_dir="CrossPlatform/flows/android", output_dir=f"CrossPlatform-Android/flows_sampled",
                    minimum=50, maximum=2000,)
    sample_pcap(input_dir="CrossPlatform/flows/ios", output_dir=f"CrossPlatform-iOS/flows_sampled",
                    minimum=50, maximum=2000,)
    sample_pcap(input_dir="/mnt/ssd1/ISCXVPN2016/flows", output_dir=f"ISCXVPN2016/flows_sampled",
                    minimum=500, maximum=4000,)
    sample_pcap(input_dir="/mnt/ssd1/USTC-TFC2016/flows", output_dir=f"USTC-TFC2016/flows_sampled",
                    minimum=500, maximum=2000,)
    sample_pcap(input_dir="/mnt/ssd1/ISCXTor2016/flows", output_dir=f"ISCXTor2016/flows_sampled",
                    minimum=10, maximum=4000,)
    

    sample_custom_pcap(input_dir="custom_flows", output_dir="Custom/flows_sampled", 
                      minimum=50, maximum=30000)

# def safe_read_5hp_list(pcap_path, if_augment=False, timeout=10):
#     with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
#         future = executor.submit(read_5hp_list, pcap_path, if_augment)
#         try:
#             return future.result(timeout=timeout)
#         except Exception as e:
#             print(f"Timeout or error in read_5hp_list for {pcap_path}: {e}")
#             return None

def pcap_to_array(pcap_dir, if_augment=False):
    assert pcap_dir.split("/")[-1] == "flows_sampled"
    image_dir = pcap_dir.replace("flows_sampled", "array_sampled")
    
    if not os.path.exists(pcap_dir):
        print(f"Directory {pcap_dir} does not exist, skipping...")
        return
    
    flow_dir_names = os.listdir(pcap_dir)
    for flow_dir_name in flow_dir_names:
        os.makedirs(f"{image_dir}/{flow_dir_name}", exist_ok=True)
        pcap_filenames = os.listdir(f"{pcap_dir}/{flow_dir_name}")
        for pcap_filename in tqdm(pcap_filenames, desc=flow_dir_name):
            try:
                pcap_path = f"{pcap_dir}/{flow_dir_name}/{pcap_filename}"
                res_list = read_5hp_list(pcap_path, local_ip="0.0.0.0", if_augment=if_augment)
                if res_list is None:
                    continue
                if not if_augment:
                    if len(res_list) > 0:
                        res = res_list[0]
                        image_filename = f"{image_dir}/{flow_dir_name}/{pcap_filename[:-len('.pcap')]}.png"
                        stat_filename = image_filename.replace(".png", ".json")
                        flow_array = res.pop("data")
                        # image = Image.fromarray(flow_array.reshape(21, 21).astype(np.uint8))
                        image = Image.fromarray((flow_array.reshape(21, 21) * 255).astype(np.uint8))
                        image.save(image_filename)
                        with open(stat_filename, "w") as f:
                            json.dump(res, f)
                else:
                    for i, res in enumerate(res_list):
                        image_filename = f"{image_dir}/{flow_dir_name}/{pcap_filename[:-len('.pcap')]}-{i}.png"
                        stat_filename = image_filename.replace(".png", ".json")
                        flow_array = res.pop("data")
                        image = Image.fromarray(flow_array.reshape(21, 21).astype(np.uint8))
                        image.save(image_filename)
                        with open(stat_filename, "w") as f:
                            json.dump(res, f)
            except Exception as e:
                print(f"Error processing {pcap_filename}: {e}")
                import traceback
                traceback.print_exc()

def process_class_dir(args):
    src_dir, dst_dir, image_size, if_augment = args
    os.makedirs(dst_dir, exist_ok=True)
    for fname in os.listdir(src_dir):
        if not fname.endswith('.pcap'):
            continue
        pcap_path = os.path.join(src_dir, fname)
        try:
            res_list = read_5hp_list(pcap_path, if_augment=if_augment)
            if res_list is None or len(res_list) == 0:
                continue
            if not if_augment:
                res = res_list[0]
                flow_array = res.pop("data")
                image_filename = os.path.join(dst_dir, fname.replace('.pcap', '.png'))
                stat_filename = image_filename.replace('.png', '.json')
                image = Image.fromarray(flow_array.reshape(image_size, image_size).astype('uint8'))
                image.save(image_filename)
                with open(stat_filename, "w") as f:
                    json.dump(res, f)
            else:
                for i, res in enumerate(res_list):
                    flow_array = res.pop("data")
                    image_filename = os.path.join(dst_dir, fname.replace('.pcap', f'-{i}.png'))
                    stat_filename = image_filename.replace('.png', '.json')
                    image = Image.fromarray(flow_array.reshape(image_size, image_size).astype('uint8'))
                    image.save(image_filename)
                    with open(stat_filename, "w") as f:
                        json.dump(res, f)
        except Exception as e:
            print(f"Error processing {pcap_path}: {e}")

def pcaps_to_arrays_flat(input_dir, output_dir, image_size=40, if_augment=False):
    os.makedirs(output_dir, exist_ok=True)
    for fname in tqdm(os.listdir(input_dir), desc="Processing pcaps"):
        if not fname.endswith('.pcap'):
            continue
        pcap_path = os.path.join(input_dir, fname)
        try:
            res_list = read_5hp_list(pcap_path, if_augment=if_augment)
            if res_list is None or len(res_list) == 0:
                continue
            if not if_augment:
                res = res_list[0]
                flow_array = res.pop("data")
                image_filename = os.path.join(output_dir, fname.replace('.pcap', '.png'))
                stat_filename = image_filename.replace('.png', '.json')
                image = Image.fromarray(flow_array.reshape(image_size, image_size).astype('uint8'))
                image.save(image_filename)
                with open(stat_filename, "w") as f:
                    json.dump(res, f)
            else:
                for i, res in enumerate(res_list):
                    flow_array = res.pop("data")
                    image_filename = os.path.join(output_dir, fname.replace('.pcap', f'-{i}.png'))
                    stat_filename = image_filename.replace('.png', '.json')
                    image = Image.fromarray(flow_array.reshape(image_size, image_size).astype('uint8'))
                    image.save(image_filename)
                    with open(stat_filename, "w") as f:
                        json.dump(res, f)
        except Exception as e:
            print(f"Error processing {pcap_path}: {e}")

def pcaps_to_arrays_by_class(input_dir, output_dir, image_size=21, if_augment=False, num_workers=8):
    os.makedirs(output_dir, exist_ok=True)
    class_dirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    args_list = []
    for class_dir in class_dirs:
        src_dir = os.path.join(input_dir, class_dir)
        dst_dir = os.path.join(output_dir, class_dir)
        args_list.append((src_dir, dst_dir, image_size, if_augment))
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        list(tqdm(executor.map(process_class_dir, args_list), total=len(args_list), desc="Processing classes (multi-process)"))

def process_bridge_hop_site_dataset(input_dir, 
                                   array_dir, 
                                   dataset_dir, 
                                   image_size=40):


    pcaps_to_arrays_by_class(input_dir, array_dir, image_size=image_size, if_augment=False)

    split_dataset(array_dir, train_ratio=0.8, valid_ratio=0.1)

def all_pcap_to_array():
    pcap_to_array("CICIoT2022/flows_sampled", if_augment=False)
    pcap_to_array("CrossPlatform-Android/flows_sampled", if_augment=True)
    pcap_to_array("CrossPlatform-iOS/flows_sampled", if_augment=True)
    pcap_to_array("ISCXVPN2016/flows_sampled", if_augment=False)
    pcap_to_array("USTC-TFC2016/flows_sampled", if_augment=False)
    pcap_to_array("ISCXTor2016/flows_sampled", if_augment=True)
    

    pcap_to_array("Custom/flows_sampled", if_augment=True)

def split_dataset(input_dir, output_dir=None, train_ratio=0.8, valid_ratio=0.1):
    if output_dir is None:
        dir_name = "dataset_sampled"
        output_dir = input_dir.replace("array_sampled", dir_name)
        if output_dir == input_dir:
            output_dir = input_dir + "_dataset_sampled"
    train_dir = f"{output_dir}/train"
    valid_dir = f"{output_dir}/valid"
    test_dir = f"{output_dir}/test"
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    if not os.path.exists(input_dir):
        print(f"Directory {input_dir} does not exist, skipping...")
        return

    random.seed(42)
    np.random.seed(42)
    filenames = find_files(input_dir, extension=".png")
    
    if len(filenames) == 0:
        print(f"No PNG files found in {input_dir}")
        return
    
    random.shuffle(filenames)
    train_size = int(len(filenames) * train_ratio)
    valid_size = int(len(filenames) * valid_ratio)
    train_files = filenames[:train_size]
    valid_files = filenames[train_size:train_size+valid_size]
    test_files = filenames[train_size+valid_size:]
    
    for filename in tqdm(filenames, desc="Splitting"):
        if filename in train_files:
            split_type = "train"
        elif filename in valid_files:
            split_type = "valid"
        else:
            split_type = "test"
        
        filename_parts = filename.replace("\\", "/").split("/")
        label = filename_parts[-2]
        base_name = filename_parts[-1]
        
        os.makedirs(f"{output_dir}/{split_type}/{label}", exist_ok=True)
        subprocess.run(f"cp '{filename}' '{output_dir}/{split_type}/{label}/{base_name}'", shell=True)
    
    print(f"Train: {len(train_files)}, Valid: {len(valid_files)}, Test: {len(test_files)}")

def split_all_datasets():
    datasets = ["CICIoT2022", "CrossPlatform-Android", "CrossPlatform-iOS", "ISCXVPN2016", "USTC-TFC2016", "ISCXTor2016", "Custom"]
    for dataset in datasets:
        split_dataset(f"{dataset}/array_sampled")

def merge_dataset():
    filenames = []
    datasets = ["CICIoT2022", "CrossPlatform-Android", "CrossPlatform-iOS", "ISCXVPN2016", "USTC-TFC2016", "ISCXTor2016", "Custom"]
    for dataset in datasets:
        dataset_path = f"{dataset}/array_sampled"
        if os.path.exists(dataset_path):
            filenames += find_files(dataset_path, extension=".png")
        else:
            print(f"Dataset {dataset} not found, skipping...")
    
    for filename in tqdm(filenames, desc="Merging"):
        filename_list = filename.split("/")
        label = filename_list[0] + "-" + filename_list[-2]
        base_name = filename_list[-1]
        os.makedirs(f"pretrain_dataset/train/{label}", exist_ok=True)
        dst_filename = f"pretrain_dataset/train/{label}/{base_name}"
        subprocess.run(f"cp '{filename}' '{dst_filename}'", shell=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true", help="Sample pcap files")
    parser.add_argument("--array", action="store_true", help="Convert pcap files to array")
    parser.add_argument("--split", action="store_true", help="Split dataset into train, valid, and test sets for finetuning")
    parser.add_argument("--merge", action="store_true", help="Merge all datasets into one for pretraining")
    parser.add_argument("--all", action="store_true", help="Run all steps")
    parser.add_argument("--custom", action="store_true", help="Process only custom dataset")
    parser.add_argument("--custom_dir", default="custom_flows", help="Directory containing custom pcap files")
    parser.add_argument("--bridge_hop_site", action="store_true", help="Process classified_by_bridge_hop_site dataset")
    parser.add_argument("--bridge_hop_site_dir", default="classified_by_bridge_hop_site", help="Directory for bridge_hop_site pcaps")
    args = parser.parse_args()

    if args.custom:
        sample_custom_pcap(input_dir=args.custom_dir, output_dir="Custom_0709_21_noobfs_test/flows_sampled")
        pcap_to_array("Custom_0709_21_noobfs_test/flows_sampled", if_augment=False)
        split_dataset("Custom_0709_21_noobfs_test/array_sampled")
    elif args.sample or args.all:
        sample_all_pcap()
    
    if args.array or args.all:
        all_pcap_to_array()
    if args.split or args.all:
        split_all_datasets()
    if args.merge or args.all:
        merge_dataset()
    if args.bridge_hop_site:

        if any(os.path.isdir(os.path.join(args.bridge_hop_site_dir, d)) for d in os.listdir(args.bridge_hop_site_dir)):

            process_bridge_hop_site_dataset(
                input_dir=args.bridge_hop_site_dir,
                array_dir="array_exit_measuretest_0730",
                dataset_dir="datasets_exit_measuretest_0730",
                image_size=40
            )
        else:

            pcaps_to_arrays_flat(
                input_dir=args.bridge_hop_site_dir,
                output_dir="array_exit_measuretest_0730",
                image_size=40,
                if_augment=False
            )
            split_dataset("array_exit_measuretest_0730", train_ratio=0.8, valid_ratio=0.1)