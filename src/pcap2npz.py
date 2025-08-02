import os
import numpy as np
from scapy.all import rdpcap
from tqdm import tqdm
import argparse

def extract_trace(pcap_path, max_len=10000):
    try:
        packets = rdpcap(pcap_path)
    except Exception as e:
        print(f"[!] Failed to read {pcap_path}: {e}")
        return None

    base_time = None
    sequence = []
    for pkt in packets:
        if 'IP' not in pkt or 'TCP' not in pkt:
            continue
        if not hasattr(pkt, 'time'):
            continue
        direction = 1 if pkt['IP'].src == '172.31.21.212' or pkt['IP'].src == '172.31.7.243' else -1
        if base_time is None:
            base_time = pkt.time
        rel_time = pkt.time - base_time
        sequence.append(direction * rel_time)
    trace = np.array(sequence[:max_len], dtype=np.float64)
    if len(trace) < max_len:
        trace = np.pad(trace, (0, max_len - len(trace)), 'constant')
    return trace

def get_site_label(fname):
    fname_lower = fname.lower()
    if "news" in fname_lower:
        return "news"
    elif "video" in fname_lower:
        return "video"
    else:
        return "image"

def process_all_mode(root_dir, out_file, max_len=10000):
    traces = []
    labels = []
    label_map = {"image": 0, "news": 1, "video": 2}
    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        for fname in tqdm(os.listdir(folder_path), desc=folder):
            if not fname.endswith('.pcap'):
                continue
            full_path = os.path.join(folder_path, fname)
            trace = extract_trace(full_path, max_len=max_len)
            if trace is not None:
                site = get_site_label(fname)
                traces.append(trace)
                labels.append(label_map[site])
    if traces:
        X = np.stack(traces)
        y = np.array(labels, dtype=np.int64)
        np.savez(out_file, X=X, y=y)
        print(f"[✓] Saved: {out_file} | X.shape={X.shape} | y.shape={y.shape}")

def process_single_mode(root_dir, out_dir, max_len=10000):
    label_map = {"image": 0, "news": 1, "video": 2}
    os.makedirs(out_dir, exist_ok=True)
    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        traces = []
        labels = []
        for fname in tqdm(os.listdir(folder_path), desc=folder):
            if not fname.endswith('.pcap'):
                continue
            full_path = os.path.join(folder_path, fname)
            trace = extract_trace(full_path, max_len=max_len)
            if trace is not None:
                site = get_site_label(fname)
                traces.append(trace)
                labels.append(label_map[site])
        if traces:
            X = np.stack(traces)
            y = np.array(labels, dtype=np.int64)
            out_file = os.path.join(out_dir, f"{folder}.npz")
            np.savez(out_file, X=X, y=y)
            print(f"[✓] Saved: {out_file} | X.shape={X.shape} | y.shape={y.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert pcap to npz for WF dataset.")
    parser.add_argument('--root_dir', type=str, required=True, help="Input dir: datset2npz/exit_allcls/exit_matched")
    parser.add_argument('--mode', type=str, choices=['all', 'single'], required=True, help="Processing mode: all or single")
    parser.add_argument('--out_file', type=str, default='all_cls.npz', help="Output npz file for all mode")
    parser.add_argument('--out_dir', type=str, default='.', help="Output dir for single mode")
    parser.add_argument('--max_len', type=int, default=10000, help="Max sequence length")
    args = parser.parse_args()

    if args.mode == 'all':
        process_all_mode(args.root_dir, args.out_file, max_len=args.max_len)
    else:
        process_single_mode(args.root_dir, args.out_dir, max_len=args.max_len)