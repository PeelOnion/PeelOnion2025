import numpy as np
import binascii
import scapy.all as scapy
import os

def find_files(data_path, extension=".pcap"):
    pcap_files = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith(extension):
                pcap_files.append(os.path.join(root, file))
    return pcap_files

def raw_packet_to_string(packet, remove_ip=True):
    ip = packet["IP"]
    if remove_ip:
        PAD_IP_ADDR = "0.0.0.0"
        ip.src, ip.dst = PAD_IP_ADDR, PAD_IP_ADDR
    header = (binascii.hexlify(bytes(ip))).decode()

    header = header[:156] if len(header) > 156 else header + '0' * (156 - len(header))
    return header

def string_to_hex_array(flow_string):
    return np.array([int(flow_string[i:i + 2], 16) for i in range(0, len(flow_string), 2)])

def read_5hp_list(pcap_filename, local_ip, if_augment=False, remove_ip=True):
    try:
        packets = scapy.rdpcap(pcap_filename)
    except Exception as e:
        print(f"Failed to read {pcap_filename}: {e}")
        return []
    data = []
    flow_packet_num = 5
    feature_len = 86  # 78(header) + 8(extended)
    flow_array_len = 441  # 21x21
    results = []

    def extract_feature(packet, base_time):
        try:
            header = raw_packet_to_string(packet, remove_ip=remove_ip)
            relative_time = packet.time - base_time
            ip = packet['IP']
            direction = 1 if ip.src == local_ip else -1
            ttl = ip.ttl
            if packet.haslayer('TCP'):
                flags = int(packet['TCP'].flags)
            else:
                flags = 0
        except:
            header = '0' * 156
            relative_time = 0.0
            direction = 0
            ttl = 0
            flags = 0
        feature = string_to_hex_array(header).astype(np.float32)
        extended = [ttl] * 4 + [flags] * 4
        feature = np.append(feature, extended)
        feature = feature / 255.0
        return feature, relative_time, direction

    if not if_augment or len(packets) <= flow_packet_num:

        base_time = packets[0].time if len(packets) > 0 else 0
        data = []
        time_list = []
        direction_list = []
        for packet in packets[:flow_packet_num]:
            feature, relative_time, direction = extract_feature(packet, base_time)
            data.append(feature)
            time_list.append(relative_time)
            direction_list.append(direction)
        while len(data) < flow_packet_num:
            data.append(np.zeros(feature_len, dtype=np.float32))
            time_list.append(0.0)
            direction_list.append(0)
        flow_static = np.concatenate(data)
        time_array = np.array(time_list, dtype=np.float32)
        time_array = time_array / (time_array.max() + 1e-9)
        direction_array = (np.array(direction_list, dtype=np.float32) + 1) / 2.0
        flow_tail = np.concatenate([time_array, direction_array])
        flow_array = np.concatenate([flow_static, flow_tail])
        if len(flow_array) < flow_array_len:
            flow_array = np.append(flow_array, np.zeros(flow_array_len - len(flow_array), dtype=flow_array.dtype))
        elif len(flow_array) > flow_array_len:
            flow_array = flow_array[:flow_array_len]
        return [{"data": flow_array}]
    else:

        for i in range(len(packets) - flow_packet_num + 1):
            base_time = packets[i].time
            data = []
            time_list = []
            direction_list = []
            for j in range(flow_packet_num):
                idx = i + j
                if idx < len(packets):
                    feature, relative_time, direction = extract_feature(packets[idx], base_time)
                else:
                    feature = np.zeros(feature_len, dtype=np.float32)
                    relative_time = 0.0
                    direction = 0
                data.append(feature)
                time_list.append(relative_time)
                direction_list.append(direction)
            flow_static = np.concatenate(data)
            time_array = np.array(time_list, dtype=np.float32)
            time_array = time_array / (time_array.max() + 1e-9)
            direction_array = (np.array(direction_list, dtype=np.float32) + 1) / 2.0
            flow_tail = np.concatenate([time_array, direction_array])
            flow_array = np.concatenate([flow_static, flow_tail])
            if len(flow_array) < flow_array_len:
                flow_array = np.append(flow_array, np.zeros(flow_array_len - len(flow_array), dtype=flow_array.dtype))
            elif len(flow_array) > flow_array_len:
                flow_array = flow_array[:flow_array_len]
            results.append({"data": flow_array})
        return results