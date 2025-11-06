"""
Extract Xsens data from .mvnx files to .pt files.

Adapted from:
https://github.com/dx118/dynaip/blob/main/datasets/extract.py
"""
import os
import torch
from utils.xsens_util import read_mvnx


def extract_mvnx(xsens_raw_dataset_dir, xsens_extract_dir):
    datasets = ['AnDy', 'UNIPD', 'Emokine', 'CIP', 'Virginia']
    for dataset in datasets:
        data_folder = os.path.join(xsens_raw_dataset_dir, dataset)
        mvnx_files = [os.path.relpath(os.path.join(foldername, filename), data_folder)
                      for foldername, _, filenames in os.walk(data_folder)
                      for filename in filenames if filename.endswith('.mvnx')]
        assert mvnx_files != [], f"No .mvnx files found in {os.path.join(xsens_raw_dataset_dir, dataset)}"
        for f in mvnx_files:
            f = os.path.join(data_folder, f)
            print(f)
            try:
                data = read_mvnx(f)
            except Exception as e:
                print("An error occurred when calling read_mvnx(f). The error is: ", e)
                print(f"Skipping this file...{f}")
                continue
            if dataset == 'AnDy':
                f = f.replace('\\', '/').replace('.xsens.mvnx', '.pt')
            else:
                f = f.replace('\\', '/').replace('.mvnx', '.pt')
            print('saving:', f.split('/')[-1])
            dataset = 'virginia_temp' if dataset == 'Virginia' else dataset
            out_dir = os.path.join(xsens_extract_dir, dataset, f.split('/')[-1])
            os.makedirs(os.path.join(xsens_extract_dir, dataset), exist_ok=True)
            torch.save(data, out_dir)

    # we manually picked a part of data which has no visible drift from original virginia-natural-motion dataset
    # you can visualize the extracted virginia-natural-motion data and select clean clips of your own
    clip_config = [
        {'name': 'P1_Day_1_1', 'start': [0], 'end': [-1]},
        {'name': 'P1_Day_1_3', 'start': [21800, 35800], 'end': [27000, 71800]},
        {'name': 'P2_Day_1_1', 'start': [0, 82000], 'end': [69000, -1]},
        {'name': 'P3_Day_1_1', 'start': [0], 'end': [50000]},
        {'name': 'P3_Day_1_2', 'start': [0], 'end': [-1]},
        {'name': 'P4_Day_1_1', 'start': [0], 'end': [18000]},
        {'name': 'P4_Day_1_2', 'start': [0], 'end': [43000]},
        {'name': 'P4_Day_1_3', 'start': [0], 'end': [18000]},
        {'name': 'P5_Day_1_1', 'start': [16000], 'end': [34000]},
        {'name': 'P6_Day_2_1', 'start': [80000], 'end': [110000]},
        {'name': 'P10_Day_1_1', 'start': [82000], 'end': [100000]},
        {'name': 'P11_Day_1_2', 'start': [31000, 44000, 206300, 229300], 'end': [33500, 51800, 210300, 238300]},
        {'name': 'P13_Day_2_1', 'start': [0], 'end': [-1]},
        {'name': 'P13_Day_2_2', 'start': [0], 'end': [22500]},
    ]

    for d in clip_config:
        name = d['name']
        start_indices = d['start']
        end_indices = d['end']

        data = torch.load(os.path.join(xsens_extract_dir, 'virginia_temp', name + '.pt'))
        for i, (start, end) in enumerate(zip(start_indices, end_indices)):
            out = {'joint': {'orientation': [], 'position': []},
                   'imu': {'free acceleration': [], 'calibrated orientation': []}
                   }
            out['joint']['orientation'] = data['joint']['orientation'][start:end].float()
            out['joint']['position'] = data['joint']['position'][start:end].float()
            out['imu']['free acceleration'] = data['imu']['free acceleration'][start:end].float()
            out['imu']['calibrated orientation'] = data['imu']['calibrated orientation'][start:end].float()
            print('saving:', '{}_{}'.format(name, i))
            os.makedirs(os.path.join(xsens_extract_dir, 'Virginia'), exist_ok=True)
            torch.save(out, os.path.join(xsens_extract_dir, 'Virginia', '{}_{}.pt'.format(name, i)))