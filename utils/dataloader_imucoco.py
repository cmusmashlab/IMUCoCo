import os
import glob
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset
from utils.sampler import WeightedDataSampler
from tqdm import tqdm

from utils import imu_config
from utils.imu_config import smpl_vertices_2_bone_direction_joints
import path_config



def collate_fn(batch):
    # Initialize dictionaries to store batched data
    batched_data = {}
    seq_attribute_names = [
        'vimu_joints', 'vimu_mesh', 'imu', 'joint_velocity', 'joint_position', 'joint_orientation', 'pose_local', 'pose_global', 'tran', 'ft_contact'
    ]
    for attribute_name in seq_attribute_names:
        if any(attribute_name in sample for sample in batch):
            batched_data[attribute_name] = torch.stack([sample[attribute_name] for sample in batch])

    if any('imu_corresponding_vertices' in sample for sample in batch):
        batched_data['imu_corresponding_vertices'] = torch.stack([sample['imu_corresponding_vertices'] for sample in batch if 'imu_corresponding_vertices' in sample])

    if any('velocity_init' in sample for sample in batch):
        batched_data['velocity_init'] = torch.stack(
            [sample['velocity_init'] for sample in batch if 'velocity_init' in sample]
        )
    if any('position_init' in sample for sample in batch):
        batched_data['position_init'] = torch.stack(
            [sample['position_init'] for sample in batch if 'position_init' in sample]
        )
    if any('tran_mask' in sample for sample in batch):
        batched_data['tran_mask'] = torch.stack([sample['tran_mask'] for sample in batch if 'tran_mask' in sample])

    if any('sample_id' in sample for sample in batch):
        batched_data['sample_id'] = [sample['sample_id'] for sample in batch]

    if any('z_ref' in sample for sample in batch):
        batched_data['z_ref'] = torch.stack([sample['z_ref'] for sample in batch if 'z_ref' in sample])

    batched_data['sequence_lengths'] = torch.tensor([sample['sequence_lengths'] for sample in batch]).long()
    batched_data['has_real_imu_mask'] = torch.tensor([sample['has_real_imu_mask'] for sample in batch]).bool()

    return batched_data


class IMUCoCoDataset(Dataset):
    def __init__(self,
                 dataset_path=path_config.parsed_pose_dataset_dir,
                 datasets=None,
                 split='train',
                 device='cuda:0',
                 parse_vmesh=True,      # virtual imu at mesh
                 parse_vjoints=True,    # virtual imu at joint
                 parse_imu=True,    # real imu
                 parse_local_pose=True,     # local pose
                 parse_global_pose=True,    # global pose
                 parse_contact=True,    # contact labels
                 parse_tran=True,   # translation
                 parse_joint_vel=True,    # joint velocity
                 parse_joint_pos=True,    # joint position
                 parse_joint_ori=True,  # joint orientation
                 parse_vinit=True, 
                 parse_pinit=True,
                 parse_zref=False,  # the true imu
                 use_joint_asp=True,
                 sequence_length=300,
                 joint_attr_to_root_position=True,
                 presample_mesh=None,
                 local_pose_r6d=True,
                 global_pose_r6d=True,
                 z_ref_cache_dir=None,  # NEW: cache directory for z_ref
                 ):
        super(IMUCoCoDataset, self).__init__()

        if datasets is None:
            datasets = ['DIP_IMU_train', 'XSens', 'AMASS']

        self.dataset_path = dataset_path
        self.datasets = datasets
        self.split = split
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.sequence_length = sequence_length

        self.parse_vmesh = parse_vmesh
        self.parse_vjoints = parse_vjoints
        self.parse_imu = parse_imu
        self.parse_local_pose = parse_local_pose
        self.parse_global_pose = parse_global_pose
        self.parse_tran = parse_tran
        self.parse_contact = parse_contact

        self.parse_joint_vel = parse_joint_vel
        self.parse_joint_pos = parse_joint_pos
        self.parse_joint_ori = parse_joint_ori

        self.parse_vinit = parse_vinit
        self.parse_pinit = parse_pinit
        self.parse_zref = parse_zref

        self.use_joint_asp = use_joint_asp
        self.joint_attr_to_root_position = joint_attr_to_root_position

        self.presample_mesh = presample_mesh

        self.xsens_imu_sensor_vertices = torch.tensor(list(imu_config.xsens_sensor_vertex_ids.values()))
        self.totalcapture_imu_sensor_vertices = (-1) * torch.ones(17)
        self.totalcapture_imu_sensor_vertices[:len(imu_config.tc_sensor_vertex_ids.values())] = torch.tensor(list(imu_config.tc_sensor_vertex_ids.values()))
        self.amass_imu_sensor_vertices = (-1) * torch.ones(17)

        self.samples = []
        self.samples_energy = []
        self.sample_ids = []

        self.local_pose_r6d = local_pose_r6d
        self.global_pose_r6d = global_pose_r6d

        self.z_ref_cache_dir = z_ref_cache_dir

        self.prepare_data()
        print("Number of samples: ", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def prepare_data(self):
        if self.split == 'train':
            # then use the train meta csv to load the samples
            for dataset in self.datasets:
                dataset_meta_file = os.path.join(self.dataset_path, f"{dataset}.csv")
                dataset_meta_df = pd.read_csv(dataset_meta_file)
                for _, row in dataset_meta_df.iterrows():
                    sample = {}
                    sample['dataset_name'] = dataset
                    sample['file_name'] = os.path.join(self.dataset_path, dataset, row['file_name'])
                    sample['length'] = row['length']
                    sample['kinematic_energy'] = row['kinematic_energy']
                    self.samples.append(sample)
                    self.samples_energy.append(row['kinematic_energy'])
                    
                    # Generate sample ID for cache
                    sample_id = f"sample_{len(self.samples)}_{self.sequence_length}"
                    self.sample_ids.append(sample_id)
                print(f"Number of samples in {dataset}: {len(dataset_meta_df)}")
            self.samples_energy = np.asarray(self.samples_energy)
        else:
            # if it is test set, just list the files
            for dataset in self.datasets:
                dataset_dir = os.path.join(self.dataset_path, dataset)
                print("listing files in ", dataset_dir + '/*.pt')
                for file_name in sorted(glob.glob(dataset_dir + '/*.pt')):
                    print(file_name, )
                    sample = {}
                    sample['dataset_name'] = dataset
                    sample['file_name'] = file_name
                    self.samples.append(sample)
                    
                    # Generate sample ID for cache
                    sample_id = f"sample_{len(self.samples)}_{self.sequence_length}"
                    self.sample_ids.append(sample_id)
            print(f"Number of testing samples {len(self.samples)}")

    def __getitem__(self, index):
        sample = self.samples[index]
        dataset_name = sample['dataset_name']
        out_data = torch.load(sample['file_name'])

        parsed_data = {}
        sample_id = self.sample_ids[index]
        parsed_data['sample_id'] = sample_id

        # Parse data selectively based on flags and segment it
        if self.parse_vmesh:
            if out_data['vimu']['vimu_mesh'].shape[2] == 3:
                print("only acc is saved: vimu mesh shape: ", out_data['vimu']['vimu_mesh'].shape)
                vacc_mesh = out_data['vimu']['vimu_mesh']  # if only acc is saved
                vori_mesh = out_data['vimu']['vimu_joints'][:, smpl_vertices_2_bone_direction_joints, 0:6]
                parsed_data['vimu_mesh'] = torch.cat([vori_mesh, vacc_mesh], dim=-1).float()  # N, D, 9
            else:
                parsed_data['vimu_mesh'] = out_data['vimu']['vimu_mesh'].float()
            if self.presample_mesh:
                parsed_data['vimu_mesh'] = parsed_data['vimu_mesh'][:, self.presample_mesh]

        if self.parse_vjoints:
            parsed_data['vimu_joints'] = out_data['vimu']['vimu_joints']
        if self.parse_imu:
            if dataset_name == 'DIP_IMU_train':
                parsed_data['imu'] = out_data['imu']['imu'].float()
                parsed_data['imu_corresponding_vertices'] = self.xsens_imu_sensor_vertices
                parsed_data['has_real_imu_mask'] = ~torch.isnan(parsed_data['imu']).any()
            elif dataset_name == 'DIP_IMU_test':
                parsed_data['imu'] = out_data['imu']['imu'].float()
                parsed_data['imu_corresponding_vertices'] = self.xsens_imu_sensor_vertices
                parsed_data['has_real_imu_mask'] = ~torch.isnan(parsed_data['imu']).any()
            elif dataset_name == 'XSens':
                parsed_data['imu'] = out_data['imu']['imu'].float()
                parsed_data['imu_corresponding_vertices'] = self.xsens_imu_sensor_vertices
                parsed_data['has_real_imu_mask'] = ~torch.isnan(parsed_data['imu']).any()
            elif dataset_name == 'TotalCapture':
                parsed_data['imu'] = out_data['imu']['imu'].float()
                parsed_data['imu_corresponding_vertices'] = self.totalcapture_imu_sensor_vertices
                parsed_data['has_real_imu_mask'] = ~torch.isnan(parsed_data['imu']).any()
            elif dataset_name == 'AMASS':
                parsed_data['imu'] = torch.zeros(out_data['gt']['pose_local'].shape[0], 17, 9).float()
                parsed_data['imu_corresponding_vertices'] = self.amass_imu_sensor_vertices
                parsed_data['has_real_imu_mask'] = torch.tensor(0).bool()

        if self.parse_local_pose:
            if self.local_pose_r6d:
                parsed_data['pose_local'] = out_data['gt']['pose_local'][:, :, :, :2].transpose(2, 3).flatten(2).float()
            else:
                parsed_data['pose_local'] = out_data['gt']['pose_local'].float()
        if self.parse_global_pose:
            if self.global_pose_r6d:
                parsed_data['pose_global'] = out_data['joint']['orientation'][:, :, :, :2].transpose(2, 3).clone().flatten(2)  # global pose is just the global joint orientation
            else:
                parsed_data['pose_global'] = out_data['joint']['orientation']
        if self.parse_tran:
            if out_data['gt']['tran'] is not None:
                parsed_data['tran_mask'] = torch.tensor(1).bool()
                parsed_data['tran'] = out_data['gt']['tran'].float()
            else:
                parsed_data['tran_mask'] = torch.tensor(0).bool()
                parsed_data['tran'] = torch.zeros(out_data['gt']['pose_local'].shape[0], 3).float()
        if self.parse_contact:
            parsed_data['ft_contact'] = out_data['gt']['ft_contact'].float()

        if self.parse_joint_vel or self.parse_vinit:
            if self.use_joint_asp:
                parsed_data['joint_velocity'] = out_data['joint']['asp_velocity']
            else:
                parsed_data['joint_velocity'] = out_data['joint']['velocity']
            
            if self.joint_attr_to_root_position:
                parsed_data['joint_velocity'][:, 1:] = parsed_data['joint_velocity'][:, 1:] - parsed_data['joint_velocity'][:, :1]

        if self.parse_joint_pos:
            if self.use_joint_asp:
                parsed_data['joint_position'] = out_data['joint']['asp_position']
            else:
                parsed_data['joint_position'] = out_data['joint']['position']

            if self.joint_attr_to_root_position:
                parsed_data['joint_position'][:, 1:] = parsed_data['joint_position'][:, 1:] - parsed_data['joint_position'][:, :1]


        if self.parse_joint_ori:
            # global joint orientation
            parsed_data['joint_orientation'] = out_data['joint']['orientation']
            parsed_data['joint_orientation'] = parsed_data['joint_orientation'][:, :, :, :2].transpose(2, 3).clone().flatten(2)

        if self.parse_vinit:
            parsed_data['velocity_init'] = parsed_data['joint_velocity'][0]
        if self.parse_pinit:
            parsed_data['position_init'] = parsed_data['joint_position'][0]

        if self.parse_zref:
            cache_path = os.path.join(self.z_ref_cache_dir, f"{sample_id}.pt")
            parsed_data['z_ref'] = torch.load(cache_path, map_location='cpu')

        seq_attribute_names = [
            'vimu_joints', 'vimu_mesh', 'imu', 'joint_velocity', 'joint_position', 'joint_orientation', 'pose_local', 'pose_global', 'tran', 'ft_contact'
        ]
        parsed_data['sequence_lengths'] = out_data['gt']['pose_local'].shape[0]
        for attr in seq_attribute_names:
            if attr in parsed_data:
                seq = parsed_data[attr]
                seq_length = seq.shape[0]
                if seq_length < self.sequence_length:
                    # Pad with zeros
                    padding = torch.zeros((self.sequence_length - seq_length, *seq.shape[1:]))
                    parsed_data[attr] = torch.cat((seq, padding), dim=0)

        return {key: torch.nn.utils.rnn.pad_sequence(value, batch_first=True) if key in seq_attribute_names else value for key, value in parsed_data.items()}


def get_imucoco_dataloader(dataset_path=path_config.parsed_pose_dataset_dir,
                        datasets=['DIP_IMU', 'XSens', 'AMASS', 'TotalCapture'],
                        seq_len=300, device='cuda:0', batch_size=32,
                        parse_vmesh=True, parse_vjoints=True, parse_imu=True,
                        parse_local_pose=True, parse_global_pose=True,
                        parse_joint_vel=True, parse_joint_pos=True,
                        parse_joint_ori=True,
                        parse_tran=True, parse_vinit=True, parse_pinit=True,
                        use_joint_asp=True, joint_attr_to_root_position=True,
                        use_kinematic_energy_sampling=False,
                        use_kinematic_energy_sampling_steps_per_epoch=200,
                        val_split=0.1, is_test_set=False,
                        presample_mesh=None,
                        workers=0,
                        prefetch_factor=None,
                        local_pose_r6d=True,
                        global_pose_r6d=True,
                        z_ref_cache_dir=None,
                        parse_zref=False,
                        ):
    if is_test_set:
        dataset = IMUCoCoDataset(dataset_path=dataset_path,
                              datasets=datasets, device=device,
                              parse_vmesh=parse_vmesh,
                              parse_vjoints=parse_vjoints,
                              parse_imu=parse_imu,
                              parse_local_pose=parse_local_pose,
                              parse_global_pose=parse_global_pose,
                              parse_joint_ori=parse_joint_ori,
                              parse_tran=parse_tran,
                              parse_joint_vel=parse_joint_vel,
                              parse_joint_pos=parse_joint_pos,
                              parse_vinit=parse_vinit,
                              parse_pinit=parse_pinit,
                              use_joint_asp=use_joint_asp,
                              split='test',
                              joint_attr_to_root_position=joint_attr_to_root_position,
                              sequence_length=seq_len,
                              presample_mesh=presample_mesh,
                              local_pose_r6d=local_pose_r6d,
                              global_pose_r6d=global_pose_r6d,
                              z_ref_cache_dir=z_ref_cache_dir,
                              parse_zref=parse_zref,
                              )
        test_data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=workers, prefetch_factor=prefetch_factor, pin_memory=True)
        return test_data_loader

    else:
        dataset = IMUCoCoDataset(dataset_path=dataset_path,
                              datasets=datasets, device=device,
                              parse_vmesh=parse_vmesh,
                              parse_vjoints=parse_vjoints,
                              parse_imu=parse_imu,
                              parse_local_pose=parse_local_pose,
                              parse_global_pose=parse_global_pose,
                              parse_tran=parse_tran,
                              parse_joint_vel=parse_joint_vel,
                              parse_joint_pos=parse_joint_pos,
                              parse_joint_ori=parse_joint_ori,
                              parse_vinit=parse_vinit,
                              parse_pinit=parse_pinit,
                              use_joint_asp=use_joint_asp,
                              split='train',
                              joint_attr_to_root_position=joint_attr_to_root_position,
                              sequence_length=seq_len,
                              presample_mesh=presample_mesh,
                              local_pose_r6d=local_pose_r6d,
                              global_pose_r6d=global_pose_r6d,
                              z_ref_cache_dir=z_ref_cache_dir,
                              parse_zref=parse_zref,
                              )

        if val_split > 0:
            # split the dataset into training and validation sets
            train_size = int((1 - val_split) * len(dataset))
            indices = torch.randperm(len(dataset)).tolist()
            train_indices = indices[:train_size]
            val_indices = indices[train_size:]

            train_dataset = torch.utils.data.Subset(dataset, train_indices)
            val_dataset = torch.utils.data.Subset(dataset, val_indices)

            if use_kinematic_energy_sampling:
                train_energy_weights = dataset.samples_energy[train_indices]
                sampler_train = WeightedDataSampler(train_energy_weights, num_samples=use_kinematic_energy_sampling_steps_per_epoch * batch_size, replacement=False)
                train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=sampler_train, collate_fn=collate_fn, num_workers=workers, prefetch_factor=prefetch_factor, pin_memory=True)
            else:
                train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=workers, prefetch_factor=prefetch_factor, pin_memory=True)

            val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=workers, prefetch_factor=prefetch_factor, pin_memory=True)

            return train_data_loader, val_data_loader
        else:
            if use_kinematic_energy_sampling:
                train_energy_weights = dataset.samples_energy
                sampler_train = WeightedDataSampler(train_energy_weights, num_samples=use_kinematic_energy_sampling_steps_per_epoch * batch_size, replacement=False)
                train_data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler_train, collate_fn=collate_fn, num_workers=workers, prefetch_factor=prefetch_factor, pin_memory=True)
                return train_data_loader
            else:
                train_data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=workers, prefetch_factor=prefetch_factor, pin_memory=True)
                return train_data_loader
