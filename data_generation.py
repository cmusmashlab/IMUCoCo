"""
Data generation pipeline for imucoco training and testing on human pose.
The code parses IMU, pose and related kinematics properties, and generates synthetic IMU data.

The code is adapted from and referenced from:
https://github.com/Xinyu-Yi/PNP
https://github.com/dx118/dynaip
https://github.com/SPICExLAB/MobilePoser
"""

import glob, os, pickle
from typing import Any
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.spatial.transform import Slerp, Rotation
import torch
from tqdm import tqdm
import argparse

import articulate as art
from utils import imu_config
from utils.xsens_extract import extract_mvnx
from path_config import raw_pose_dataset_dir, parsed_pose_dataset_dir


# set the limit of number of files to be processed. this allows us to quickly test the pipeline without using too much storage.
dataset_file_limit = {
    'AnDy': -1,
    'UNIPD': -1,
    'Emokine': -1,
    'CIP': -1,
    'Virginia': -1,
    'AMASS': -1,
}
# to avoid exploding RAM
max_len_at_a_time = 12000

'''
Pose Dataset Directories

Training Data: AMASS, XSENS, DIP_IMU (train split)
Testing Data: TotalCapture, DIP_IMU (test split)
'''

dip_imu_dataset_dir = raw_pose_dataset_dir + '/DIP_IMU'
amass_dataset_dir = raw_pose_dataset_dir + '/AMASS'
amass_data_names = ['ACCAD',
                    'BioMotionLab_NTroje',
                    'BMLhandball',
                    'BMLmovi',
                    'CMU',
                    'DanceDB', 'DFaust_67',
                    ## EKUT
                    'Eyes_Japan_Dataset', 'HUMAN4D',
                    'HumanEva', 'KIT', 'MPI_HDM05',
                    'PosePrior', 'MoSh', 'SFU',
                    'SOMA',
                    'SSM', 'TCDHands', 'Transitions',
                    ## 'WEIZMANN',
                    'YOGI',
                    'EKUT']


totalcapture_dataset_dir = raw_pose_dataset_dir + '/TotalCapture'
xsens_mvnx_dataset_dir = raw_pose_dataset_dir + '/XSens_MVNX'
xsens_dataset_dir = raw_pose_dataset_dir + '/XSens'
xsens_data_names = ['AnDy', 'UNIPD', 'Emokine', 'CIP', 'Virginia']

'''
Dataset Real IMU Specs
'''
dip_imu_dataset_n_imu = 17
total_capture_n_imu = 13

'''
Dataset Virtual IMU Specs & SMPL Model
'''
body_model = imu_config.body_model
smpl_joints_sample_acc_points = [0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 24, 25, 15, 16, 17, 26, 18, 19, 20, 21, 22, 23, 27, 28]
# 24 left toe mesh (3255), 25 right toe mesh(6703), 26 head top mesh (412), 27 left finger mesh (2423), 28 right finger mesh (5885)
smpl_joints_sample_acc_points_mesh = [3255, 6703, 412, 2423, 5885]
constant_vertices = []  # some vertices_positions get constant acceleration, so we remove them

# used for roughly estimating the kinematic energy of motion, loosely following DiffusionPoser table 7 but with more joints
body_mass = torch.tensor([11.7, 9.3, 9.3, 7.69, 9.3, 9.3, 3.84, 0.35, 0.35, 1.92, 0.1, 0.1, 1.92, 2.0, 2.0, 1.92, 2.0, 2.0, 1.2, 1.2, 0.35, 0.35, 0.1, 0.1])

segment_sequence_len = 300


def _foot_ground_probs(joint):
    """Compute foot-ground contact probabilities."""
    dist_lfeet = torch.norm(joint[1:, 10] - joint[:-1, 10], dim=1)
    dist_rfeet = torch.norm(joint[1:, 11] - joint[:-1, 11], dim=1)
    lfoot_contact = (dist_lfeet < 0.008).int()
    rfoot_contact = (dist_rfeet < 0.008).int()
    lfoot_contact = torch.cat((torch.zeros(1, dtype=torch.int), lfoot_contact))
    rfoot_contact = torch.cat((torch.zeros(1, dtype=torch.int), rfoot_contact))
    return torch.stack((lfoot_contact, rfoot_contact), dim=1)


def _syn_acc(v, smooth_n=4):
    r"""
    Synthesize accelerations from vertex positions.
    """
    mid = smooth_n // 2
    acc = torch.stack([(v[i] + v[i + 2] - 2 * v[i + 1]) * 3600 for i in range(0, v.shape[0] - 2)])
    acc = torch.cat((torch.zeros_like(acc[:1]), acc, torch.zeros_like(acc[:1])))
    if mid != 0:
        acc[smooth_n:-smooth_n] = torch.stack(
            [(v[i] + v[i + smooth_n * 2] - 2 * v[i + smooth_n]) * 3600 / smooth_n ** 2
             for i in range(0, v.shape[0] - smooth_n * 2)])
    return acc

def _fill_dip_nan(tensor):
    nan_indices = torch.isnan(tensor)
    filled_tensor = tensor.clone()
    for t in range(tensor.size(0)):
        for i in range(tensor.size(1)):
            for j in range(tensor.size(2)):
                if nan_indices[t, i, j]:
                    left_idx = t - 1
                    while left_idx >= 0 and torch.isnan(tensor[left_idx, i, j]):
                        left_idx -= 1
                    left_neighbor_value = tensor[left_idx, i, j] if left_idx >= 0 else 0

                    right_idx = t + 1
                    while right_idx < tensor.size(0) and torch.isnan(tensor[right_idx, i, j]):
                        right_idx += 1
                    right_neighbor_value = tensor[right_idx, i, j] if right_idx < tensor.size(0) else 0

                    filled_tensor[t, i, j] = (left_neighbor_value + right_neighbor_value) / 2
    return filled_tensor



def __kinematic_energy(body_velocity):
    body_velocity_norm = torch.norm(body_velocity, dim=2)
    # this energy is not exactly based on *center of mass* of each bone, but since it is just a rough estimate to sample the pose, we keep it simple for easier implementation
    kinematic_energy = 0.5 * (body_velocity_norm ** 2) * body_mass
    m_kinematic_energy = torch.mean(kinematic_energy).item()
    return m_kinematic_energy


def __assert_equal_len(data, parse_asp_joint_info=True, parse_origin_joint_info=True):
    assert data['gt']['pose_local'].shape[0] == data['gt']['ft_contact'].shape[0]
    assert data['vimu']['vimu_joints'].shape[0] == data['gt']['pose_local'].shape[0]
    assert data['joint']['orientation'].shape[0] == data['gt']['pose_local'].shape[0]
    if parse_asp_joint_info:
        assert data['gt']['pose_local'].shape[0] == data['joint']['asp_position'].shape[0] == data['joint']['asp_velocity'].shape[0], f"{data['gt']['pose_local'].shape[0]}, {data['joint']['asp_position'].shape[0]}, {data['joint']['asp_velocity'].shape[0]}"
    if parse_origin_joint_info:
        assert data['gt']['pose_local'].shape[0] == data['joint']['position'].shape[0] == data['joint']['velocity'].shape[0]
    if data['imu']['imu'] is not None:
        assert data['imu']['imu'].shape[0] == data['gt']['pose_local'].shape[0]
    if data['gt']['tran'] is not None:
        assert data['gt']['tran'].shape[0] == data['gt']['pose_local'].shape[0]
    if data['vimu']['vimu_mesh'] is not None:
        assert data['vimu']['vimu_mesh'].shape[0] == data['gt']['pose_local'].shape[0]


def _save_segment_sequence_data(data, data_path, data_name, dataset_name, meta_csv_file, seq_len=segment_sequence_len, parse_asp_joint_info=True, parse_origin_joint_info=True):
    data_len = data['gt']['pose_local'].shape[0]
    num_segments = data_len // seq_len

    samples = [] # meta information of the samples
    for seg_idx in range(num_segments):
        start_idx = seg_idx * seq_len
        end_idx = (seg_idx + 1) * seq_len

        out_data = {'joint': {'orientation': None, 'velocity': None, 'position': None, 'asp_position': None, 'asp_velocity': None},
                    'imu': {'imu': None},
                    'vimu': {'vimu_joints': None, 'vimu_mesh': None},
                    'gt': {'pose_local': None, 'tran': None, 'ft_contact': None}}

        out_data['gt']['pose_local'] = data['gt']['pose_local'][start_idx:end_idx].clone()
        out_data['gt']['ft_contact'] = data['gt']['ft_contact'][start_idx:end_idx].clone()

        if data['gt']['tran'] is not None:
            out_data['gt']['tran'] = data['gt']['tran'][start_idx:end_idx].clone()

        if data['imu']['imu'] is not None:
            out_data['imu']['imu'] = data['imu']['imu'][start_idx:end_idx].clone()

        out_data['vimu']['vimu_joints'] = data['vimu']['vimu_joints'][start_idx:end_idx].clone()

        if data['vimu']['vimu_mesh'] is not None:
            out_data['vimu']['vimu_mesh'] = data['vimu']['vimu_mesh'][start_idx:end_idx].clone()

        out_data['joint']['orientation'] = data['joint']['orientation'][start_idx:end_idx].clone()

        segment_ek = 0
        if parse_origin_joint_info:
            joint_position = data['joint']['position'][start_idx:end_idx].clone()
            joint_velocity = data['joint']['velocity'][start_idx:end_idx].clone()
            if not parse_asp_joint_info:
                segment_ek = __kinematic_energy(joint_velocity)
            out_data['joint']['position'] = joint_position
            out_data['joint']['velocity'] = joint_velocity

        if parse_asp_joint_info:
            asp_joint_position = data['joint']['asp_position'][start_idx:end_idx].clone()
            asp_joint_velocity = data['joint']['asp_velocity'][start_idx:end_idx].clone()
            segment_ek = __kinematic_energy(asp_joint_velocity)
            out_data['joint']['asp_position'] = asp_joint_position
            out_data['joint']['asp_velocity'] = asp_joint_velocity

        file_name = f"{data_name}_seg{seg_idx}.pt"
        samples.append({
            'dataset_name': dataset_name,
            'file_name': file_name,
            'length': end_idx - start_idx,
            'kinematic_energy': segment_ek,
        })
        torch.save(out_data, os.path.join(data_path, file_name))

    # Handle remaining data if any
    remaining = data_len % seq_len
    start_idx = data_len - seq_len
    end_idx = data_len
    seg_idx = num_segments

    if remaining > 0:
        out_data = {'joint': {'orientation': None, 'velocity': None, 'position': None, 'asp_position': None, 'asp_velocity': None},
                    'imu': {'imu': None},
                    'vimu': {'vimu_joints': None, 'vimu_mesh': None},
                    'gt': {'pose_local': None, 'tran': None, 'ft_contact': None}}

        out_data['gt']['pose_local'] = data['gt']['pose_local'][start_idx:end_idx].clone()
        out_data['gt']['ft_contact'] = data['gt']['ft_contact'][start_idx:end_idx].clone()

        if data['gt']['tran'] is not None:
            out_data['gt']['tran'] = data['gt']['tran'][start_idx:end_idx].clone()

        if data['imu']['imu'] is not None:
            out_data['imu']['imu'] = data['imu']['imu'][start_idx:end_idx].clone()

        out_data['vimu']['vimu_joints'] = data['vimu']['vimu_joints'][start_idx:end_idx].clone()
        if data['vimu']['vimu_mesh'] is not None:
            out_data['vimu']['vimu_mesh'] = data['vimu']['vimu_mesh'][start_idx:end_idx].clone()

        out_data['joint']['orientation'] = data['joint']['orientation'][start_idx:end_idx].clone()

        segment_ek = 0
        if parse_origin_joint_info:
            joint_position = data['joint']['position'][start_idx:end_idx].clone()
            joint_velocity = data['joint']['velocity'][start_idx:end_idx].clone()
            if not parse_asp_joint_info:
                segment_ek = __kinematic_energy(joint_velocity)
            out_data['joint']['position'] = joint_position
            out_data['joint']['velocity'] = joint_velocity

        if parse_asp_joint_info:
            asp_joint_position = data['joint']['asp_position'][start_idx:end_idx].clone()
            asp_joint_velocity = data['joint']['asp_velocity'][start_idx:end_idx].clone()
            segment_ek = __kinematic_energy(asp_joint_velocity)
            out_data['joint']['asp_position'] = asp_joint_position
            out_data['joint']['asp_velocity'] = asp_joint_velocity

        file_name = f"{data_name}_seg{seg_idx}.pt"
        samples.append({
            'dataset_name': dataset_name,
            'file_name': file_name,
            'length': end_idx - start_idx,
            'kinematic_energy': segment_ek,
        })
        torch.save(out_data, os.path.join(data_path, file_name))

    # append samples to meta_csv_file
    meta_df = pd.DataFrame(samples)
    meta_df.to_csv(meta_csv_file, index=False, mode='a', header=not os.path.exists(meta_csv_file))
    return samples, meta_df



def process_dipimu(args):
    """
    DIP IMU 17 Sensor Placement:
    https://github.com/eth-ait/dip18/issues/16
    sensor_placement = ["head", "sternum", "pelvis", "lshoulder", "rshoulder", "lupperarm", "rupperarm", "llowerarm", "rlowerarm", "lupperleg", "rupperleg", "llowerleg", "rlowerleg", "lhand", "rhand", "lfoot", "rfoot"]
    """
    parse_asp_joint_info = args.parse_asp_joint_info
    parse_origin_joint_info = args.parse_origin_joint_info
    vertex_orientation_approach = args.vertex_orientation_approach
    fullbody = args.fullbody

    out_dir_train = os.path.join(parsed_pose_dataset_dir, f'DIP_IMU_train')
    out_dir_test = os.path.join(parsed_pose_dataset_dir, f'DIP_IMU_test')
    out_dir_real_imu_position_only_train = os.path.join(parsed_pose_dataset_dir, f'DIP_IMU_train_real_imu_position_only')
    out_dir_real_imu_position_only_test = os.path.join(parsed_pose_dataset_dir, f'DIP_IMU_test_real_imu_position_only')
    os.makedirs(out_dir_train, exist_ok=True)
    os.makedirs(out_dir_test, exist_ok=True)
    os.makedirs(out_dir_real_imu_position_only_train, exist_ok=True)
    os.makedirs(out_dir_real_imu_position_only_test, exist_ok=True)

    # list files in raw/DIP_IMU/s_**/*.pkl
    l = []
    split = ['s_01', 's_02', 's_03', 's_04', 's_05', 's_06', 's_07', 's_08', 's_09', 's_10']
    dip_imu_test_subjects = ['s_09', 's_10']
    for subject_name in split:
        file_lists = os.listdir(os.path.join(dip_imu_dataset_dir, subject_name))
        file_lists = [file for file in file_lists if not file.endswith('.DS_Store')]
        for motion_name in file_lists:
            l.append((subject_name, motion_name))

    for subject_name, motion_name in tqdm(l):
        path = os.path.join(dip_imu_dataset_dir, subject_name, motion_name)

        data = pickle.load(open(path, 'rb'), encoding='latin1')

        acc = torch.from_numpy(data['imu_acc']).float()
        ori = torch.from_numpy(data['imu_ori']).float()
        pose_aa = torch.from_numpy(data['gt']).float()
        pose = art.math.axis_angle_to_rotation_matrix(pose_aa).view(-1, 24, 3, 3)

        # fill nan with nearest neighbors
        if True in torch.isnan(acc):
            acc = _fill_dip_nan(acc)
        if True in torch.isnan(ori):
            ori = _fill_dip_nan(ori.view(-1, dip_imu_dataset_n_imu, 9))

        glb_pose, gt_joints_positions, gt_vertex_positions = body_model.forward_kinematics(pose=pose, calc_mesh=True)

        out_data = {'joint': {'orientation': None, 'velocity': None, 'position': None, 'asp_position': None, 'asp_velocity': None},
                    'imu': {'imu': None},
                    'vimu': {'vimu_joints': None, 'vimu_mesh': None},
                    'gt': {'pose_local': None, 'tran': None, 'ft_contact': None}}

        out_data['joint']['orientation'] = glb_pose
        if parse_origin_joint_info:
            joint_velocity = (gt_joints_positions[1:] - gt_joints_positions[:-1]) * 60
            joint_velocity = torch.cat((joint_velocity, joint_velocity[-1].unsqueeze(0)), 0) 
            out_data['joint']['position'] = gt_joints_positions
            out_data['joint']['velocity'] = joint_velocity

        # convert ori to r6d
        ori = ori.view(-1, dip_imu_dataset_n_imu, 3, 3)[:, :, :, :2].transpose(2, 3).clone().flatten(2)
        out_data['imu']['imu'] = torch.cat([ori, acc], dim=-1)

        # get virtual imu sensors for joints
        joint_acc_positions = torch.cat([gt_joints_positions, gt_vertex_positions[:, smpl_joints_sample_acc_points_mesh]], dim=1)[:, smpl_joints_sample_acc_points]
        vacc = _syn_acc(joint_acc_positions)
        vori = glb_pose.view(-1, 24, 3, 3)[:, :, :, :2].transpose(2, 3).flatten(2)
        vimu_joints = torch.cat([vori, vacc], dim=-1)
        out_data['vimu']['vimu_joints'] = vimu_joints

        if parse_asp_joint_info:
            vvel = (joint_acc_positions[1:] - joint_acc_positions[:-1]) * 60
            vvel = torch.cat((vvel, vvel[-1].unsqueeze(0)), 0)
            out_data['joint']['asp_position'] = joint_acc_positions
            out_data['joint']['asp_velocity'] = vvel

        # get virtual imu sensors for mesh vertices_positions
        vacc_mesh = _syn_acc(gt_vertex_positions)
        if vertex_orientation_approach == 'bone':
            vimu_mesh = vacc_mesh  # bone orientation will be filled in at dataloader
        elif vertex_orientation_approach == 'face':
            vori_mesh = imu_config.compute_vertex_orientation(gt_vertex_positions, glb_pose, calibrate=True)
            vori_mesh = vori_mesh.view(-1, 6890, 3, 3)[:, :, :, :2].transpose(2, 3).flatten(2)
            vimu_mesh = torch.cat([vori_mesh, vacc_mesh], dim=-1)
        out_data['vimu']['vimu_mesh'] = vimu_mesh

        out_data['gt']['pose_local'] = pose  # local gt
        out_data['gt']['tran'] = None
        out_data['gt']['ft_contact'] = _foot_ground_probs(gt_joints_positions)  # N, 2

        __assert_equal_len(out_data, parse_asp_joint_info, parse_origin_joint_info)

        if subject_name in dip_imu_test_subjects:
            phase = 'test'
        else:
            phase = 'train'

        data_name = f"{subject_name}_{motion_name.replace('.pkl', '')}"
        if phase == 'train':
            if fullbody:
                # save a version of full mesh IMU
                _save_segment_sequence_data(data=out_data, data_path=out_dir_train, data_name=data_name, dataset_name='DIP_IMU_train', meta_csv_file=os.path.join(parsed_pose_dataset_dir, 'DIP_IMU_train.csv'), seq_len=segment_sequence_len, parse_asp_joint_info=parse_asp_joint_info, parse_origin_joint_info=parse_origin_joint_info)
            # also, save a version only at real imu positions for faster loading
            out_data['vimu']['vimu_mesh'] = None
            _save_segment_sequence_data(data=out_data, data_path=out_dir_real_imu_position_only_train, data_name=data_name, dataset_name='DIP_IMU_real_imu_position_only', meta_csv_file=os.path.join(parsed_pose_dataset_dir, 'DIP_IMU_train_real_imu_position_only.csv'), seq_len=segment_sequence_len, parse_asp_joint_info=parse_asp_joint_info, parse_origin_joint_info=parse_origin_joint_info)
        else:
            # save a version of full mesh IMU
            torch.save(out_data, os.path.join(out_dir_test, data_name + '.pt'))
            # also, save a version only at real imu positions for faster loading
            out_data['vimu']['vimu_mesh'] = None
            torch.save(out_data, os.path.join(out_dir_real_imu_position_only_test, data_name + '.pt'))
        
        print(f"Saved DIP IMU data segment {data_name}")


def process_xsens(args):
    """
    XSens IMU 17 Sensor Placement:
    [Pelvis, T8, Head, RightShoulder, RightUpperArm, RightForeArm, RightHand, LeftShoulder, LeftUpperArm, LeftForeArm, LeftHand, RightUpperLeg, RightLowerLeg, RightFoot, LeftUpperLeg, LeftLowerLeg, LeftFoot]
    https://aferro.dynu.net/blender_mvnx/MVNUserManual.1147412416.pdf, page 92
    """

    parse_asp_joint_info = args.parse_asp_joint_info
    parse_origin_joint_info = args.parse_origin_joint_info
    vertex_orientation_approach = args.vertex_orientation_approach
    fullbody = args.fullbody

    out_dir = os.path.join(parsed_pose_dataset_dir, 'XSens')
    out_dir_real_imu_position_only = os.path.join(parsed_pose_dataset_dir, 'XSens_real_imu_position_only')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir_real_imu_position_only, exist_ok=True)

    xsens_to_dip_imu_order = [
        2, 1, 0, 7, 3, 8, 4, 9, 5, 14, 11, 15, 12, 10, 6, 16, 13
    ]

    def _glb_mat_xsens_to_glb_mat_smpl(glb_full_pose_xsens):
        # refer to https://github.com/dx118/dynaip/blob/main/model/model.py#L169
        glb_full_pose_smpl = torch.eye(3).repeat(glb_full_pose_xsens.shape[0], 24, 1, 1)
        indices = [0, 19, 15, 1, 20, 16, 3, 21, 17, 4, 22, 18, 5, 11, 7, 6, 12, 8, 13, 9, 13, 9, 13, 9]
        for idx, i in enumerate(indices):
            glb_full_pose_smpl[:, idx, :] = glb_full_pose_xsens[:, i, :]
        return glb_full_pose_smpl

    # list files in raw/XSens/xxxx/
    l = []
    for dataset_name in os.listdir(xsens_dataset_dir):
        if '.DS_Store' in dataset_name:
            continue
        if 'UNIPD' in dataset_name or 'Emokine' in dataset_name:
            continue
        num_processed_output_files = 0
        file_lists = os.listdir(os.path.join(xsens_dataset_dir, dataset_name))
        file_lists = [file for file in file_lists if not file.endswith('.DS_Store')]
        for motion_name in file_lists:
            l.append((dataset_name, motion_name))
            num_processed_output_files += 1
            if dataset_file_limit[dataset_name] != -1 and num_processed_output_files >= dataset_file_limit[dataset_name]:
                print(f'{dataset_name} dataset file limit reached')
                continue
        if len(file_lists) == 0:
            print(f'{dataset_name} dataset is empty!!')
            return

    for (dataset_name, motion_name) in tqdm(l):
        temp_data = torch.load(os.path.join(xsens_dataset_dir, dataset_name, motion_name))

        out_data = {'joint': {'orientation': None, 'velocity': None, 'position': None, 'asp_position': None, 'asp_velocity': None},
                    'imu': {'imu': None},
                    'vimu': {'vimu_joints': None, 'vimu_mesh': None},
                    'gt': {'pose_local': None, 'tran': None, 'ft_contact': None}}

        # normalize and to r6d
        xsens_glb_pose = art.math.quaternion_to_rotation_matrix(temp_data['joint']['orientation']).view(-1, 23, 3, 3)
        glb_pose = _glb_mat_xsens_to_glb_mat_smpl(xsens_glb_pose)
        pose_local = body_model.inverse_kinematics_R(glb_pose).view(glb_pose.shape[0], 24, 3, 3)

        acc = temp_data['imu']['free acceleration'].view(-1, 17, 3)
        ori = art.math.quaternion_to_rotation_matrix(temp_data['imu']['calibrated orientation']).view(-1, 17, 3, 3)
        # convert to DIP IMU order
        # [head, chest, pelvis, lshoulder, rshoulder, lupperarm, rupperarm, llowerarm, rlowerarm, lupperleg, rupperleg, llowerleg, rlowerleg, lhand, rhand, lfoot, rfoot]
        acc = acc[:, xsens_to_dip_imu_order]    
        ori = ori[:, xsens_to_dip_imu_order]

        out_data = {
            'joint': {'orientation': None, 'velocity': None, 'position': None, 'asp_position': None, 'asp_velocity': None},
            'imu': {'imu': None},
            'vimu': {'vimu_joints': None, 'vimu_mesh': None},
            'gt': {'pose_local': None, 'tran': None, 'ft_contact': None},
        }

        glb_pose, gt_joints_positions, gt_vertex_positions = body_model.forward_kinematics(pose=pose_local, calc_mesh=True)

        out_data['joint']['orientation'] = glb_pose
        if parse_origin_joint_info:
            joint_velocity = (gt_joints_positions[1:] - gt_joints_positions[:-1]) * 60
            joint_velocity = torch.cat((joint_velocity, joint_velocity[-1].unsqueeze(0)), 0)
            out_data['joint']['position'] = gt_joints_positions
            out_data['joint']['velocity'] = joint_velocity

        joint_acc_positions = torch.cat([gt_joints_positions, gt_vertex_positions[:, smpl_joints_sample_acc_points_mesh]], dim=1)[:, smpl_joints_sample_acc_points]
        vacc = _syn_acc(joint_acc_positions)
        vori = glb_pose.view(-1, 24, 3, 3)[:, :, :, :2].transpose(2, 3).flatten(2)

        vimu_joints = torch.cat([vori, vacc], dim=-1)
        out_data['vimu']['vimu_joints'] = vimu_joints

        ori = ori.view(-1, 17, 3, 3)[:, :, :, :2].transpose(2, 3).flatten(2)
        out_data['imu']['imu'] = torch.cat([ori, acc], dim=-1)

        if parse_asp_joint_info:
            vvel = (joint_acc_positions[1:] - joint_acc_positions[:-1]) * 60
            vvel = torch.cat((vvel, vvel[-1].unsqueeze(0)), 0)
            out_data['joint']['asp_position'] = joint_acc_positions
            out_data['joint']['asp_velocity'] = vvel

        vacc_mesh = _syn_acc(gt_vertex_positions)
        if vertex_orientation_approach == 'face':
            vori_mesh = imu_config.compute_vertex_orientation(gt_vertex_positions, glb_pose, calibrate=True)
            vori_mesh = vori_mesh.view(-1, 6890, 3, 3)[:, :, :, :2].transpose(2, 3).flatten(2)
            vimu_mesh = torch.cat([vori_mesh, vacc_mesh], dim=-1)
        else:
            vimu_mesh = vacc_mesh

        out_data['vimu']['vimu_mesh'] = vimu_mesh

        out_data['gt']['pose_local'] = pose_local
        out_data['gt']['ft_contact'] = _foot_ground_probs(gt_joints_positions)

        __assert_equal_len(out_data, parse_asp_joint_info, parse_origin_joint_info)

        data_name = f"{dataset_name}_{motion_name.replace('.pt', '')}"
        if fullbody:
            # save a version of full mesh IMU
            _save_segment_sequence_data(data=out_data, data_path=out_dir, data_name=data_name, dataset_name='XSens', meta_csv_file=os.path.join(parsed_pose_dataset_dir, 'XSENS.csv'), seq_len=segment_sequence_len, parse_asp_joint_info=parse_asp_joint_info, parse_origin_joint_info=parse_origin_joint_info)
        # also, save a version only at real imu positions for faster loading
        out_data['vimu']['vimu_mesh'] = None
        _save_segment_sequence_data(data=out_data, data_path=out_dir_real_imu_position_only, data_name=data_name, dataset_name='XSens_real_imu_position_only', meta_csv_file=os.path.join(parsed_pose_dataset_dir, 'XSENS_real_imu_position_only.csv'), seq_len=segment_sequence_len, parse_asp_joint_info=parse_asp_joint_info, parse_origin_joint_info=parse_origin_joint_info)
        print(f"Saved Xsens data segment {data_name}")


def process_totalcapture(args):
    parse_asp_joint_info = args.parse_asp_joint_info
    parse_origin_joint_info = args.parse_origin_joint_info
    vertex_orientation_approach = args.vertex_orientation_approach
    fullbody = args.fullbody

    out_dir = os.path.join(parsed_pose_dataset_dir, 'TotalCapture')
    out_dir_real_imu_position_only = os.path.join(parsed_pose_dataset_dir, 'TotalCapture_real_imu_position_only')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir_real_imu_position_only, exist_ok=True)

    vicon_gt_dir = os.path.join(totalcapture_dataset_dir, 'pos_ori')  # download from TotalCapture page
    imu_dir = os.path.join(totalcapture_dataset_dir, 'gyro_mag')  # download from TotalCapture page
    calib_dir = os.path.join(totalcapture_dataset_dir, 'imu')  # download from TotalCapture page
    DIP_smpl_dir = os.path.join(totalcapture_dataset_dir, 'dip_smpl')  # SMPL pose calculated by DIP. Download from DIP page
    joint_names = ['Head', 'Sternum', 'Pelvis', 'L_UpArm', 'R_UpArm', 'L_LowArm', 'R_LowArm', 'L_UpLeg', 'R_UpLeg', 'L_LowLeg', 'R_LowLeg', 'L_Foot', 'R_Foot']
    n_extracted_imus = len(joint_names)
    for subject_name in ['s1', 's2', 's3', 's4', 's5']:
        file_lists = os.listdir(os.path.join(imu_dir, subject_name))
        file_lists = [file for file in file_lists if not file.endswith('.DS_Store')]
        for action_name in tqdm(sorted(file_lists)):
            out_data = {'joint': {'orientation': None, 'velocity': None, 'position': None, 'asp_position': None, 'asp_velocity': None},
                        'imu': {'imu': None},
                        'vimu': {'vimu_joints': None, 'vimu_mesh': None},
                        'gt': {'pose_local': None, 'tran': None, 'ft_contact': None}}
            # read imu file
            f = open(os.path.join(imu_dir, subject_name, action_name), 'r')
            line = f.readline().split('\t')
            n_sensors, n_frames = int(line[0]), int(line[1])
            R = torch.zeros(n_frames, n_extracted_imus, 4)
            a = torch.zeros(n_frames, n_extracted_imus, 3)
            for i in range(n_frames):
                assert int(f.readline()) == i + 1, 'parse imu file error'
                for _ in range(n_sensors):
                    line = f.readline().split('\t')
                    if line[0] in joint_names:
                        j = joint_names.index(line[0])
                        R[i, j] = torch.tensor([float(_) for _ in line[1:5]])  # wxyz
                        a[i, j] = torch.tensor([float(_) for _ in line[5:8]])
            R = art.math.quaternion_to_rotation_matrix(R).view(-1, n_extracted_imus, 3, 3)

            # read calibration file
            name = subject_name + '_' + action_name.split('_')[0].lower()
            RSB = torch.zeros(n_extracted_imus, 3, 3)
            RIM = torch.zeros(n_extracted_imus, 3, 3)
            with open(os.path.join(calib_dir, subject_name, name + '_calib_imu_bone.txt'), 'r') as f:
                n_sensors = int(f.readline())
                for _ in range(n_sensors):
                    line = f.readline().split()
                    if line[0] in joint_names:
                        j = joint_names.index(line[0])
                        q = torch.tensor([float(line[4]), float(line[1]), float(line[2]), float(line[3])])  # wxyz
                        RSB[j] = art.math.quaternion_to_rotation_matrix(q)[0].t()
            with open(os.path.join(calib_dir, subject_name, name + '_calib_imu_ref.txt'), 'r') as f:
                n_sensors = int(f.readline())
                for _ in range(n_sensors):
                    line = f.readline().split()
                    if line[0] in joint_names:
                        j = joint_names.index(line[0])
                        q = torch.tensor([float(line[4]), float(line[1]), float(line[2]), float(line[3])])  # wxyz
                        RIM[j] = art.math.quaternion_to_rotation_matrix(q)[0].t()
            RSB = RSB.matmul(torch.tensor([[-1, 0, 0], [0, 0, -1], [0, -1, 0.]]))  # change bone frame to SMPL
            RIM = RIM.matmul(torch.tensor([[-1, 0, 0], [0, 1, 0], [0, 0, -1.]]))  # change global frame to SMPL

            # read root translation
            tran = []
            with open(os.path.join(vicon_gt_dir, subject_name.upper(), action_name.split('_')[0].lower(), 'gt_skel_gbl_pos.txt')) as f:
                idx = f.readline().split('\t').index('Hips')
                while True:
                    line = f.readline()
                    if line == '':
                        break
                    t = [float(_) * 0.0254 for _ in line.split('\t')[idx].split(' ')]  # inches_to_meters
                    tran.append([-t[0], t[1], -t[2]])
            tran = torch.tensor(tran)

            # read SMPL pose parameters calculated by DIP
            f = os.path.join(DIP_smpl_dir, name + '.pkl')
            DIP_pose = None
            if os.path.exists(f):
                d = pickle.load(open(f, 'rb'), encoding='latin1')
                DIP_pose = torch.from_numpy(d['gt']).float()
            else:
                continue

            # align data
            n_aligned_frames = min(n_frames, tran.shape[0], DIP_pose.shape[0] if DIP_pose is not None else 1e8, DIP_pose.shape[0] if DIP_pose is not None else 1e8)
            if DIP_pose is not None:
                DIP_pose = DIP_pose[-n_aligned_frames:]
            tran = tran[-n_aligned_frames:] - tran[-n_aligned_frames]
            R = R[-n_aligned_frames:]
            a = a[-n_aligned_frames:]

            # calibrate to the global frame
            ori_glb = RIM.transpose(1, 2).matmul(R).matmul(RSB)
            acc_glb = RIM.transpose(1, 2).matmul(R).matmul(a.unsqueeze(-1)).squeeze(-1)
            acc_glb = acc_glb - torch.tensor([0.0, 9.8067, 0.0])

            ori = ori_glb.view(-1, total_capture_n_imu, 3, 3)[:, :, :, :2].transpose(2, 3).clone().flatten(2)
            imu = torch.cat([ori, acc_glb], dim=-1)  # N, D=13, 9
            # pad 0s to make it 17 sensors
            imu = torch.cat([imu, torch.zeros(n_aligned_frames, dip_imu_dataset_n_imu - total_capture_n_imu, 9)], dim=1)
            out_data['imu']['imu'] = imu

            p = art.math.axis_angle_to_rotation_matrix(DIP_pose).view(-1, 24, 3, 3)
            # calculate the global rotations of joints
            glb_pose, gt_joints_positions, gt_vertex_positions = body_model.forward_kinematics(pose=p, tran=tran, calc_mesh=True)

            out_data['joint']['orientation'] = glb_pose  # N 90
            if parse_origin_joint_info:
                joint_velocity = (gt_joints_positions[1:] - gt_joints_positions[:-1]) * 60
                joint_velocity = torch.cat((joint_velocity, joint_velocity[-1].unsqueeze(0)), 0)  #
                out_data['joint']['position'] = gt_joints_positions  # N, 24, 3
                out_data['joint']['velocity'] = joint_velocity  # N, 24, 3, 3

            # get virtual imu sensors for joints
            # we sample acceleration at these points, which corresponds to each joint (bone)'s end-point's acceleration
            joint_acc_positions = torch.cat([gt_joints_positions, gt_vertex_positions[:, smpl_joints_sample_acc_points_mesh]], dim=1)[:, smpl_joints_sample_acc_points]
            vacc = _syn_acc(joint_acc_positions)
            vori = glb_pose.view(-1, 24, 3, 3)[:, :, :, :2].transpose(2, 3).flatten(2)
            vimu_joints = torch.cat([vori, vacc], dim=-1)  # N, D, 9
            out_data['vimu']['vimu_joints'] = vimu_joints

            if parse_asp_joint_info:
                vvel = (joint_acc_positions[1:] - joint_acc_positions[:-1]) * 60
                vvel = torch.cat((vvel, vvel[-1].unsqueeze(0)), 0)  #
                out_data['joint']['asp_position'] = joint_acc_positions  # N, 24, 3
                out_data['joint']['asp_velocity'] = vvel  # N, 24, 3

            vacc_mesh = _syn_acc(gt_vertex_positions)
            if vertex_orientation_approach == 'bone':
                vimu_mesh = vacc_mesh
            elif vertex_orientation_approach == 'face':
                vori_mesh = imu_config.compute_vertex_orientation(gt_vertex_positions, glb_pose, calibrate=True)
                vori_mesh = vori_mesh.view(-1, 6890, 3, 3)[:, :, :, :2].transpose(2, 3).flatten(2)
                vimu_mesh = torch.cat([vori_mesh, vacc_mesh], dim=-1)  # N, D, 9
            out_data['vimu']['vimu_mesh'] = vimu_mesh

            out_data['gt']['pose_local'] = p  # local gt
            out_data['gt']['tran'] = tran  # DIP_IMU does not have translation
            out_data['gt']['ft_contact'] = _foot_ground_probs(gt_joints_positions)  # N, 2
            __assert_equal_len(out_data, parse_asp_joint_info, parse_origin_joint_info)

            if fullbody:
                # save a version of full mesh IMU
                out_path = os.path.join(out_dir, f'{subject_name}_{action_name}'.replace(".sensors", ".pt"))
                torch.save(out_data, out_path)
            # also, save a version only at real imu positions for faster loading
            out_data['vimu']['vimu_mesh'] = None
            out_path = os.path.join(out_dir_real_imu_position_only, f'{subject_name}_{action_name}'.replace(".sensors", ".pt"))
            torch.save(out_data, out_path)

            print(f"Saved TotalCapture data segment {subject_name}_{action_name}")


def process_amass(args):
    parse_asp_joint_info = args.parse_asp_joint_info
    parse_origin_joint_info = args.parse_origin_joint_info
    vertex_orientation_approach = args.vertex_orientation_approach
    fullbody = args.fullbody

    out_dir = os.path.join(parsed_pose_dataset_dir, 'AMASS')
    out_dir_real_imu_position_only = os.path.join(parsed_pose_dataset_dir, 'AMASS_real_imu_position_only')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir_real_imu_position_only, exist_ok=True)

    amass_rot = torch.tensor([[[1, 0, 0], [0, 0, 1], [0, -1, 0.]]])  # align axis with dip calibration
    num_processed_output_files = 0

    for dataset_name in amass_data_names:
        print('Processing %s' % dataset_name)
        file_lists = (
                glob.glob(os.path.join(amass_dataset_dir, dataset_name, dataset_name, '*/*_poses.npz')) +
                glob.glob(os.path.join(amass_dataset_dir, dataset_name, dataset_name, '*/*_stageii.npz')) +
                glob.glob(os.path.join(amass_dataset_dir, dataset_name, '*/*_poses.npz')) +
                glob.glob(os.path.join(amass_dataset_dir, dataset_name, '*/*_stageii.npz')) +
                glob.glob(os.path.join(amass_dataset_dir, dataset_name, '*/*/*_poses.npz')) +
                glob.glob(os.path.join(amass_dataset_dir, dataset_name, '*/*/*_stageii.npz'))
        )
        if len(file_lists) == 0:
            print(f"!! Cannot find any data files for {dataset_name}!! ")
            return
        file_lists = [file for file in file_lists if not file.endswith('.DS_Store')]

        for npz_fname in tqdm(file_lists):
            seq_name = npz_fname[npz_fname.rfind(dataset_name):-4]
            cdata = np.load(npz_fname)
            out_data = {'joint': {'orientation': None, 'velocity': None, 'position': None, 'asp_position': None, 'asp_velocity': None},
                        'imu': {'imu': None},
                        'vimu': {'vimu_joints': None, 'vimu_mesh': None},
                        'gt': {'pose_local': None, 'tran': None, 'ft_contact': None}}
            if 'mocap_framerate' in cdata:
                framerate = int(cdata['mocap_framerate'])
            elif 'mocap_frame_rate' in cdata:
                framerate = int(cdata['mocap_frame_rate'])
            else:
                print('\tFail to process %s: no framerate' % seq_name)
                continue
            if cdata['poses'].shape[0] < framerate * 0.5:
                print('\tFail to process %s: too short' % seq_name)
                continue

            # Split data into segments if necessary
            total_frames = cdata['poses'].shape[0]
            if total_frames > max_len_at_a_time + 2000:
                print(f"Data is too long, segmenting {seq_name}: total frames {total_frames} exceed max_len_at_a_time {max_len_at_a_time}")
                num_segments = (total_frames + max_len_at_a_time - 1) // max_len_at_a_time  # Ceiling division
                segments = []
                for i in range(num_segments):
                    start_idx = i * max_len_at_a_time
                    end_idx = min((i + 1) * max_len_at_a_time, total_frames)
                    segment_data = {
                        'poses': cdata['poses'][start_idx:end_idx],
                        'trans': cdata['trans'][start_idx:end_idx],
                        'mocap_framerate': framerate
                    }
                    segments.append(segment_data)
            else:
                segments = [{'poses': cdata['poses'], 'trans': cdata['trans'], 'mocap_framerate': framerate}]

            for segment_idx, segment in enumerate(segments):
                poses = segment['poses']
                trans = segment['trans']
                framerate = segment['mocap_framerate']
                if framerate == 120:
                    smplh_pose = torch.from_numpy(poses[::2].astype(np.float32)).reshape(-1, 52, 3)
                    pose = smplh_pose[:, :24]
                    pose[:, 22] = smplh_pose[:, 25]
                    pose[:, 23] = smplh_pose[:, 40]
                    tran = torch.from_numpy(trans[::2].astype(np.float32)).view(-1, 3)
                elif framerate in {60, 59}:
                    smplh_pose = torch.from_numpy(poses.astype(np.float32)).reshape(-1, 52, 3)
                    pose = smplh_pose[:, :24]
                    pose[:, 22] = smplh_pose[:, 25]
                    pose[:, 23] = smplh_pose[:, 40]
                    tran = torch.from_numpy(trans.astype(np.float32)).view(-1, 3)
                else:
                    smplh_pose = poses.reshape(-1, 52, 3)
                    origin_tran = trans.reshape(-1, 3)
                    origin_t = np.arange(smplh_pose.shape[0]) / framerate
                    t = np.arange(0, origin_t[-1], 1 / 60)
                    pose = np.empty((len(t), 24, 3))
                    for i in range(24):
                        if i == 22:
                            j = 25
                        if i == 23:
                            j = 40
                        else:
                            j = i
                        pose[:, i] = Slerp(origin_t, Rotation.from_rotvec(smplh_pose[:, j]))(t).as_rotvec()
                    tran = interp1d(origin_t, origin_tran, axis=0)(t)
                    pose = torch.from_numpy(pose.astype(np.float32))
                    tran = torch.from_numpy(tran.astype(np.float32)).view(-1, 3)

                tran = amass_rot.matmul(tran.unsqueeze(-1)).view_as(tran)
                # rotate the root joint to the model frame
                pose[:, 0, :3] = art.math.rotation_matrix_to_axis_angle(amass_rot.matmul(art.math.axis_angle_to_rotation_matrix(pose[:, 0, :3])))
                pose = art.math.axis_angle_to_rotation_matrix(pose.contiguous()).contiguous().view(-1, 24, 3, 3)

                # calculate the global rotations of joints
                glb_pose, gt_joints_positions, gt_vertex_positions = body_model.forward_kinematics(pose=pose, tran=tran, calc_mesh=True)
                out_data['joint']['orientation'] = glb_pose
                if parse_origin_joint_info:
                    joint_velocity = (gt_joints_positions[1:] - gt_joints_positions[:-1]) * 60
                    joint_velocity = torch.cat((joint_velocity, joint_velocity[-1].unsqueeze(0)), 0)
                    out_data['joint']['position'] = gt_joints_positions
                    out_data['joint']['velocity'] = joint_velocity

                # get virtual imu sensors for joints
                joint_acc_positions = torch.cat([gt_joints_positions, gt_vertex_positions[:, smpl_joints_sample_acc_points_mesh]], dim=1)[:, smpl_joints_sample_acc_points]
                vacc = _syn_acc(joint_acc_positions)
                vori = glb_pose.view(-1, 24, 3, 3)[:, :, :, :2].transpose(2, 3).flatten(2)
                vimu_joints = torch.cat([vori, vacc], dim=-1)
                out_data['vimu']['vimu_joints'] = vimu_joints

                if parse_asp_joint_info:
                    vvel = (joint_acc_positions[1:] - joint_acc_positions[:-1]) * 60
                    vvel = torch.cat((vvel, vvel[-1].unsqueeze(0)), 0) 
                    out_data['joint']['asp_position'] = joint_acc_positions
                    out_data['joint']['asp_velocity'] = vvel

                vacc_mesh = _syn_acc(gt_vertex_positions)
                if vertex_orientation_approach == 'bone':
                    vimu_mesh = vacc_mesh
                elif vertex_orientation_approach == 'face':
                    vori_mesh = imu_config.compute_vertex_orientation(gt_vertex_positions, glb_pose, calibrate=True)
                    vori_mesh = vori_mesh.view(-1, 6890, 3, 3)[:, :, :, :2].transpose(2, 3).flatten(2)
                    vimu_mesh = torch.cat([vori_mesh, vacc_mesh], dim=-1)  # N, D, 9

                out_data['vimu']['vimu_mesh'] = vimu_mesh
                out_data['gt']['pose_local'] = pose
                out_data['gt']['tran'] = tran
                out_data['gt']['ft_contact'] = _foot_ground_probs(gt_joints_positions)  # N, 2
                __assert_equal_len(out_data, parse_asp_joint_info, parse_origin_joint_info)

                seq_name = seq_name.replace("/", "_")
                data_name = f"{dataset_name}_{seq_name}_{segment_idx + 1}"

                if fullbody:
                    # save a version of full mesh IMU
                    _save_segment_sequence_data(data=out_data, data_path=out_dir, data_name=data_name, dataset_name='AMASS', meta_csv_file=os.path.join(parsed_pose_dataset_dir, 'AMASS.csv'), seq_len=segment_sequence_len, parse_asp_joint_info=parse_asp_joint_info, parse_origin_joint_info=parse_origin_joint_info)
                
                # also, save a version only at real imu positions for faster loading
                out_data['vimu']['vimu_mesh'] = vimu_mesh[:, list[Any](imu_config.xsens_sensor_vertex_ids.values())]
                _save_segment_sequence_data(data=out_data, data_path=out_dir_real_imu_position_only, data_name=data_name, dataset_name='AMASS_real_imu_position_only', meta_csv_file=os.path.join(parsed_pose_dataset_dir, 'AMASS_real_imu_position_only.csv'), seq_len=segment_sequence_len, parse_asp_joint_info=parse_asp_joint_info, parse_origin_joint_info=parse_origin_joint_info)
                
                print(f"Saved AMASS data segment {data_name}")

            num_processed_output_files += 1
            if dataset_file_limit['AMASS'] != -1 and num_processed_output_files >= dataset_file_limit['AMASS']:
                print('AMASS dataset file limit reached')
                return


def process_imucoco(args):
    """Process the IMUCoCo test dataset into training/eval-ready .pt files.

    For each participant + (focus, activity) we read three .npz files produced by the
    IMUCoCo dataset preparation pipeline:
        {focus}_{activity}.npz                  pose_local (T,24,3,3) and trans (T,3)
        {focus}_{activity}_imu.npz              per-device raw IMU streams
        {focus}_{activity}_calibration_lab.npz  R_nav2model, R_bone2sensor per device

    The IMU stream is calibrated into the SMPL model frame:
        ori_glb = R_nav2model @ R_sensor2nav @ R_bone2sensor
        acc_glb = R_nav2model @ acc_nav  -  accel_offset_model
    The pose stream is forward-kinematic'd to obtain joint orientations, joint/vertex
    positions, virtual-IMU readings at the acceleration sampling points, and foot-
    ground contact probabilities.

    IMUCoCo is a test-only dataset, so we save one .pt per (participant, focus, activity)
    full clip (no segmentation) to:
        parsed_pose_dataset_dir/IMUCoCo_real_imu_position_only/
    along with a single meta CSV listing all clips.
    """
    parse_asp_joint_info = args.parse_asp_joint_info
    parse_origin_joint_info = args.parse_origin_joint_info

    imucoco_raw_dir = os.path.join(raw_pose_dataset_dir, 'IMUCoCo')
    out_dir_real_imu_position_only = os.path.join(parsed_pose_dataset_dir, 'IMUCoCo_real_imu_position_only')
    os.makedirs(out_dir_real_imu_position_only, exist_ok=True)

    # Find all pose .npz files (each PXX/ folder contains pose, imu, and calibration files;
    # we filter out the IMU and calibration ones so we iterate one row per activity).
    pose_files = sorted(glob.glob(os.path.join(imucoco_raw_dir, 'P*', '*.npz')))
    pose_files = [f for f in pose_files
                  if not f.endswith('_imu.npz') and not f.endswith('_calibration_lab.npz')]

    print(f"Found {len(pose_files)} IMUCoCo pose files")

    for pose_file in tqdm(pose_files, desc="IMUCoCo"):
        pid = os.path.basename(os.path.dirname(pose_file))  # P01, P02, ...
        activity_name = os.path.splitext(os.path.basename(pose_file))[0]  # Upper_Walking
        data_name = f"IMUCoCo_{pid}_{activity_name}"

        # Check if already processed
        out_path_check = os.path.join(out_dir_real_imu_position_only, f"{data_name}.pt")
        if os.path.exists(out_path_check):
            continue

        out_data = {'joint': {'orientation': None, 'velocity': None, 'position': None,
                              'asp_position': None, 'asp_velocity': None},
                    'imu': {'imu': None},
                    'vimu': {'vimu_joints': None, 'vimu_mesh': None},
                    'gt': {'pose_local': None, 'tran': None, 'ft_contact': None}}

        # Load pose
        pose_data = np.load(pose_file)
        pose_local = torch.tensor(pose_data['pose_local']).float()  # (T, 24, 3, 3)
        tran = torch.tensor(pose_data['trans']).float()  # (T, 3)
        tran = tran - tran[:1]  # relative to first frame

        n_frames = pose_local.shape[0]
        if n_frames < 30:  # absolute minimum (1 second at 30Hz / 0.5s at 60Hz)
            print(f"  Skipping {data_name}: only {n_frames} frames")
            continue

        # Load real IMU data and calibration
        imu_file = pose_file.replace('.npz', '_imu.npz')
        cal_file = pose_file.replace('.npz', '_calibration_lab.npz')

        if not os.path.exists(imu_file) or not os.path.exists(cal_file):
            print(f"  Skipping {data_name}: missing IMU or calibration")
            continue

        imu_data = np.load(imu_file, allow_pickle=True)
        cal_data = np.load(cal_file, allow_pickle=True)

        # Calibration constants. R_nav2model and accel_offset_model are shared across all
        # devices in this recording (they describe the world↔model frame transform measured
        # at the wrist during the lab T-pose); R_bone2sensor is per device.
        R_nav2model = torch.tensor(cal_data['R_nav2model']).float()
        accel_offset_model = torch.tensor(cal_data['accel_offset_model']).float() \
            if 'accel_offset_model' in cal_data.files else torch.zeros(3)

        devices = list(imu_data['devices'])
        n_imu_frames = int(imu_data['n_frames'])
        n_devices = len(devices)

        # Align IMU and pose to the shorter of the two streams.
        T = min(n_frames, n_imu_frames)

        imu_ori = torch.zeros(T, n_devices, 3, 3)
        imu_acc = torch.zeros(T, n_devices, 3)

        for di, dev in enumerate(devices):
            key_q = f'{dev}_quaternion'
            key_a = f'{dev}_user_acceleration'
            key_rb = f'{dev}_R_bone2sensor'

            if key_q not in imu_data or key_rb not in cal_data:
                continue

            quat = torch.tensor(imu_data[key_q][:T]).float()  # (T, 4) wxyz
            acc = torch.tensor(imu_data[key_a][:T]).float()    # (T, 3) user acceleration (no gravity)

            # R_sensor2nav from quaternion
            w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
            R_s2n = torch.zeros(T, 3, 3)
            R_s2n[:, 0, 0] = 1-2*(y*y+z*z); R_s2n[:, 0, 1] = 2*(x*y-w*z); R_s2n[:, 0, 2] = 2*(x*z+w*y)
            R_s2n[:, 1, 0] = 2*(x*y+w*z); R_s2n[:, 1, 1] = 1-2*(x*x+z*z); R_s2n[:, 1, 2] = 2*(y*z-w*x)
            R_s2n[:, 2, 0] = 2*(x*z-w*y); R_s2n[:, 2, 1] = 2*(y*z+w*x); R_s2n[:, 2, 2] = 1-2*(x*x+y*y)

            R_b2s = torch.tensor(cal_data[key_rb]).float()  # (3, 3)

            # ori_glb = R_nav2model @ R_sensor2nav @ R_bone2sensor
            imu_ori[:, di] = R_nav2model @ R_s2n @ R_b2s

            # acc_glb = R_nav2model @ acc_nav - accel_offset_model
            imu_acc[:, di] = (R_nav2model @ acc[:, :, None]).squeeze(-1) - accel_offset_model

        # Format: orientation as 6D (first 2 rows of rotmat) + acceleration = 9D per device
        ori_6d = imu_ori[:, :, :, :2].transpose(2, 3).flatten(2)  # (T, n_devices, 6)
        imu = torch.cat([ori_6d, imu_acc], dim=-1)  # (T, n_devices, 9)
        out_data['imu']['imu'] = imu[:T]

        # Truncate pose to match
        pose_local = pose_local[:T]
        tran = tran[:T]

        # Forward kinematics
        glb_pose, gt_joints_positions, gt_vertex_positions = body_model.forward_kinematics(
            pose=pose_local, tran=tran, calc_mesh=True)

        out_data['joint']['orientation'] = glb_pose
        if parse_origin_joint_info:
            joint_velocity = (gt_joints_positions[1:] - gt_joints_positions[:-1]) * 60
            joint_velocity = torch.cat((joint_velocity, joint_velocity[-1].unsqueeze(0)), 0)
            out_data['joint']['position'] = gt_joints_positions
            out_data['joint']['velocity'] = joint_velocity

        # Synthetic per-joint IMU at the acceleration sampling points (used by some
        # downstream evaluation utilities). vimu_mesh is left as None — IMUCoCo only
        # uses real IMUs at test time, so we don't need full-mesh virtual IMU.
        joint_acc_positions = torch.cat(
            [gt_joints_positions, gt_vertex_positions[:, smpl_joints_sample_acc_points_mesh]], dim=1
        )[:, smpl_joints_sample_acc_points]
        vacc = _syn_acc(joint_acc_positions)
        vori = glb_pose.view(-1, 24, 3, 3)[:, :, :, :2].transpose(2, 3).flatten(2)
        vimu_joints = torch.cat([vori, vacc], dim=-1)
        out_data['vimu']['vimu_joints'] = vimu_joints

        if parse_asp_joint_info:
            vvel = (joint_acc_positions[1:] - joint_acc_positions[:-1]) * 60
            vvel = torch.cat((vvel, vvel[-1].unsqueeze(0)), 0)
            out_data['joint']['asp_position'] = joint_acc_positions
            out_data['joint']['asp_velocity'] = vvel

        out_data['vimu']['vimu_mesh'] = None

        out_data['gt']['pose_local'] = pose_local
        out_data['gt']['tran'] = tran
        out_data['gt']['ft_contact'] = _foot_ground_probs(gt_joints_positions)

        __assert_equal_len(out_data, parse_asp_joint_info, parse_origin_joint_info)

        # IMUCoCo is a test-only dataset — save the full activity clip (no segmentation)
        # and append a single row to the dataset-level meta CSV.
        meta_csv_file = os.path.join(parsed_pose_dataset_dir, 'IMUCoCo_real_imu_position_only.csv')
        file_name = f"{data_name}.pt"
        torch.save(out_data, os.path.join(out_dir_real_imu_position_only, file_name))
        ek = __kinematic_energy(out_data['joint']['asp_velocity']) if parse_asp_joint_info else (
            __kinematic_energy(out_data['joint']['velocity']) if parse_origin_joint_info else 0)
        meta_row = pd.DataFrame([{
            'dataset_name': 'IMUCoCo_real_imu_position_only',
            'file_name': file_name,
            'length': out_data['gt']['pose_local'].shape[0],
            'kinematic_energy': float(ek),
        }])
        meta_row.to_csv(meta_csv_file, index=False, mode='a',
                        header=not os.path.exists(meta_csv_file))

    print(f"IMUCoCo processing done. Output: {out_dir_real_imu_position_only}")



def main(args):
    if args.extract_xsens:
        extract_mvnx(xsens_mvnx_dataset_dir, xsens_dataset_dir)
    if args.process_totalcapture:
        process_totalcapture(args)
    if args.process_dipimu:
        process_dipimu(args)
    if args.process_amass:
        process_amass(args)
    if args.process_xsens:
        process_xsens(args)
    if args.process_imucoco:
        process_imucoco(args)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--fullbody', action='store_true', help='generate full body synthetic IMU data (requiring large storage space and is used for training IMUCoCo)', default=False)
    parser.add_argument('--no_extract_xsens', action='store_false', dest='extract_xsens', default=True, help='Skip XSens data extraction (default: True)')
    parser.add_argument('--no_process_totalcapture', action='store_false', dest='process_totalcapture', default=True, help='Skip TotalCapture data processing (default: True)')
    parser.add_argument('--no_process_dipimu', action='store_false', dest='process_dipimu', default=True, help='Skip DIP-IMU data processing (default: True)')
    parser.add_argument('--no_process_xsens', action='store_false', dest='process_xsens', default=True, help='Skip XSens data processing (default: True)')
    parser.add_argument('--no_process_amass', action='store_false', dest='process_amass', default=True, help='Skip AMASS data processing (default: True)')
    parser.add_argument('--no_process_imucoco', action='store_false', dest='process_imucoco', default=True, help='Skip IMUCoCo data processing (default: True)')
    parser.add_argument('--no_parse_asp_joint_info', action='store_false', dest='parse_asp_joint_info', default=True, help='Skip parsing acceleration-sampling-points joint velocity and positions (default: True)')
    parser.add_argument('--no_parse_origin_joint_info', action='store_false', dest='parse_origin_joint_info', default=True, help="Skip parsing joint velocity and positions at the joint itself (default: True)")
    
    parser.add_argument('--vertex_orientation_approach', type=str, default='face', help="Approach for Mesh Orientation (default: face)")
    args = parser.parse_args()


    print("Generating full body IMU data")
    print("Extract XSens data", args.extract_xsens)
    print("Parse ASP", args.parse_asp_joint_info)
    print("Parse Original", args.parse_origin_joint_info)
    print("Approach for Mesh Orientation", args.vertex_orientation_approach)

    main(args)