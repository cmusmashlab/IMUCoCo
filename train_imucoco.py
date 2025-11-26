import argparse
import os
import random
import time

import numpy as np
import torch
from tqdm import tqdm

from models.imucoco import IMUCoCo
from utils import imu_config
from utils.dataloader_imucoco import get_imucoco_dataloader
from utils.dataloader_hpe import get_hpe_dataloader
from utils.logger import ExpLogger
from utils.sampler import HopDecaySampler
import path_config

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

max_epoch_ph1 = 250
max_epoch_ph2 = 180

batch_size_ph1 = 256
batch_size = 6
max_patience = 30
n_hidden = 128

sampler_stage_max = 5
sampler_next_stage_loss_threshold = 0.01

hop_sampler = HopDecaySampler()

def _save_joint_feature_cache(z_ref_cache_dir, sample_id, joint_features):
    """Save joint features to cache directory"""
    os.makedirs(z_ref_cache_dir, exist_ok=True)
    cache_path = os.path.join(z_ref_cache_dir, f"{sample_id}.pt")
    torch.save(joint_features, cache_path)

def _setup_model_and_logger(exp_id):
    """Setup logger and model. Returns (logger, device, imu_coco, save_dir)"""
    logger = ExpLogger(exp_name=f'imu_coco_{exp_id}', exp_type='imu_coco')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    print('Seed:', random_seed)

    save_dir = "./exp_out/" + "imu_coco" + '/' + f"imu_coco_{exp_id}" + "/"
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    logger.log_meta_msg({'seed': random_seed})
    logger.log_meta_msg({'device': str(device)})
    logger.log_meta_msg({'max_epoch_ph1': max_epoch_ph1})
    logger.log_meta_msg({'max_epoch_ph2': max_epoch_ph2})
    logger.log_meta_msg({'sampler_stage_max': sampler_stage_max})
    logger.log_meta_msg({'sampler_next_stage_loss_threshold': sampler_next_stage_loss_threshold})
    logger.log_meta_msg({'batch_size_ph1': batch_size_ph1})
    logger.log_meta_msg({'batch_size': batch_size})
    logger.log_meta_msg({'max_patience': max_patience})

    vertex_coordinates_with_category = torch.tensor(imu_config.vertex_coordinates_with_category).float().to(device)
    joint_coordinates_with_category = torch.tensor(imu_config.joint_coordinates_with_category).float().to(device)

    # the max value of each dimension in the vertex_position_encoding_matrix
    coordinate_max, coordinate_min = (torch.max(vertex_coordinates_with_category[:, 1:], dim=0).values,
                                      torch.min(vertex_coordinates_with_category[:, 1:], dim=0).values)

    imucoco = IMUCoCo(
        coordinate_origins=joint_coordinates_with_category,
        coordinate_max=coordinate_max,
        coordinate_min=coordinate_min,
        smpl_mesh_coordinates = vertex_coordinates_with_category,
        n_hidden=128,
        n_kr_hidden=32,
        n_mfe_layers=2,
        n_jnm_layers=3,
        n_sce_freq=4,
        n_sce_emb=40,
        online_mode=False,
        joint_node_allocation_map=None,
        joint_node_max_err_tolerance=-1,
    ).to(device)

    return logger, device, imucoco, save_dir, vertex_coordinates_with_category, joint_coordinates_with_category

def train_phase1(imucoco, logger, device, save_dir, joint_coordinates_with_category):
    """
    Phase 1: Training Joint2Joint Pose
    Returns the trained imu_coco model
    """

    logger.log_msg("STAGE 1: Training Joint2Joint Pose", verbose=True)
    
    
    cur_joint_position_encoding = joint_coordinates_with_category.unsqueeze(0).repeat(batch_size_ph1, 1, 1)  # (batch_size, n_joints, 4)

    # use the dataset that does not have mesh IMU for faster loading
    hpe_train_dataloader, hpe_val_dataloader = get_hpe_dataloader(
        dataset_path=path_config.parsed_pose_dataset_dir,
        datasets=['AMASS_real_imu_position_only', 'DIP_IMU_train_real_imu_position_only', 'XSens_real_imu_position_only'],
        device=device, batch_size=batch_size_ph1,
        parse_vjoints=True, parse_imu=True,
        parse_local_pose=True, parse_global_pose=True,
        parse_joint_vel=True, parse_joint_pos=True,
        parse_tran=True, parse_vinit=False, parse_pinit=False,
        use_joint_asp=True, joint_attr_to_root_position=True,
        use_kinematic_energy_sampling=True,
        use_kinematic_energy_sampling_steps_per_epoch=100,
        val_split=0.001, is_test_set=False,
        local_pose_r6d=True,
        global_pose_r6d=True,
        workers=4,
        prefetch_factor=None
    )
    best_j2j_only_pretraining_val_loss = float('inf')
    ph1_patience = 0

    imucoco.init_sensor_coordinate_encoders()
    imucoco.freeze_sce()  # initialize to have no modulation at the origin of the joint


    for epoch in range(max_epoch_ph1):
        # train step
        imucoco.train()
        train_loss = 0
        train_pose_loss = 0
        train_kr_loss = 0
        for batch_idx, batch_data in enumerate(tqdm(hpe_train_dataloader)):
            vimu_joints = batch_data['vimu_joints'].to(device=device, non_blocking=True) 
            joint_vel = batch_data['joint_velocity'].to(device=device, non_blocking=True)
            joint_pos = batch_data['joint_position'].to(device=device, non_blocking=True)
            joint_glb_ori = batch_data['joint_orientation'].to(device=device, non_blocking=True)

            tran_mask = batch_data['tran_mask'].to(device=device, non_blocking=True)
            gt_pose_local = batch_data['pose_local'].to(device=device, non_blocking=True)

            if vimu_joints.shape[0] != cur_joint_position_encoding.shape[0]:
                cur_joint_position_encoding = joint_coordinates_with_category.unsqueeze(0).repeat(vimu_joints.shape[0], 1, 1)
            jfe_feat, train_loss_step, pose_loss_step, kr_loss_step = imucoco.forward_joint(imu_inputs_joints=vimu_joints, coordinate_joint=cur_joint_position_encoding,
                                                                               pose_gt=joint_glb_ori, train=True,
                                                                               k_gt=(joint_vel, joint_vel[:, :, 0], joint_pos, joint_glb_ori, gt_pose_local),
                                                                               tran_mask=tran_mask)
            train_loss += train_loss_step
            train_pose_loss += pose_loss_step
            train_kr_loss += kr_loss_step
            logger.log_msg(f"__ Joint-Only-Pretraining Epoch: {epoch} step {batch_idx}, Step Train Loss: {train_loss_step:.4f}, Step Pose Loss: {pose_loss_step:.4f}, Step KR Loss: {kr_loss_step:.4f}", verbose=True)

        train_loss /= len(hpe_train_dataloader)
        train_pose_loss /= len(hpe_train_dataloader)
        train_kr_loss /= len(hpe_train_dataloader)
        logger.log_msg(f"__ Joint-Only-Pretraining Epoch: {epoch} , Train Loss: {train_loss:.4f}, Train Pose Loss: {train_pose_loss:.4f}, Train KR Loss: {train_kr_loss:.4f}", verbose=True)
        
        # eval step
        imucoco.eval()
        val_loss = 0
        val_pose_loss = 0
        val_kr_loss = 0
        for batch_idx, batch_data in enumerate(tqdm(hpe_val_dataloader)):
            vimu_joints = batch_data['vimu_joints'].to(device=device, non_blocking=True) 
            joint_vel = batch_data['joint_velocity'].to(device=device, non_blocking=True)
            joint_pos = batch_data['joint_position'].to(device=device, non_blocking=True)
            joint_glb_ori = batch_data['joint_orientation'].to(device=device, non_blocking=True)

            tran_mask = batch_data['tran_mask'].to(device=device, non_blocking=True)
            gt_pose_local = batch_data['pose_local'].to(device=device, non_blocking=True)

            if vimu_joints.shape[0] != cur_joint_position_encoding.shape[0]:
                cur_joint_position_encoding = joint_coordinates_with_category.unsqueeze(0).repeat(vimu_joints.shape[0], 1, 1)
            jfe_feat, val_loss_step, pose_loss_step, kr_loss_step = imucoco.forward_joint(imu_inputs_joints=vimu_joints, coordinate_joint=cur_joint_position_encoding,
                                                                                 pose_gt=joint_glb_ori, train=False,
                                                                                 k_gt=(joint_vel, joint_vel[:, :, 0], joint_pos, joint_glb_ori, gt_pose_local),
                                                                                 tran_mask=tran_mask)
            val_loss += val_loss_step
            val_pose_loss += pose_loss_step
            val_kr_loss += kr_loss_step
        val_loss /= len(hpe_val_dataloader)
        val_pose_loss /= len(hpe_val_dataloader)
        val_kr_loss /= len(hpe_val_dataloader)
        logger.log_msg(f"Joint-Only-Pretraining Epoch: {epoch}, Val Loss: {val_loss:.4f}, Val Pose Loss: {val_pose_loss:.4f}, Val KR Loss: {val_kr_loss:.4f}", verbose=True)

        if val_loss < best_j2j_only_pretraining_val_loss:
            best_j2j_only_pretraining_val_loss = val_loss
            torch.save(imucoco.state_dict(), os.path.join(save_dir, f"imucoco_ph1_best.pth"))
            logger.log_msg("joint only pretraining model saved.", verbose=True)
            ph1_patience = 0
        else:
            ph1_patience += 1
            if ph1_patience >= max_patience:
                break

    del hpe_train_dataloader, hpe_val_dataloader

    return imucoco


def generate_z_ref_cache(imucoco, logger, device, joint_coordinates_with_category):
    logger.log_msg("Generating joint feature cache for Phase 2...", verbose=True)
    z_ref_cache_dir = os.path.join(path_config.parsed_pose_dataset_dir, "z_ref_cache")
    cache_generation_dataloader = get_imucoco_dataloader(
        dataset_path=path_config.parsed_pose_dataset_dir,
        datasets=['AMASS', 'DIP_IMU_train', 'XSens'], seq_len=300,
        device=device, batch_size=batch_size,
        parse_vmesh=True, parse_vjoints=True, parse_imu=True,
        parse_local_pose=True, parse_global_pose=True,
        parse_joint_vel=True, parse_joint_pos=True,
        parse_tran=True, use_joint_asp=True, joint_attr_to_root_position=True,
        use_kinematic_energy_sampling=False,
        use_kinematic_energy_sampling_steps_per_epoch=-1,
        val_split=0, is_test_set=False,
        presample_mesh=None,
        workers=4,
        prefetch_factor=None,
        parse_zref=False,
        z_ref_cache_dir=z_ref_cache_dir
    )

    # Generate cache using the pretrained model
    imucoco.eval()
    sample_counter = 0
    cur_joint_position_encoding = joint_coordinates_with_category.unsqueeze(0).repeat(batch_size_ph1, 1, 1)  # (batch_size, n_joints, 4)
    for batch_idx, batch_data in enumerate(tqdm(cache_generation_dataloader)):
        with torch.no_grad():
            vimu_joints = batch_data['vimu_joints'].to(device=device, non_blocking=True)

            joint_vel = batch_data['joint_velocity'].to(device=device, non_blocking=True)
            joint_pos = batch_data['joint_position'].to(device=device, non_blocking=True)
            joint_glb_ori = batch_data['joint_orientation'].to(device=device, non_blocking=True)

            tran_mask = batch_data['tran_mask'].to(device=device, non_blocking=True)
            gt_pose_local = batch_data['pose_local'].to(device=device, non_blocking=True)
            
            sample_ids = batch_data['sample_id']

            if vimu_joints.shape[0] != cur_joint_position_encoding.shape[0]:
                cur_joint_position_encoding = joint_coordinates_with_category.unsqueeze(0).repeat(vimu_joints.shape[0], 1, 1)

            jfe_feat, val_loss_step, pose_loss_step, kr_loss_step = imucoco.forward_joint(imu_inputs_joints=vimu_joints, coordinate_joint=cur_joint_position_encoding,
                                                                    pose_gt=joint_glb_ori, train=False,
                                                                    k_gt=(joint_vel, joint_vel[:, :, 0], joint_pos, joint_glb_ori, gt_pose_local),
                                                                    tran_mask=tran_mask)

            # Save joint features to cache
            for i in range(vimu_joints.shape[0]):
                sample_id = sample_ids[i]
                _save_joint_feature_cache(z_ref_cache_dir, sample_id, jfe_feat[i])
                sample_counter += 1
            logger.log_msg(f"Saved joint feature cache for sample {sample_ids}, loss: {val_loss_step:.4f}, pose loss: {pose_loss_step:.4f}, kr loss: {kr_loss_step:.4f}", verbose=True)

    logger.log_msg(f"Joint feature cache generation complete. Generated {sample_counter} cache files.", verbose=True)
    del cache_generation_dataloader
    return z_ref_cache_dir


def train_phase2(imucoco, logger, device, save_dir, vertex_coordinates_with_category, joint_coordinates_with_category, z_ref_cache_dir=None):
    """
    Phase 2: Train Mesh2Joint Training Iteratively
    Returns the trained imu_coco model
    """
    if z_ref_cache_dir is None:
        z_ref_cache_dir = os.path.join(path_config.parsed_pose_dataset_dir, "z_ref_cache")
    logger.log_msg("STAGE 2: Train Mesh2Joint", verbose=True)

    patience = 0
    best_val_loss = float('inf')
    train_dataloader, val_dataloader = get_imucoco_dataloader(dataset_path=path_config.parsed_pose_dataset_dir,
                                                        datasets=['AMASS', 'DIP_IMU_train', 'XSens'],
                                                        device=device, batch_size=batch_size, seq_len=300,
                                                        parse_vmesh=True, parse_vjoints=True, parse_imu=True,
                                                        parse_local_pose=True, parse_global_pose=True,
                                                        parse_joint_vel=True, parse_joint_pos=True,
                                                        parse_tran=True, use_joint_asp=True, joint_attr_to_root_position=True,
                                                        use_kinematic_energy_sampling=True,
                                                        use_kinematic_energy_sampling_steps_per_epoch=2000,
                                                        val_split=0.001, is_test_set=False,
                                                        presample_mesh=None,
                                                        workers=4,
                                                        prefetch_factor=None,
                                                        parse_zref=True,
                                                        z_ref_cache_dir=z_ref_cache_dir)

    for joint_i in range(24):
        print("imu_coco origins", imucoco.sces[joint_i].coordinate_origin)

    total_step_end2end = 0

    # use learned KR PR to supervise; unfreeze sces to learn to adapt to coordinates; freeze mfe to prevent forgetting
    imucoco.freeze_krpr()
    imucoco.freeze_mfe()
    imucoco.unfreeze_sce()

    sampler_stage_tracker = {i: 0 for i in range(24)} # starting with 0 for all joint nodes

    for epoch in range(max_epoch_ph2):
        # training step
        start_time = time.time()

        train_loss = 0
        train_mesh_pose_loss = 0
        train_mesh_kr_loss = 0
        train_align_loss = 0

        loss_by_joints_records = [10.0] * 24

        # train step
        imucoco.train()
        for batch_idx, batch_data in enumerate(tqdm(train_dataloader)):
            vimu_joints = batch_data['vimu_joints'].to(device=device, non_blocking=True)
            vimu_mesh = batch_data['vimu_mesh'].to(device=device, non_blocking=True)  # (batch_size, T, 6890, 9)
            joint_vel = batch_data['joint_velocity'].to(device=device, non_blocking=True)
            joint_pos = batch_data['joint_position'].to(device=device, non_blocking=True)
            z_ref_feats = batch_data['z_ref'].to(device=device, non_blocking=True)
            gt_pose_local = batch_data['pose_local'].to(device=device, non_blocking=True)
            joint_glb_ori = batch_data['joint_orientation'].to(device=device, non_blocking=True)
            tran_mask = batch_data['tran_mask'].to(device=device, non_blocking=True)

            train_vertices_masks = []
            for joint_idx in range(24):
                train_vertices_ji = hop_sampler.sample_vertices(joint_idx, sampler_stage_tracker[joint_idx], n_samples=384)
                train_vertices_masks.append(train_vertices_ji)

            cur_mesh_position_encodings = vertex_coordinates_with_category.unsqueeze(0).expand(vimu_mesh.shape[0], vertex_coordinates_with_category.shape[0], 4)  # (batch_size, n_mesh, 4)

            total_loss_step, loss_by_joints_step, mesh_pose_loss_step, mesh_kr_loss_step, mesh_align_loss_step= \
                imucoco.forward_mesh(imu_inputs_mesh=vimu_mesh, coordinate_mesh=cur_mesh_position_encodings,
                                        joint_ref_feats=z_ref_feats, 
                                        k_gt=(joint_vel, joint_vel[:, :, 0], joint_pos, joint_glb_ori, gt_pose_local), 
                                        pose_gt=joint_glb_ori,
                                        imu_inputs_mesh_masks=train_vertices_masks, tran_mask=tran_mask,
                                        train=True)
            train_loss = train_loss + total_loss_step
            train_mesh_pose_loss = train_mesh_pose_loss + mesh_pose_loss_step
            train_mesh_kr_loss = train_mesh_kr_loss + mesh_kr_loss_step
            train_align_loss = train_align_loss + mesh_align_loss_step

            for j in range(24):
                loss_by_joints_records[j] = loss_by_joints_records[j] * 0.7 + loss_by_joints_step[j] * 0.3
                if loss_by_joints_records[j] < sampler_next_stage_loss_threshold and sampler_stage_tracker[j] < sampler_stage_max:
                        sampler_stage_tracker[j] = sampler_stage_tracker[j] + 1
                        logger.log_msg(f"joint {j} loss {loss_by_joints_records[j]:.4f}, stage moved to {sampler_stage_tracker[j]}", verbose=True)

            logger.log_msg(
                f"End2End Training - total_step_end2end {total_step_end2end} (epoch: {epoch}, step {batch_idx}), Step Train Loss: {total_loss_step:.4f}, mesh_pose_loss_step: {mesh_pose_loss_step}, mesh_kr_loss_step: {mesh_kr_loss_step}, mesh_align_loss_step: {mesh_align_loss_step}",
                verbose=True)
            
            imucoco.scheduler.step()
            total_step_end2end += 1

        train_loss = train_loss / len(train_dataloader)
        train_mesh_pose_loss = train_mesh_pose_loss / len(train_dataloader)
        train_mesh_kr_loss = train_mesh_kr_loss / len(train_dataloader)
        train_align_loss = train_align_loss / len(train_dataloader)

        end_time = time.time()
        logger.log_msg(f"End2End Training Epoch: {epoch} complete, Epoch Time: {end_time - start_time:.2f}")
        logger.log_msg(f"End2End Training Epoch: {epoch} complete, Train Loss {train_loss:.4f}, Train Mesh Pose Loss: {train_mesh_pose_loss:.4f}, Train Mesh KR Loss: {train_mesh_kr_loss:.4f}, Train Align Loss: {train_align_loss:.4f}", verbose=True)
        
        torch.save(imucoco.state_dict(), os.path.join(save_dir, f"imu_coco_end2end_step_{total_step_end2end}.pth"))
        logger.log_msg("train model saved.", verbose=True)

        # validation step
        logger.log_msg("validating....", verbose=True)
        imucoco.eval()
        val_loss = 0
        val_mesh_pose_loss = 0
        val_mesh_kr_loss = 0
        val_align_loss = 0

        for batch_idx, batch_data in enumerate(tqdm(val_dataloader)):
            vimu_joints = batch_data['vimu_joints'].to(device=device, non_blocking=True)
            vimu_mesh = batch_data['vimu_mesh'].to(device=device, non_blocking=True)  # (batch_size, T, 6890, 9)
            joint_vel = batch_data['joint_velocity'].to(device=device, non_blocking=True)
            joint_pos = batch_data['joint_position'].to(device=device, non_blocking=True)
            gt_pose_local = batch_data['pose_local'].to(device=device, non_blocking=True)
            joint_glb_ori = batch_data['joint_orientation'].to(device=device, non_blocking=True)
            tran_mask = batch_data['tran_mask'].to(device=device, non_blocking=True)
            z_ref_feats = batch_data['z_ref'].to(device=device, non_blocking=True)
            

            train_vertices_masks = []
            for joint_idx in range(24):
                train_vertices_ji = hop_sampler.sample_vertices(joint_idx, stage=sampler_stage_max, n_samples=384)
                train_vertices_masks.append(train_vertices_ji)

            cur_mesh_position_encodings = vertex_coordinates_with_category.unsqueeze(0).expand(vimu_mesh.shape[0], vertex_coordinates_with_category.shape[0], 4)  # (batch_size, n_mesh, 4)
            
            total_loss_step, loss_by_joints_step, mesh_pose_loss_step, mesh_kr_loss_step, mesh_align_loss_step= \
                imucoco.forward_mesh(imu_inputs_mesh=vimu_mesh, coordinate_mesh=cur_mesh_position_encodings,
                                        joint_ref_feats=z_ref_feats, 
                                        k_gt=(joint_vel, joint_vel[:, :, 0], joint_pos, joint_glb_ori, gt_pose_local), 
                                        pose_gt=joint_glb_ori,
                                        imu_inputs_mesh_masks=train_vertices_masks, tran_mask=tran_mask,
                                        train=False)


            step_loss = total_loss_step
            val_loss = val_loss + step_loss
            val_mesh_pose_loss += mesh_pose_loss_step
            val_mesh_kr_loss += mesh_kr_loss_step
            val_align_loss += mesh_align_loss_step

        val_loss /= len(val_dataloader)
        val_mesh_pose_loss /= len(val_dataloader)
        val_mesh_kr_loss /= len(val_dataloader)
        val_align_loss /= len(val_dataloader)

        logger.log_msg(f"End2End Validating Epoch: {epoch}, step {total_step_end2end}, Val Loss {val_loss:.4f}, Val Mesh Pose Loss: {val_mesh_pose_loss:.4f}, Val Mesh KR Loss: {val_mesh_kr_loss:.4f}, Val Align Loss: {val_align_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(imucoco.state_dict(), os.path.join(save_dir, f"imucoco_ph2_best.pth"))
            logger.log_msg("BEST model saved.", verbose=True)
            patience = 0
        else:
            patience += 1
            if patience >= max_patience:
                break
    return imucoco

def train(exp_id, args):
    logger, device, imucoco, save_dir, vertex_coordinates_with_category, joint_coordinates_with_category = _setup_model_and_logger(exp_id)
    if args.train_phase1:
        imucoco = train_phase1(imucoco, logger, device, save_dir, joint_coordinates_with_category)
    imucoco.load_state_dict(torch.load(os.path.join(save_dir, f"imucoco_ph1_best.pth")))
    
    if args.generate_z_ref:
        z_ref_cache_dir = generate_z_ref_cache(imucoco, logger, device, joint_coordinates_with_category)
    else:
        z_ref_cache_dir = None
    
    if args.train_phase2:
        imucoco = train_phase2(imucoco, logger, device, save_dir, vertex_coordinates_with_category,joint_coordinates_with_category, z_ref_cache_dir=z_ref_cache_dir)
    imucoco.load_state_dict(torch.load(os.path.join(save_dir, f"imucoco_ph2_best.pth")))
    return imucoco

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training configs")
    parser.add_argument('--no_phase1', action='store_false', dest='train_phase1', default=True, help='Skip Phase 1: Joint2Joint pretraining (default: True)')
    parser.add_argument('--no_phase2', action='store_false', dest='train_phase2', default=True, help='Skip Phase 2: Mesh2Joint & End2End training (default: True)')
    parser.add_argument('--no_generate_z_ref', action='store_false', dest='generate_z_ref', default=True, help='Skip generating joint feature cache for Phase 2 (default: True)')
    parser.add_argument('--exp_id', type=int, default=1, help='Experiment exp id number (default: 1)')
    
    args = parser.parse_args()
    
    print("Training IMUCoCo")
    print("Train Phase 1 (Joint2Joint)", args.train_phase1)
    print("Train Phase 2 (Mesh2Joint & End2End)", args.train_phase2)
    print("Generate z_ref cache", args.generate_z_ref)
    print("Experiment ID", args.exp_id)
    
    train(args.exp_id, args)
