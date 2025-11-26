import argparse
import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from models.dtp import Poser
from models.imucoco import IMUCoCo
from utils import imu_config
from utils.dataloader_hpe import get_hpe_dataloader
from utils.imu_config import xsens_real_imu_index
from utils.logger import ExpLogger
from utils.evalautor import HPEEvaluator, compute_mean_results_by_placement
import path_config

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

random_seed = 1000
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)


pose_model_name = 'dtp'
max_epochs = 50
max_epochs_pose = 25
batch_size = 32
learning_rate = 1e-3

def train(exp_id, logger):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    save_dir = os.path.join(path_config.exp_out_dir, 'hpe', f"imu_coco_hpe_{exp_id}_{pose_model_name}")
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    logger.log_meta_msg({'seed': random_seed})
    logger.log_meta_msg({'pose_model': pose_model_name})
    logger.log_meta_msg({'device': str(device)})
    logger.log_meta_msg({'learning_rate': learning_rate})
    logger.log_meta_msg({'max_epochs': max_epochs})
    logger.log_meta_msg({'batch_size': batch_size})

    vertex_coordinates_with_category = torch.tensor(imu_config.vertex_coordinates_with_category).float().to(device)
    joint_coordinates_with_category = torch.tensor(imu_config.joint_coordinates_with_category).float().to(device)
    coordinate_max, coordinate_min = (torch.max(vertex_coordinates_with_category[:, 1:], dim=0).values,
                                      torch.min(vertex_coordinates_with_category[:, 1:], dim=0).values)

    imucoco_model = IMUCoCo(
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
        joint_node_allocation_map=path_config.saved_imucoco_loss_map_path,
        joint_node_max_err_tolerance=-1,
    ).to(device)
    imucoco_model.load_state_dict(torch.load(path_config.saved_imucoco_checkpoint_path, map_location=device), strict=False)
    imucoco_model.freeze()
    imucoco_model.eval()
    poser_model = Poser(joint_feature_dim=128, 
                        n_hidden=300, 
                        n_glb=40, 
                        num_layer=3,
                        n_total_devices=24, 
                        load_tran_module=True).to(device)

    train_dataloader = get_hpe_dataloader(
        dataset_path=path_config.parsed_pose_dataset_dir,
        datasets=['AMASS_real_imu_position_only', 'DIP_IMU_train_real_imu_position_only', 'XSens_real_imu_position_only'],
        device=device, batch_size=batch_size,
        parse_vjoints=True, parse_imu=True,
        parse_local_pose=True, parse_global_pose=True,
        parse_joint_vel=True, parse_joint_pos=False,
        parse_tran=True, parse_vinit=True, parse_pinit=False,
        use_joint_asp=True, joint_attr_to_root_position=True,
        use_kinematic_energy_sampling=True,
        use_kinematic_energy_sampling_steps_per_epoch=150,
        val_split=0, is_test_set=False,
        local_pose_r6d=False,
        global_pose_r6d=True,
        workers=4,
        prefetch_factor=None
    )
    print("Finished loading dataset.")
    
    optimizer = torch.optim.Adam(list(imucoco_model.parameters()) + list(poser_model.parameters()), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs // 2)
    logger.log_msg(f"Training for {max_epochs} Epochs", verbose=True)


    vertex_coordinates = torch.tensor(imu_config.vertex_coordinates).float().to(device)
    standard_case_names = imu_config.tc_imu_standard_eval_case[0]
    standard_eval_vertex_id = [imu_config.tc_sensor_vertex_ids[v_name] for v_name in standard_case_names]
    # amass, xsens, and dip all use the xsens imu indices
    standard_xsens_imu_index = [imu_config.xsens_real_imu_index[v_name] for v_name in standard_case_names]
    standard_eval_coordinates = vertex_coordinates[standard_eval_vertex_id]
    # train & validate with the standard 6 imu placements
    imucoco_model.set_current_device_coordinates(standard_eval_coordinates)
    imucoco_model.buffer_placement_codes_with_current_devices(parallel=False)

    stage = 'pose'

    for epoch in range(26, max_epochs):
        # training step
        train_loss = 0.
        poser_model.train()

        for batch_idx, batch_data in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            gt_vel = batch_data['joint_velocity'].to(device=device, non_blocking=True)
            gt_pose_glb_r6d = batch_data['pose_global'].to(device=device, non_blocking=True)
            gt_pose_local = batch_data['pose_local'].to(device=device, non_blocking=True)
            gt_contact = batch_data['ft_contact'].to(device=device, non_blocking=True)
            b_imu_data = batch_data['imu'].to(device=device, non_blocking=True)
            b_imu_data = b_imu_data[:, :, standard_xsens_imu_index]
            vel_init = batch_data['velocity_init'].to(device=device, non_blocking=True)
            seq_len = batch_data['sequence_lengths']
            tran_mask = batch_data['tran_mask'].to(device=device, non_blocking=True)

            feat_m = imucoco_model.inference_time_forward_mesh(b_imu_data)

            p_out, v_out, local_pred, joint_pos_pred, root_vel_pred, ft_contact_pred = poser_model.forward(
                x=feat_m, v_init=vel_init, glb_init=gt_pose_glb_r6d[:, 0], seq_len=seq_len, compute_tran='train')
            pose_loss, glb_rot_loss = poser_model.pose_loss_func(p_out=p_out, v_out=v_out, local_pred=local_pred, pos_pred=joint_pos_pred,
                                                                     glb_pose_gt=gt_pose_glb_r6d, local_pose_gt=gt_pose_local, vel_gt=gt_vel, seq_len=seq_len)
            tran_loss = poser_model.tran_loss_func(contact_pred=ft_contact_pred, contact_gt=gt_contact, rvel_pred=root_vel_pred, rvel_gt=gt_vel[:, :, 0], tran_mask=tran_mask, seq_len=seq_len)

            if stage == 'pose':
                loss = pose_loss + 0.1 * tran_loss
            elif stage == 'tran':
                loss = tran_loss
            loss.backward()
            optimizer.step()
            step_loss = (pose_loss + tran_loss).item()
            logger.log_msg(f"__epoch {epoch}, train step {batch_idx} loss {step_loss}, pose_loss {pose_loss.item()}, tran_loss {tran_loss.item()}", verbose=True)
            train_loss += step_loss

        torch.save(poser_model.state_dict(), os.path.join(save_dir, f"poser_{exp_id}_{pose_model_name}_epoch{epoch}.pth"))
        train_loss = train_loss / len(train_dataloader)
        logger.log_msg(f"epoch {epoch}, train_loss: {train_loss}", verbose=True)
        scheduler.step()

        if stage == 'pose':
            torch.save(poser_model.state_dict(), os.path.join(save_dir, f"poser_{exp_id}_{pose_model_name}_best.pth"))
            logger.log_msg(f"model saved.", verbose=True)
            if epoch >= max_epochs_pose:
                logger.log_msg("Stopped pose training. Move to translation training.", verbose=True)
                stage = 'tran'
                poser_model.freeze_poser()
                
                train_dataloader = get_hpe_dataloader(
                                                dataset_path=path_config.parsed_pose_dataset_dir,
                                                datasets=['AMASS_real_imu_position_only'],
                                                device=device, batch_size=batch_size,
                                                parse_vjoints=True, parse_imu=True,
                                                parse_local_pose=True, parse_global_pose=True,
                                                parse_joint_vel=True, parse_joint_pos=False,
                                                parse_tran=True, parse_contact=True, parse_vinit=True, parse_pinit=False,
                                                use_joint_asp=True, joint_attr_to_root_position=True,
                                                use_kinematic_energy_sampling=True,
                                                use_kinematic_energy_sampling_steps_per_epoch=150,
                                                val_split=0, is_test_set=False,
                                                local_pose_r6d=False,
                                                global_pose_r6d=True,
                                                workers=4,
                                                prefetch_factor=None
                                            )
        elif stage == 'tran':
            torch.save(poser_model.state_dict(), os.path.join(save_dir, f"poser_{exp_id}_{pose_model_name}_best.pth"))
            logger.log_msg(f"model saved.", verbose=True)
                    
    poser_model.load_state_dict(torch.load(os.path.join(save_dir, f"poser_{exp_id}_{pose_model_name}_best.pth"), map_location=device), strict=False)
    poser_model.eval()

    return imucoco_model, poser_model


def test_tc(exp_id, logger, imucoco_model=None, poser_model=None, simple=False):
    test_dataloader = get_hpe_dataloader(dataset_path=path_config.parsed_pose_dataset_dir,
                                             datasets=['TotalCapture_real_imu_position_only'],
                                             device=device, batch_size=1,
                                             parse_vjoints=True, parse_imu=True,
                                             parse_local_pose=True, parse_global_pose=True,
                                             parse_joint_vel=True, parse_joint_pos=True,
                                             parse_tran=True, parse_contact=False, parse_vinit=True, parse_pinit=False,
                                             use_joint_asp=True, joint_attr_to_root_position=True,
                                             use_kinematic_energy_sampling=False,
                                             val_split=0, is_test_set=True,
                                             local_pose_r6d=False, global_pose_r6d=True)
    if imucoco_model is None:
        vertex_coordinates_with_category = torch.tensor(imu_config.vertex_coordinates_with_category).float().to(device)
        joint_coordinates_with_category = torch.tensor(imu_config.joint_coordinates_with_category).float().to(device)
        coordinate_max, coordinate_min = (torch.max(vertex_coordinates_with_category[:, 1:], dim=0).values,
                                      torch.min(vertex_coordinates_with_category[:, 1:], dim=0).values)
        imucoco_model = IMUCoCo(
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
            joint_node_allocation_map=path_config.saved_imucoco_loss_map_path,
            joint_node_max_err_tolerance=-1,
        ).to(device)
        imucoco_model.load_state_dict(torch.load(path_config.saved_imucoco_checkpoint_path, map_location=device), strict=False)
        imucoco_model.freeze()
        imucoco_model.eval()

    if poser_model is None:
        poser_model = Poser(joint_feature_dim=128, 
                            n_hidden=300, 
                            n_glb=40, 
                            num_layer=3,
                            n_total_devices=24, 
                            load_tran_module=True).to(device)


        
        poser_model.load_state_dict(torch.load(path_config.saved_hpe_checkpoint_path, map_location=device), strict=False)
        poser_model.eval()
        

    my_pose_evaluator = HPEEvaluator('smpl/SMPL_MALE.pkl', device=device, fps=60)
    
    tc_vertex_id_cases = [[imu_config.tc_sensor_vertex_ids[v_name] for v_name in case] for case in imu_config.tc_imu_6_subset_eval_cases]
    tc_vertex_name_cases = imu_config.tc_imu_6_subset_eval_cases
    tc_imu_index_cases = [[imu_config.tc_real_imu_index[v_name] for v_name in case] for case in tc_vertex_name_cases]
    vertex_coordinates = torch.tensor(imu_config.vertex_coordinates).float().to(device)
    all_results = []
    all_results_mean_by_placement_case = []
    for vertex_id_case, vertex_name_case, vertex_index_case in zip(tc_vertex_id_cases, tc_vertex_name_cases, tc_imu_index_cases):
        case_all_results = []
        real_imu_coordinates = vertex_coordinates[vertex_id_case]
        imucoco_model.set_current_device_coordinates(real_imu_coordinates)
        imucoco_model.buffer_placement_codes_with_current_devices(parallel=False)
        for batch_idx, batch_data in enumerate(tqdm(test_dataloader)):
            gt_pose_glb_r6d = batch_data['pose_global'].to(device=device, non_blocking=True)  # (batch_size, T, 24, 6)
            gt_pose_local = batch_data['pose_local'].to(device=device, non_blocking=True)
            b_imu_data = batch_data['imu'].to(device=device, non_blocking=True)
            vel_init = batch_data['velocity_init'].to(device=device, non_blocking=True)  # (batch_size, 24, 3)
            seq_len = batch_data['sequence_lengths']

            tran_gt = batch_data['tran'].to(device=device, non_blocking=True)
            b_imu_data = b_imu_data[:, :, vertex_index_case]

            feat_m = imucoco_model.inference_time_forward_mesh(b_imu_data)
            glb_pose_pred, pose_local_pred, tran_out = poser_model.forward(x=feat_m, v_init=vel_init, glb_init=gt_pose_glb_r6d[:, 0], seq_len=seq_len, compute_tran='transpose')

            result_b = my_pose_evaluator(pose_p=glb_pose_pred[0], pose_t=gt_pose_local[0], shape_p=None, shape_t=None, tran_p=tran_out, tran_t=tran_gt[0],
                                         instrumented_arm=None, instrumented_leg=None, is_predicted_pose_local=False, is_pose_p_r6d=False, is_pose_t_r6d=False)
            case_all_results.append(result_b)


        case_mean_results = compute_mean_results_by_placement(case_all_results)
        case_mean_results['input_names'] = '+'.join(vertex_name_case)
        all_results.extend(case_all_results)
        all_results_mean_by_placement_case.append(case_mean_results)
        logger.log_msg(f"Test Case {'+'.join(vertex_name_case)}, result {case_mean_results}", verbose=True)
        if simple:
            return

    df_all_results = pd.DataFrame(all_results)
    df_all_results.to_csv(os.path.join(path_config.exp_out_dir, 'hpe', f"{pose_model_name}_{exp_id}_test_total_capture_all_results.csv"), index=False)
    df_mean_results = pd.DataFrame(all_results_mean_by_placement_case)
    df_mean_results.to_csv(os.path.join(path_config.exp_out_dir, 'hpe', f"{pose_model_name}_{exp_id}_test_total_capture__mean_results.csv"), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="IMUCoCo HPE training and testing script")
    parser.add_argument('--train', action='store_true', help='Run training')
    parser.add_argument('--test_tc', action='store_true', help='Run TotalCapture testing')
    parser.add_argument('--exp_id', type=str, required=False, default=1, help='Experiment ID')

    args = parser.parse_args()
    logger = ExpLogger(exp_name=f'imu_coco_hpe_{args.exp_id}_{pose_model_name}', exp_type='hpe')
    imucoco_model, poser_model = None, None
    if args.train:
        imucoco_model, poser_model = train(args.exp_id, logger)
    if args.test_tc:
        test_tc(args.exp_id, logger, imucoco_model, poser_model)