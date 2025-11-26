import argparse
import os
import random
import time

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from PIL import Image

from models.imucoco import IMUCoCo
from utils import imu_config
from utils.dataloader_imucoco import get_imucoco_dataloader
from articulate.math import r6d_to_rotation_matrix, radian_to_degree, angle_between
import path_config

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Constants
MESH_ITERATION_BATCH_SIZE = 2048  # Process vertices in batches to manage memory
NUM_JOINTS = 24
NUM_VERTICES = 6890

random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)

def plot_body_mesh_heatmap_signal(heatmap_values,
                                  filename,
                                  title='Heatmap',
                                  value_title='',
                                  large_is_worse=True,
                                  range_low=0,
                                  range_high=-1,
                                  unit='',
                                  use_scale='sqrt'):
    """
    Generates plot that has several components.
    1. generate a body mesh with heatmap. The values of the heatmap can be customized, such as to represent orientation, acceleration, etc.

    Parameters:
    - heatmap_values: numpy array or torch tensor of size (6980), representing heatmap values for each vertex.
    - filename: string, the name of the file to save the output (e.g., 'output.png').
    - title: string, the label for the heatmap (default is 'Heatmap').
    - model_path: string, path to the SMPL model file (default is 'pose/smpl/SMPL_NEUTRAL.pkl').
    - large_is_worse: boolean, whether large value means worse. If True, larger value will be red, and smaller value will be blue.
    - range_low: the lower bound of the range of values to visualize. if -1, it will be the min of the provided value. default is 0.
    - range_high: the upper bound of the range of values to visualize. if -1, it will be the max of the provided value. default is -1.
    - scale: could be 'linear', 'log', 'sqrt', '075',
    """

    # Load the SMPL model
    import numpy as np
    import pyvista as pv
    import torch
    
    # Convert PyTorch tensor to numpy array if needed
    if torch.is_tensor(heatmap_values):
        heatmap_values = heatmap_values.detach().cpu().numpy()

    # Define rotation matrices
    def rotation_matrix_y(angle):
        rad = np.radians(angle)
        return np.array([
            [np.cos(rad), 0, np.sin(rad)],
            [0, 1, 0],
            [-np.sin(rad), 0, np.cos(rad)]
        ])

    def rotation_matrix_z(angle):
        rad = np.radians(angle)
        return np.array([
            [np.cos(rad), -np.sin(rad), 0],
            [np.sin(rad), np.cos(rad), 0],
            [0, 0, 1]
        ])

    # Compute total rotation matrix
    R_y1 = rotation_matrix_y(45)
    R_z = rotation_matrix_z(90)
    R_y2 = rotation_matrix_y(45)
    R_z2 = rotation_matrix_z(30)

    R_total = R_y1 @ R_z @ R_y2 @ R_z2  # Matrix multiplication

    smpl_model = imu_config.body_model
    vertices = imu_config.initial_vertex_positions.detach().cpu().numpy().squeeze()  # (N, 3)
    faces = smpl_model.face  # (F, 3)
    faces_with_sizes = np.hstack([np.full((faces.shape[0], 1), 3), faces]).flatten()

    # Create PyVista mesh
    vertices = vertices @ R_total.T

    mesh = pv.PolyData(vertices, faces_with_sizes)
    if heatmap_values.shape[0] != vertices.shape[0]:
        raise ValueError(f"Heatmap size {heatmap_values.shape[0]} does not match vertices {vertices.shape[0]}.")

    # Determine value range
    if range_low == -1:
        range_low = np.min(heatmap_values)
    if range_high == -1:
        range_high = np.max(heatmap_values)

    mesh[title] = heatmap_values

    if use_scale == 'log':
        mesh['scaled_values'] = np.log(np.clip(mesh[title], 1e-3, None))
        range_low = np.log(np.max([range_low, 1e-3]))
        range_high = np.log(range_high)

    elif use_scale == 'sqrt':
        mesh['scaled_values'] = np.sqrt(mesh[title])
        range_low = np.sqrt(range_low)
        range_high = np.sqrt(range_high)
    elif use_scale == '075':
        mesh['scaled_values'] = mesh[title] ** 0.75
        range_low = range_low ** 0.75
        range_high = range_high ** 0.75

    elif use_scale == 'linear':
        mesh['scaled_values'] = mesh[title]

    cmap = 'coolwarm' if large_is_worse else 'coolwarm_r'

    # PyVista plotting
    plotter = pv.Plotter(off_screen=True, window_size=[1600, 1600])  # Increased resolution
    plotter.add_mesh(mesh, scalars=title, cmap=cmap, show_edges=False, smooth_shading=True, clim=[range_low, range_high])
    plotter.set_background('white')
    plotter.remove_scalar_bar()  # Ensure PyVista scalar bar is removed

    value_title = f"{value_title} ({unit})"

    plotter.set_background('white')

    # Save the screenshot
    screenshot_filename = filename.replace('.png', '_heatmap.png')
    os.makedirs(os.path.dirname(screenshot_filename), exist_ok=True)
    plotter.show(screenshot=screenshot_filename)
    plotter.close()

    # ---------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(12, 1))
    norm = mcolors.Normalize(vmin=range_low, vmax=range_high)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Define custom 6 ticks (emphasizing lower ranges)
    # Define 6 meaningful tick values based on the scale
    if use_scale == 'log':
        tick_labels = np.exp(np.linspace(range_low, range_high, 6))
        tick_values = np.linspace(range_low, range_high, 6)

    elif use_scale == 'sqrt':
        tick_labels = np.linspace(range_low, range_high, 6) ** 2  # Evenly spaced in original scale
        tick_values = np.linspace(range_low, range_high, 6)  # Transform to sqrt-space

    elif use_scale == '075':
        tick_labels = np.linspace(range_low, range_high, 6) ** (1 / 0.75)
        tick_values = np.linspace(range_low, range_high, 6)

    else:  # Linear scale
        tick_labels = np.linspace(range_low, range_high, 6)  # Evenly spaced in original scale
        tick_values = np.linspace(range_low, range_high, 6)  # No transformation

    # Round for cleaner labels
    tick_labels = np.round(tick_labels, 2)

    cbar = plt.colorbar(sm, orientation='horizontal', cax=ax, ticks=tick_values)
    cbar.set_label(f"{value_title}", fontsize=12)
    cbar.set_ticks(tick_values)
    cbar.set_ticklabels(tick_labels)

    # Save the color bar as a separate image
    colorbar_filename = screenshot_filename.replace('.png', '_colorbar.png')
    plt.savefig(colorbar_filename, dpi=300, bbox_inches='tight')
    plt.close()

    # ---- Step 3: Merge Heatmap and Color Bar ---- #
    heatmap_img = Image.open(screenshot_filename)
    colorbar_img = Image.open(colorbar_filename)

    # Resize colorbar to match the heatmap width
    colorbar_width = heatmap_img.width
    colorbar_img = colorbar_img.resize((colorbar_width, int(colorbar_img.height * 0.5)))  # Increase height slightly for readability

    # Create a new image with extra space for the color bar
    combined_height = heatmap_img.height + colorbar_img.height
    combined_img = Image.new("RGB", (heatmap_img.width, combined_height), "white")

    # Paste images together
    combined_img.paste(heatmap_img, (0, 0))
    combined_img.paste(colorbar_img, (0, heatmap_img.height))

    # Save final combined image
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    combined_img.save(filename)

    # remove the heatmao and colorbar images
    os.remove(screenshot_filename)
    os.remove(colorbar_filename)

    print(f"Saved final combined image: {filename}")


# Note: vertex_position_encoding and joint_position_encoding are loaded from files
# but not used in the current implementation. They can be removed or used for
# distance-based analysis if needed in the future.


def eval_angular_error(ori_r6d_pred, ori_r6d_true, expand_gt_dim=True):
    """
    :param ori_r6d_pred: (batch_size, T, D, 6) - predicted R6D for each vertex
    :param ori_r6d_true: (batch_size, T, D, 6) - ground truth R6D repeated for each vertex
    :return: ang_error: (D) - error for each vertex
    """
    # print("Debug: ori_r6d_pred.shape", ori_r6d_pred.shape)
    # print("Debug: ori_r6d_true.shape", ori_r6d_true.shape)

    batch_size, T, D, _ = ori_r6d_pred.shape
    
    if expand_gt_dim:
        ori_r6d_true = ori_r6d_true.unsqueeze(2).repeat(1, 1, ori_r6d_pred.shape[2], 1)  # (batch_size, T, D, 6)

    # Convert to rotation matrices
    ori_mat_pred = r6d_to_rotation_matrix(ori_r6d_pred.reshape(-1, 6))  # (batch_size * T * D, 3, 3)
    ori_mat_true = r6d_to_rotation_matrix(ori_r6d_true.reshape(-1, 6))  # (batch_size * T * D, 3, 3)
    
    ang_error = radian_to_degree(angle_between(ori_mat_true, ori_mat_pred).view(batch_size, T, D))  # (batch_size * T * D)
    # print("ang_error", ang_error.shape, ang_error)
    # Average over batch and time for each vertex
    mean_ang_error = torch.mean(ang_error, dim=(0, 1))  # (D)
    

    return mean_ang_error


def eval_translational_error(tr_pred, tr_true, expand_gt_dim=True):
    """
    :param tr_pred: (batch_size, T, D, 3) - predicted kinematic values for each vertex
    :param tr_true: (batch_size, T, D, 3) - ground truth kinematic values repeated for each vertex
    :return: kin_error: (D) - error for each vertex
    """
    # print("Debug: tr_pred.shape", tr_pred.shape)
    # print("Debug: tr_true.shape", tr_true.shape)
    if expand_gt_dim:
        tr_true = tr_true.unsqueeze(2).repeat(1, 1, tr_pred.shape[2], 1)  # (batch_size, T, D, 3)

    # Compute the L2 norm (Euclidean distance) between predicted and true positions
    error = torch.norm(tr_pred - tr_true, dim=-1, p=2)  # (batch_size, T, D)

    # Compute the mean translational error across the batch and time dimensions
    mean_tr_error = torch.mean(error, dim=(0, 1))  # (D,)


    return mean_tr_error


def load_model_and_data(checkpoint_path, dataset_path, num_samples):
    """Load the IMUCoCo model and prepare data loaders."""
    vertex_coordinates_with_category = torch.tensor(imu_config.vertex_coordinates_with_category).float().to(device)
    joint_coordinates_with_category = torch.tensor(imu_config.joint_coordinates_with_category).float().to(device)
    
    coordinate_max, coordinate_min = (torch.max(vertex_coordinates_with_category[:, 1:], dim=0).values,
                                      torch.min(vertex_coordinates_with_category[:, 1:], dim=0).values)

    imu_coco = IMUCoCo(
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

    imu_coco.load_state_dict(torch.load(checkpoint_path), strict=False)
    imu_coco.eval()

    
    z_ref_cache_dir = os.path.join(path_config.parsed_pose_dataset_dir, "z_ref_cache")
    train_dataloader = get_imucoco_dataloader(dataset_path=path_config.parsed_pose_dataset_dir,
                                                        datasets=['AMASS', 'DIP_IMU_train', 'XSens'],
                                                        device=device, batch_size=1, seq_len=300,
                                                        parse_vmesh=True, parse_vjoints=True, parse_imu=True,
                                                        parse_local_pose=True, parse_global_pose=True,
                                                        parse_joint_vel=True, parse_joint_pos=True,
                                                        parse_tran=True, use_joint_asp=True, joint_attr_to_root_position=True,
                                                        use_kinematic_energy_sampling=True,
                                                        use_kinematic_energy_sampling_steps_per_epoch=num_samples,
                                                        val_split=0, is_test_set=False,
                                                        presample_mesh=None,
                                                        workers=4,
                                                        prefetch_factor=None,
                                                        parse_zref=True,
                                                        z_ref_cache_dir=z_ref_cache_dir)
    
    return imu_coco, train_dataloader, vertex_coordinates_with_category

def compute_loss_maps(imu_coco, train_dataloader, vertex_coordinates_with_category, num_samples, save_dir):
    """Compute loss maps for each vertex-joint pair."""
    os.makedirs(os.path.join(save_dir, 'partial'), exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(train_dataloader)):
            if batch_idx >= num_samples:
                break
                
            imu_coco.eval()

            # Initialize error tensors for this batch
            b_j_error_gori = torch.zeros(NUM_JOINTS, NUM_VERTICES).to(device=device, non_blocking=True)
            b_j_error_lori = torch.zeros(NUM_JOINTS, NUM_VERTICES).to(device=device, non_blocking=True)
            b_j_err_vel = torch.zeros(NUM_JOINTS, NUM_VERTICES).to(device=device, non_blocking=True)
            b_j_err_pos = torch.zeros(NUM_JOINTS, NUM_VERTICES).to(device=device, non_blocking=True)
            b_j_err_rvel = torch.zeros(NUM_JOINTS, NUM_VERTICES).to(device=device, non_blocking=True)
            b_j_err_align = torch.zeros(NUM_JOINTS, NUM_VERTICES).to(device=device, non_blocking=True)
            b_j_err_kinematic = torch.zeros(NUM_JOINTS, NUM_VERTICES).to(device=device, non_blocking=True)
            
            # Load batch data
            vimu_mesh = batch_data['vimu_mesh'].to(device=device, non_blocking=True)  # (batch_size, T, 6890, 9)
            joint_vel = batch_data['joint_velocity'].to(device=device, non_blocking=True)
            joint_pos = batch_data['joint_position'].to(device=device, non_blocking=True)
            z_ref_feats = batch_data['z_ref'].to(device=device, non_blocking=True)
            gt_pose_local = batch_data['pose_local'].to(device=device, non_blocking=True)
            joint_glb_ori = batch_data['joint_orientation'].to(device=device, non_blocking=True)
            tran_mask = batch_data['tran_mask'].to(device=device, non_blocking=True)

            seq_len = batch_data['sequence_lengths']
            
            # Chop everything to seq_len
            vimu_mesh = vimu_mesh[:, :seq_len]
            joint_vel = joint_vel[:, :seq_len]
            joint_pos = joint_pos[:, :seq_len]
            z_ref_feats = z_ref_feats[:, :seq_len]
            gt_pose_local = gt_pose_local[:, :seq_len]
            joint_glb_ori = joint_glb_ori[:, :seq_len]
            
            batch_size, seq_len = vimu_mesh.shape[0], vimu_mesh.shape[1]

            # Process each joint
            for joint_idx in range(NUM_JOINTS):
                for vertex_batch_index in range(NUM_VERTICES // MESH_ITERATION_BATCH_SIZE + 1):
                    print(f"..... Processing sample {batch_idx} -- joint {joint_idx} -- vertex batch {vertex_batch_index} of {NUM_VERTICES // MESH_ITERATION_BATCH_SIZE + 1}...")

                    vertex_start = vertex_batch_index * MESH_ITERATION_BATCH_SIZE
                    vertex_end = min(vertex_batch_index * MESH_ITERATION_BATCH_SIZE + MESH_ITERATION_BATCH_SIZE, NUM_VERTICES)
                    n_vertex = vertex_end - vertex_start
                    if n_vertex == 0:
                        continue

                    vimu_mesh_partial = vimu_mesh[:, :, vertex_start: vertex_end]
                    
                    # Forward pass through IMUCoCo
                    feat_m = imu_coco.mfes[joint_idx](vimu_mesh_partial)  # (batch_size, T, D, n_hidden)
                    q_m = imu_coco.sces[joint_idx](vertex_coordinates_with_category[vertex_start: vertex_end]).unsqueeze(0)  # (batch_size, D, n_layers, 2, n_hidden)
                    feat_m2j = imu_coco.jnms[joint_idx](feat_m, q_m)  # (batch_size, T, D, n_hidden)

                    z_ref_joint = z_ref_feats[:, :, joint_idx:joint_idx+1, :]  # (batch_size, T, 1, n_hidden)
                    z_ref_joint = z_ref_joint.repeat(1, 1, n_vertex, 1)  # (batch_size, T, D, n_hidden)

                    # Forward kinematic regression for each vertex
                    vel_out, rvel_out, pos_out, gori_out, lori_out = imu_coco._forward_kr(joint_idx, feat_m2j)
    
                    # Ground truth for the specific joint
                    vel_gt = joint_vel[:, :, joint_idx] # (batch_size, T, 3)
                    pos_gt = joint_pos[:, :, joint_idx] # (batch_size, T, 3)
                    gori_gt = joint_glb_ori[:, :, joint_idx] # (batch_size, T, 6)
                    lori_gt = gt_pose_local[:, :, joint_idx] # (batch_size, T, 6)
                    rvel_gt = joint_vel[:, :, 0] # (batch_size, T, 3)
        
                    # Calculate error for each vertex to the specific joint
                    b_j_error_gori_partial = eval_angular_error(gori_out, gori_gt)  # [D] - error for each vertex
                    b_j_error_lori_partial = eval_angular_error(lori_out, lori_gt)  # [D] - error for each vertex
                    b_j_err_vel_partial = eval_translational_error(vel_out, vel_gt)  # [D] - error for each vertex
                    b_j_err_pos_partial = eval_translational_error(pos_out, pos_gt)  # [D] - error for each vertex
                    b_j_err_rvel_partial = eval_translational_error(rvel_out, rvel_gt)  # [D] - error for each vertex

                    b_j_err_kinematic_partial = imu_coco.kr_loss(k_pred=(vel_out, rvel_out, pos_out, gori_out, lori_out), 
                                                                 k_gt=(vel_gt, rvel_gt, pos_gt, gori_gt, lori_gt), 
                                                                 tran_mask=tran_mask, joint_idx=joint_idx, keep_n_mesh_dim=True)
                    
                    # Calculate alignment error for each vertex (cosine similarity)
                    b_j_err_align_partial = F.cosine_embedding_loss(feat_m2j.view(-1, feat_m2j.shape[-1]), z_ref_joint.view(-1, z_ref_joint.shape[-1]), target=torch.ones(batch_size * seq_len * n_vertex, device=feat_m2j.device), reduction='none')
                    b_j_err_align_partial = b_j_err_align_partial.view(batch_size, seq_len, n_vertex)
                    b_j_err_align_partial = b_j_err_align_partial.mean(dim=(0, 1))

                    # Store results
                    b_j_error_gori[joint_idx, vertex_start: vertex_end] = b_j_error_gori_partial
                    b_j_error_lori[joint_idx, vertex_start: vertex_end] = b_j_error_lori_partial
                    b_j_err_vel[joint_idx, vertex_start: vertex_end] = b_j_err_vel_partial
                    b_j_err_pos[joint_idx, vertex_start: vertex_end] = b_j_err_pos_partial
                    b_j_err_rvel[joint_idx, vertex_start: vertex_end] = b_j_err_rvel_partial
                    b_j_err_kinematic[joint_idx, vertex_start: vertex_end] = b_j_err_kinematic_partial
                    b_j_err_align[joint_idx, vertex_start: vertex_end] = b_j_err_align_partial

            # Save partial results for this batch
            torch.save(b_j_error_gori, os.path.join(save_dir, 'partial', f"b_j_error_gori_sample_{batch_idx}.pth"))
            torch.save(b_j_error_lori, os.path.join(save_dir, 'partial', f"b_j_error_lori_sample_{batch_idx}.pth"))
            torch.save(b_j_err_vel, os.path.join(save_dir, 'partial', f"b_j_error_vel_sample_{batch_idx}.pth"))
            torch.save(b_j_err_rvel, os.path.join(save_dir, 'partial', f"b_j_error_rvel_sample_{batch_idx}.pth"))
            torch.save(b_j_err_pos, os.path.join(save_dir, 'partial', f"b_j_error_pos_sample_{batch_idx}.pth"))
            torch.save(b_j_err_align, os.path.join(save_dir, 'partial', f"b_j_error_align_sample_{batch_idx}.pth"))
            torch.save(b_j_err_kinematic, os.path.join(save_dir, 'partial', f"b_j_error_kinematic_sample_{batch_idx}.pth"))

def aggregate_and_visualize_results(num_samples, save_dir, imu_coco):
    """Aggregate partial results and generate final visualizations."""
    os.makedirs(os.path.join(save_dir, 'err_map'), exist_ok=True)

    # Initialize accumulation tensors for all samples
    all_error_gori = torch.zeros(NUM_JOINTS, NUM_VERTICES).to(device=device, non_blocking=True)
    all_error_lori = torch.zeros(NUM_JOINTS, NUM_VERTICES).to(device=device, non_blocking=True)
    all_err_velocity = torch.zeros(NUM_JOINTS, NUM_VERTICES).to(device=device, non_blocking=True)
    all_err_rvel = torch.zeros(NUM_JOINTS, NUM_VERTICES).to(device=device, non_blocking=True)
    all_err_pos = torch.zeros(NUM_JOINTS, NUM_VERTICES).to(device=device, non_blocking=True)
    all_err_align = torch.zeros(NUM_JOINTS, NUM_VERTICES).to(device=device, non_blocking=True)
    all_err_kinematic = torch.zeros(NUM_JOINTS, NUM_VERTICES).to(device=device, non_blocking=True)
    
    # Accumulate errors across all samples
    for sample_idx in range(num_samples):
        for error_type in ['gori', 'lori', 'vel', 'rvel', 'pos', 'align', 'kinematic']:
            error_file = os.path.join(save_dir, 'partial', f"b_j_error_{error_type}_sample_{sample_idx}.pth")
            if os.path.exists(error_file):
                error = torch.load(error_file)
                if torch.isnan(error).any():
                    continue
                if error_type == 'gori':
                    all_error_gori += error * (1 / num_samples)
                elif error_type == 'lori':
                    all_error_lori += error * (1 / num_samples)
                elif error_type == 'vel':
                    all_err_velocity += error * (1 / num_samples)
                elif error_type == 'rvel':
                    all_err_rvel += error * (1 / num_samples)
                elif error_type == 'pos':
                    all_err_pos += error * (1 / num_samples)
                elif error_type == 'align':
                    all_err_align += error * (1 / num_samples)
                elif error_type == 'kinematic':
                    all_err_kinematic += error * (1 / num_samples)
            else:
                print(f"Error file {error_file} does not exist")
    
    # Compute combined loss map
    all_err_loss_map = imu_coco.lw_ph2_kr_mesh * all_err_kinematic + imu_coco.lw_ph2_align * all_err_align
   
    # Save and visualize results
    error_types = [
        ('gori', all_error_gori, 'Global Orientation Error', 'degrees', 0, 50),
        ('lori', all_error_lori, 'Local Orientation Error', 'degrees', 0, 30),
        ('velocity', all_err_velocity, 'Velocity Error', 'm/s', 0, 2.0),
        ('rvel', all_err_rvel, 'Root Velocity Error', 'm/s', 0, 1.0),
        ('pos', all_err_pos, 'Position Error', 'm', 0, 0.5),
        ('kinematic', all_err_kinematic, 'Kinematic Error', '', 0, 0.4),
        ('align', all_err_align, 'Alignment Error', '', 0, 0.2),
        ('loss_map', all_err_loss_map, 'Loss Map', '', 0, 0.2)
    ]
    
    for error_type, error_tensor, title_prefix, unit, range_low, range_high in error_types:
        torch.save(error_tensor, os.path.join(save_dir, 'err_map', f"all_error_{error_type}.pth"))
        
        for joint_idx in range(NUM_JOINTS):
            filename = os.path.join(save_dir, 'err_map', f"error_{error_type}_joint_{joint_idx}.png")
            title = f'{title_prefix} for Joint {joint_idx}'
            plot_body_mesh_heatmap_signal(
                error_tensor[joint_idx], 
                filename, 
                title=title, 
                value_title=title, 
                unit=unit, 
                range_low=range_low, 
                range_high=range_high, 
                use_scale='linear'
            )
    torch.save(all_error_gori, os.path.join(save_dir, 'err_map', f"error_gori_full.pth"))
    torch.save(all_error_lori, os.path.join(save_dir, 'err_map', f"error_lori_full.pth"))
    torch.save(all_err_velocity, os.path.join(save_dir, 'err_map', f"error_vel_full.pth"))
    torch.save(all_err_rvel, os.path.join(save_dir, 'err_map', f"error_rvel_full.pth"))
    torch.save(all_err_pos, os.path.join(save_dir, 'err_map', f"error_pos_full.pth"))
    torch.save(all_err_align, os.path.join(save_dir, 'err_map', f"error_align_full.pth"))
    torch.save(all_err_kinematic, os.path.join(save_dir, 'err_map', f"error_kinematic_full.pth"))
    torch.save(all_err_loss_map, os.path.join(save_dir, 'err_map', f"error_loss_map_full.pth"))

    print(f"Loss map generation completed. Results saved in {save_dir}")


def generate_imucoco_loss_map(checkpoint_path, dataset_path, num_samples, save_dir, random_seed=42):
    # Create output directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Load model and data
    print("Loading model and data...")
    imu_coco, train_dataloader, vertex_coordinates_with_category = load_model_and_data(
        checkpoint_path, dataset_path, num_samples
    )
    
    # # Compute loss maps
    # print(f"Computing loss maps for {num_samples} samples...")
    # compute_loss_maps(imu_coco, train_dataloader, vertex_coordinates_with_category, num_samples, save_dir)
    
    # Aggregate and visualize results
    print("Aggregating results and generating visualizations...")
    aggregate_and_visualize_results(num_samples, save_dir, imu_coco)


def main():
    parser = argparse.ArgumentParser(description="Generate IMUCoCo loss maps for vertex-to-joint transfer")
    parser.add_argument('--checkpoint_path', type=str, required=False, default=path_config.saved_imucoco_checkpoint_path,
                        help='Path to the IMUCoCo checkpoint file')
    parser.add_argument('--dataset_path', type=str, default=path_config.pose_datasets_dir,
                        help='Path to the dataset directory')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of samples to use for building loss map')
    parser.add_argument('--save_dir', type=str, default=path_config.saved_imucoco_loss_map_dir,
                        help='Directory to save results')
    args = parser.parse_args()
    
    generate_imucoco_loss_map(
        checkpoint_path=args.checkpoint_path,
        dataset_path=args.dataset_path,
        num_samples=args.num_samples,
        save_dir=args.save_dir,
    )


if __name__ == '__main__':
    main()

