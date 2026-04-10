"""IMUCoCo human activity recognition (HAR) train + eval.

Pipeline:
    1. encode IMU streams with the frozen IMUCoCo model into per-joint feature vectors
    2. feed the (N, C=128, T, V=24, M=1) tensor to a ST-GCN backbone
    3. classify with a linear head into one of 10 activity classes
    4. leave-2-participants-out 6-fold CV across the 12 participants

Training is done with the standard 3-IMU placement (wrist, pocket, ear). At test
time we evaluate the standard placement plus the 5 swap cases (a1..a5 in the
focus region) to measure how well placement transfer holds.

Defaults match the paper: lr=2e-5, batch_size=64, epochs=50, 5s windows, 1s
overlap. Note that there is no public pretrained HAR checkpoint — you must
train from scratch.

Usage (single fold smoke test):
    python evaluate_imucoco_har.py --exp_id smoke --train --epochs 1 --folds 1 --focus Upper

Run all 6 folds for all 3 focuses (paper config):
    python evaluate_imucoco_har.py --exp_id full --train
"""
import argparse
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from tqdm import tqdm

from models.stgcn import ST_GCN_18
from models.imucoco import IMUCoCo
from utils import imu_config
from utils.dataloader_har import get_har_dataloader, ACTIVITY_TO_IDX
from utils.logger import ExpLogger
import path_config


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

ACTIVITY_MODEL_NAME = 'imucoco_stgcn'
N_ACTIVITIES = len(ACTIVITY_TO_IDX)
IMUCOCO_FEATURE_DIM = 128

# Standard 3-IMU placement used during training and as the baseline test case.
TRAIN_DEVICES = ['wrist', 'pocket', 'ear']

# Six fold splits used in the paper. Each entry is a list of testing participants.
PARTICIPANT_FOLDS = [
    ['P01', 'P02'], ['P03', 'P04'], ['P05', 'P06'],
    ['P07', 'P08'], ['P09', 'P10'], ['P11', 'P12'],
]


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def _build_imucoco_model():
    vc_with_cat = torch.tensor(imu_config.vertex_coordinates_with_category).float().to(device)
    jc_with_cat = torch.tensor(imu_config.joint_coordinates_with_category).float().to(device)
    cmax = torch.max(vc_with_cat[:, 1:], dim=0).values
    cmin = torch.min(vc_with_cat[:, 1:], dim=0).values
    model = IMUCoCo(
        coordinate_origins=jc_with_cat,
        coordinate_max=cmax,
        coordinate_min=cmin,
        smpl_mesh_coordinates=vc_with_cat,
        n_hidden=IMUCOCO_FEATURE_DIM, n_kr_hidden=32,
        n_mfe_layers=2, n_jnm_layers=3, n_sce_freq=4, n_sce_emb=40,
        online_mode=False,
        joint_node_allocation_map=path_config.saved_imucoco_loss_map_path,
        joint_node_max_err_tolerance=-1,
    ).to(device)
    model.load_state_dict(torch.load(path_config.saved_imucoco_checkpoint_path,
                                     map_location=device), strict=False)
    model.freeze()
    model.eval()
    return model, vc_with_cat


class STGCNHAR(nn.Module):
    """ST-GCN backbone with a linear classifier head for HAR."""

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.backbone = ST_GCN_18(in_channels=in_channels, edge_importance_weighting=True)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        # x: (N, C, T, V, M=1) → backbone returns (N, 512, 1, 1) → flatten → (N, 512)
        feat = self.backbone(x).flatten(1)
        return self.classifier(feat)


def _select_devices_in_batch(b_imu, b_vids, vc_with_cat, device_indices):
    """Select a subset of IMU slots and look up their per-vertex coordinates."""
    sub_imu = b_imu[:, :, device_indices]
    sub_vids = b_vids[:, device_indices]
    coords = vc_with_cat[sub_vids]  # (B, D, 4) where 4 = (cat, x, y, z)
    return sub_imu, coords


def _encode_imu_with_imucoco(imu_coco_model, sub_imu, coords):
    """Run IMUCoCo over a (possibly heterogeneous) batch and return (N, C, T, V, 1)."""
    if torch.all(coords == coords[0:1]):
        imu_coco_model.set_current_device_coordinates(coords[0])
        imu_coco_model.buffer_placement_codes_with_current_devices(parallel=False)
        feat = imu_coco_model.inference_time_forward_mesh(sub_imu)
    else:
        # Heterogeneous (e.g. mixed left/right dominant); group by unique coordinate set.
        unique_coords, inverse_idx = torch.unique(coords, return_inverse=True, dim=0)
        feat = torch.empty(sub_imu.shape[0], sub_imu.shape[1], 24, IMUCOCO_FEATURE_DIM,
                           device=sub_imu.device)
        for g in range(unique_coords.shape[0]):
            sel = (inverse_idx == g).nonzero(as_tuple=False).squeeze(-1)
            imu_coco_model.set_current_device_coordinates(unique_coords[g])
            imu_coco_model.buffer_placement_codes_with_current_devices(parallel=False)
            feat[sel] = imu_coco_model.inference_time_forward_mesh(sub_imu[sel])
    # (B, T, V=24, C) → (B, C, T, V, 1)
    return feat.permute(0, 3, 1, 2).unsqueeze(-1).contiguous()


def _device_indices(case):
    return [{'wrist': 0, 'pocket': 1, 'ear': 2,
             'a1': 3, 'a2': 4, 'a3': 5, 'a4': 6, 'a5': 7}[name.split('_')[0]] for name in case]


def train_one_fold(exp_id, focus, fold_idx, epochs, batch_size, lr, save_root,
                   train_dl, valid_dl, test_dl, vc_with_cat, imucoco_model, logger):
    """Train an ST-GCN HAR head for one CV fold and return per-test-case results."""

    activity_model = STGCNHAR(in_channels=IMUCOCO_FEATURE_DIM, num_classes=N_ACTIVITIES).to(device)
    optimizer = torch.optim.Adam(activity_model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs // 2))
    ce_loss = nn.CrossEntropyLoss().to(device)

    train_device_idx = _device_indices(TRAIN_DEVICES)

    fold_dir = os.path.join(save_root, f'fold_{fold_idx}_{focus}')
    os.makedirs(fold_dir, exist_ok=True)

    best_val_f1 = -1.0
    best_state = None

    for epoch in range(epochs):
        activity_model.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()
        for batch in tqdm(train_dl, desc=f'fold{fold_idx}/{focus} train e{epoch}'):
            b_imu = batch['imu'].to(device, non_blocking=True)
            b_vids = batch['imu_corresponding_vertices'].to(device, non_blocking=True)
            b_y = batch['activity'].to(device, non_blocking=True)
            sub_imu, coords = _select_devices_in_batch(b_imu, b_vids, vc_with_cat, train_device_idx)
            feat = _encode_imu_with_imucoco(imucoco_model, sub_imu, coords)
            logits = activity_model(feat)
            loss = ce_loss(logits, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()
        train_loss = epoch_loss / max(1, n_batches)

        # Validation (also on the standard 3-IMU placement).
        activity_model.eval()
        val_pred, val_true = [], []
        with torch.no_grad():
            for batch in valid_dl:
                b_imu = batch['imu'].to(device, non_blocking=True)
                b_vids = batch['imu_corresponding_vertices'].to(device, non_blocking=True)
                b_y = batch['activity'].to(device, non_blocking=True)
                sub_imu, coords = _select_devices_in_batch(b_imu, b_vids, vc_with_cat, train_device_idx)
                feat = _encode_imu_with_imucoco(imucoco_model, sub_imu, coords)
                pred = activity_model(feat).argmax(dim=1)
                val_pred.append(pred); val_true.append(b_y)
        if val_pred:
            yp = torch.cat(val_pred).cpu().numpy()
            yt = torch.cat(val_true).cpu().numpy()
            val_acc = float((yp == yt).mean())
            val_f1 = float(f1_score(yt, yp, average='macro'))
        else:
            val_acc, val_f1 = float('nan'), float('nan')

        logger.log_msg(
            f'fold{fold_idx} {focus} epoch {epoch}: train_loss={train_loss:.4f} '
            f'val_acc={val_acc:.4f} val_f1={val_f1:.4f} ({time.time()-t0:.1f}s)',
            verbose=True)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in activity_model.state_dict().items()}

    if best_state is not None:
        activity_model.load_state_dict(best_state)
        torch.save(best_state, os.path.join(fold_dir, 'best.pth'))

    # ---- test on standard 3 + 5 swap cases ----
    activity_model.eval()
    region = focus.lower()
    test_cases = [list(TRAIN_DEVICES)]
    swap_idx = {'upper': 0, 'lower': 1, 'torso': 2}[region]
    for ai in range(1, 6):
        c = list(TRAIN_DEVICES)
        c[swap_idx] = f'a{ai}_{region}'
        test_cases.append(c)

    fold_results = []
    for case in test_cases:
        case_name = '+'.join(case)
        dev_idx = _device_indices(case)
        case_pred, case_true = [], []
        with torch.no_grad():
            for batch in test_dl:
                b_imu = batch['imu'].to(device, non_blocking=True)
                b_vids = batch['imu_corresponding_vertices'].to(device, non_blocking=True)
                b_y = batch['activity'].to(device, non_blocking=True)
                sub_imu, coords = _select_devices_in_batch(b_imu, b_vids, vc_with_cat, dev_idx)
                feat = _encode_imu_with_imucoco(imucoco_model, sub_imu, coords)
                pred = activity_model(feat).argmax(dim=1)
                case_pred.append(pred); case_true.append(b_y)
        if not case_pred:
            continue
        yp = torch.cat(case_pred).cpu().numpy()
        yt = torch.cat(case_true).cpu().numpy()
        acc = float((yp == yt).mean())
        macro_f1 = float(f1_score(yt, yp, average='macro'))
        fold_results.append({
            'fold': fold_idx, 'focus': focus, 'case': case_name,
            'accuracy': acc, 'macro_f1': macro_f1, 'n_samples': int(yt.shape[0]),
        })
        logger.log_msg(f'  test {case_name}: acc={acc:.4f} f1={macro_f1:.4f}', verbose=True)
    return fold_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', type=str, default='1')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--focus', type=str, default=None,
                        help='If set, only run this single focus (Upper/Lower/Torso).')
    parser.add_argument('--folds', type=int, default=None,
                        help='If set, only run the first N folds (for smoke testing).')
    args = parser.parse_args()

    _set_seed(args.seed)
    save_root = os.path.join(path_config.exp_out_dir, 'har',
                             f'imu_coco_har_{args.exp_id}_{ACTIVITY_MODEL_NAME}')
    os.makedirs(save_root, exist_ok=True)
    logger = ExpLogger(exp_name=f'imu_coco_har_{args.exp_id}_{ACTIVITY_MODEL_NAME}', exp_type='har')
    logger.log_meta_msg({'seed': args.seed, 'lr': args.lr,
                         'batch_size': args.batch_size, 'epochs': args.epochs})

    if not args.train:
        print('--train not provided, nothing to do (HAR has no public pretrained checkpoint).')
        return

    imucoco_model, vc_with_cat = _build_imucoco_model()

    focuses = [args.focus] if args.focus else ['Upper', 'Lower', 'Torso']
    folds = PARTICIPANT_FOLDS[:args.folds] if args.folds else PARTICIPANT_FOLDS

    all_results = []
    for focus in focuses:
        for fold_idx, test_pids in enumerate(folds):
            val_pids = folds[(fold_idx + 1) % len(folds)]
            logger.log_msg(f'\n=== {focus} fold {fold_idx}: test={test_pids} val={val_pids} ===',
                           verbose=True)
            train_dl, valid_dl, test_dl = get_har_dataloader(
                focus=focus, window_len_s=5, overlap_s=1,
                testing_subjects=test_pids, validation_subjects=val_pids,
                batch_size=args.batch_size)
            fold_results = train_one_fold(
                args.exp_id, focus, fold_idx, args.epochs, args.batch_size, args.lr,
                save_root, train_dl, valid_dl, test_dl, vc_with_cat, imucoco_model, logger)
            all_results.extend(fold_results)

    if all_results:
        df = pd.DataFrame(all_results)
        out_csv = os.path.join(save_root, 'all_results.csv')
        df.to_csv(out_csv, index=False)
        print(f'\nSaved {len(all_results)} rows to {out_csv}')


if __name__ == '__main__':
    main()
