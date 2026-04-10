"""IMUCoCo HAR dataloader.

Reads the per-activity .pt files produced by data_generation.process_imucoco
(stored as `IMUCoCo_{pid}_{Focus}_{Activity}.pt` inside
`parsed_pose_dataset_dir/IMUCoCo_real_imu_position_only/`) and yields fixed-length
windows of IMU data labelled with the activity class.

Designed for leave-one-(or-many)-participant-out cross validation: callers pass
`testing_subjects` and `validation_subjects` (lists of participant ids such as
'P01') and the loader splits files accordingly.
"""
import csv
import glob
import os

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from utils import imu_config
import path_config


# All 10 activity classes recorded in the IMUCoCo user study.
ACTIVITY_TO_IDX = {
    'Walking': 0, 'Running': 1, 'Golf Swing': 2, 'Shot Put': 3, 'Squats': 4,
    'Cleaning Table': 5, 'Vacuum Floor': 6, 'Drinking': 7,
    'Computer Work': 8, 'Watching TV': 9,
}


def _load_participant_dominant_hand():
    """Load participant_id -> dominant_hand from the IMUCoCo participant info CSV."""
    info = {}
    info_path = os.path.join(path_config.raw_pose_dataset_dir, 'IMUCoCo', 'participant_info.csv')
    if os.path.exists(info_path):
        with open(info_path) as f:
            for row in csv.DictReader(f):
                info[row['participant_id']] = row['dominant_hand'].strip().lower()
    return info


def _parse_filename(file_name):
    """Decode an IMUCoCo .pt filename into (pid, focus, activity).

    Files are named e.g. 'IMUCoCo_P01_Upper_Walking.pt'. The activity name itself
    can contain spaces or underscores so we recover it as everything after the
    third underscore.
    """
    base = os.path.splitext(os.path.basename(file_name))[0]  # 'IMUCoCo_P01_Upper_Walking'
    parts = base.split('_', 3)
    if len(parts) < 4 or parts[0] != 'IMUCoCo':
        return None, None, None
    return parts[1], parts[2], parts[3]  # pid, focus, activity


def _devices_for_focus(focus):
    """Return the canonical 8 device names in dataset order for a given focus."""
    region = focus.lower()
    return ['wrist', 'pocket', 'ear'] + [f'a{i}_{region}' for i in range(1, 6)]


def collate_fn(batch):
    return {
        'imu': torch.stack([b['imu'] for b in batch]),
        'activity': torch.tensor([b['activity'] for b in batch], dtype=torch.long),
        'imu_corresponding_vertices': torch.stack([b['imu_corresponding_vertices'] for b in batch]),
        'subject_id': [b['subject_id'] for b in batch],
        'focus': [b['focus'] for b in batch],
        'dominant_hand': [b['dominant_hand'] for b in batch],
    }


class IMUCoCoActivityDataset(Dataset):
    """Sliding-window IMUCoCo HAR dataset for a single instrumentation focus."""

    def __init__(self, dataset_path, focus, window_len_s=5, overlap_s=1, fps=60,
                 testing_subjects=None, validation_subjects=None, split='train'):
        if focus not in ('Upper', 'Lower', 'Torso'):
            raise ValueError(f"focus must be Upper/Lower/Torso, got {focus!r}")
        self.dataset_path = dataset_path
        self.focus = focus
        self.split = split
        self.window_size = int(window_len_s * fps)
        self.stride = int((window_len_s - overlap_s) * fps)
        self.devices = _devices_for_focus(focus)
        self.participant_dominant = _load_participant_dominant_hand()

        # Discover all .pt files for this focus.
        files = sorted(glob.glob(os.path.join(dataset_path, 'IMUCoCo_real_imu_position_only',
                                              f'IMUCoCo_*_{focus}_*.pt')))
        testing_subjects = set(testing_subjects or [])
        validation_subjects = set(validation_subjects or [])
        selected = []
        for f in files:
            pid, _, _ = _parse_filename(f)
            if pid is None:
                continue
            if split == 'train' and pid not in testing_subjects and pid not in validation_subjects:
                selected.append(f)
            elif split == 'valid' and pid in validation_subjects:
                selected.append(f)
            elif split == 'test' and pid in testing_subjects:
                selected.append(f)
        self.selected_files = selected
        self.windows = self._build_windows()

    def _build_windows(self):
        windows = []
        for fp in tqdm(self.selected_files, desc=f"har:{self.focus}:{self.split}"):
            pid, focus, activity = _parse_filename(fp)
            activity_idx = ACTIVITY_TO_IDX.get(activity)
            if activity_idx is None:
                continue
            data = torch.load(fp, weights_only=False)
            imu = data['imu']['imu']  # (T, 8, 9)
            T = imu.shape[0]
            dominant = self.participant_dominant.get(pid, 'right')
            vids_dict = (imu_config.imucoco_right_dominant_person_3_standard_vids
                         if dominant == 'right'
                         else imu_config.imucoco_left_dominant_person_3_standard_vids)
            vids = torch.tensor([vids_dict[d] for d in self.devices])
            for start in range(0, T - self.window_size + 1, max(1, self.stride)):
                end = start + self.window_size
                windows.append({
                    'imu': imu[start:end],
                    'activity': activity_idx,
                    'imu_corresponding_vertices': vids,
                    'subject_id': pid,
                    'focus': focus,
                    'dominant_hand': dominant,
                })
        print(f"  built {len(windows)} windows from {len(self.selected_files)} files "
              f"({self.focus}/{self.split})")
        return windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx]


def get_har_dataloader(focus='Upper', window_len_s=5, overlap_s=1,
                       testing_subjects=None, validation_subjects=None,
                       batch_size=64, num_workers=0,
                       dataset_path=None):
    """Build (train, valid, test) HAR dataloaders for one instrumentation focus."""
    dataset_path = dataset_path or path_config.parsed_pose_dataset_dir
    train_ds = IMUCoCoActivityDataset(dataset_path, focus, window_len_s, overlap_s,
                                      testing_subjects=testing_subjects,
                                      validation_subjects=validation_subjects, split='train')
    valid_ds = IMUCoCoActivityDataset(dataset_path, focus, window_len_s, overlap_s,
                                      testing_subjects=testing_subjects,
                                      validation_subjects=validation_subjects, split='valid')
    test_ds = IMUCoCoActivityDataset(dataset_path, focus, window_len_s, overlap_s,
                                     testing_subjects=testing_subjects,
                                     validation_subjects=validation_subjects, split='test')
    common = dict(batch_size=batch_size, collate_fn=collate_fn,
                  num_workers=num_workers, pin_memory=True)
    return (
        torch.utils.data.DataLoader(train_ds, shuffle=True, **common),
        torch.utils.data.DataLoader(valid_ds, shuffle=False, **common),
        torch.utils.data.DataLoader(test_ds, shuffle=False, **common),
    )
