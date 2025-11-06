import numpy as np
import torch
import sys
import os
from torch.utils.data import Sampler
# Add the parent directory to the path so we can import utils modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.imu_config import category_inverse_density, hop_distances, smpl_vertices_2_bone_direction_joints, smpl_vertex_id_2_category_idx

class HopDecaySampler:
    def __init__(self, max_stage=10, device=None):
        self.max_stage = max_stage
        self.device = device if device is not None else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.vertex_to_joint = np.array(smpl_vertices_2_bone_direction_joints)  # (6890,)
        self.vertex_to_category = np.array(smpl_vertex_id_2_category_idx)  # (6890,)
        self.hop_matrix = hop_distances  # (24, 24)  # joint-to-joint hops
        self.n_vertices = len(self.vertex_to_joint)
        self.n_joints = self.hop_matrix.shape[0]

    def get_vertex_hops_to_joint(self, joint_idx):
        return self.hop_matrix[self.vertex_to_joint, joint_idx]  # (n_vertices,)

    def get_decay_weights(self, joint_idx, stage, gamma=0.5):
        hops = self.get_vertex_hops_to_joint(joint_idx)
        decay = np.exp(-gamma * np.maximum(hops - stage, 0))
        decay = decay * category_inverse_density[self.vertex_to_category[list(range(6890))]]
        decay = decay / decay.sum()
        return decay

    def sample_vertices(self, joint_idx, stage, n_samples=384, gamma=0.5, return_weights=False):
        weights = self.get_decay_weights(joint_idx, stage, gamma)
        sampled_vertices = np.random.choice(self.n_vertices, size=n_samples, replace=False, p=weights)
        tensor_vertices = torch.tensor(sampled_vertices).long().to(self.device)
        if return_weights:
            return tensor_vertices, weights
        return tensor_vertices 


class WeightedDataSampler(Sampler):
    """Weighted sampling without replacement, but order shuffled to remove bias."""
    def __init__(self, weights, num_samples, replacement=False):
        self.weights = torch.as_tensor(weights, dtype=torch.float)
        self.num_samples = num_samples
        self.replacement = replacement

    def __iter__(self):
        # weighted draw
        print("weights", len(self.weights))
        print("num_samples", self.num_samples)
        print("replacement", self.replacement)
        indices = torch.multinomial(self.weights, self.num_samples, self.replacement)
        # uniform shuffle to break correlation between weight and position
        perm = torch.randperm(len(indices))
        yield from indices[perm].tolist()

    def __len__(self):
        return self.num_samples