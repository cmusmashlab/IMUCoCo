"""
This poser model is adapted from DynaIP and TransPose.
https://github.com/dx118/dynaip/blob/main/model/model.py
https://github.com/Xinyu-Yi/TransPose/blob/main/net.py
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import articulate as art
from articulate.math import r6d_to_rotation_matrix
from utils.imu_config import body_model


class RNN(nn.Module):
    def __init__(self, n_input, n_output, n_hidden, n_rnn_layer=2, bidirectional=False, dropout=0):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(n_hidden, n_hidden, n_rnn_layer, bidirectional=bidirectional, batch_first=True)
        self.linear1 = nn.Linear(n_input, n_hidden)
        self.linear2 = nn.Linear(n_hidden * (2 if bidirectional else 1), n_output)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_len=None, h=None):
        if x_len is not None:
            length = x_len
        else:
            length = [_.shape[0] for _ in x]
        total_len = x.shape[1]
        x = self.dropout(F.relu(self.linear1(x)))
        x = self.rnn(pack_padded_sequence(x, length, enforce_sorted=False, batch_first=True), h)[0]
        x = pad_packed_sequence(x, total_length=total_len, batch_first=True)[0]
        x = self.linear2(x)
        return x

    def init_hidden(self, batch_size, device):
        return (torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size, dtype=torch.float, device=device),
                torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size, dtype=torch.float, device=device))

    def forward_online(self, x, h):
        x = self.dropout(F.relu(self.linear1(x)))
        x, h = self.rnn(x, h)
        x = self.linear2(x)
        return x, h


class RNNWithInit(RNN):
    def __init__(self, n_input: int, n_output: int, n_hidden: int, n_init: int, n_rnn_layer: int
                 , bidirectional=False, dropout=0.2):
        super().__init__(n_input, n_output, n_hidden, n_rnn_layer, bidirectional, dropout)
        self.n_rnn_layer = n_rnn_layer
        self.n_hidden = n_hidden
        self.init_net = torch.nn.Sequential(
            torch.nn.Linear(n_init, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_hidden * n_rnn_layer),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden * n_rnn_layer, 2 * (2 if bidirectional else 1) * n_rnn_layer * n_hidden)
        )

    def forward(self, x, x_init, x_len=None):
        nd, nh = self.rnn.num_layers * (2 if self.rnn.bidirectional else 1), self.rnn.hidden_size
        h, c = self.init_net(x_init).view(-1, 2, nd, nh).permute(1, 2, 0, 3)
        return super(RNNWithInit, self).forward(x, x_len, (h, c))

    def forward_online(self, x, h=None):
        return super(RNNWithInit, self).forward_online(x, h)

    def init_hidden_with_states(self, x_init):
        nd, nh = self.rnn.num_layers * (2 if self.rnn.bidirectional else 1), self.rnn.hidden_size
        hh = self.init_net(x_init).view(-1, 2, nd, nh).permute(1, 2, 0, 3)
        return hh[0], hh[1]

class SubPoser(nn.Module):
    def __init__(self, n_input, v_output, p_output, n_hidden, num_layer, dropout, n_glb):
        super(SubPoser, self).__init__()

        self.rnn1 = RNNWithInit(n_init=v_output, n_input=n_input - n_glb,
                                n_hidden=n_hidden, n_output=v_output,
                                n_rnn_layer=num_layer, dropout=dropout)
        self.rnn2 = RNNWithInit(n_init=p_output, n_input=n_input + v_output,
                                n_hidden=n_hidden, n_output=p_output,
                                n_rnn_layer=num_layer, dropout=dropout)

    def forward(self, x_part, x_glb, v_init, p_init, seq_len):
        v = self.rnn1(x_part, v_init, seq_len)
        p = self.rnn2(torch.cat([x_part, x_glb, v], dim=-1), p_init, seq_len)
        return v, p

    def forward_online(self, x_part, x_glb, h_part=None, h_glb=None):
        v, h_part = self.rnn1.forward_online(x_part, h_part)
        p, h_glb = self.rnn2.forward_online(torch.cat([x_part, x_glb, v], dim=-1), h_glb)
        return v, p, h_part, h_glb


class Poser(nn.Module):
    def __init__(self, joint_feature_dim=128, n_hidden=256, n_glb=256, num_layer=2,
                 n_total_devices=24, load_physics_optimizer=False, load_tran_module=False):
        super(Poser, self).__init__()

        self.n_glb = n_glb

        self.glb = RNN(n_input=n_total_devices * joint_feature_dim, n_output=self.n_glb, n_hidden=n_hidden, n_rnn_layer=num_layer, dropout=0.2)

        self.upper_in_joints = [0, 18, 19, 16, 17, 13, 14, 20, 21, 22, 23] # input joints
        self.upper_v_joints = [20, 21]  # joints to regress to velocity
        self.upper_p_joints = [18, 19, 16, 17, 13, 14]  # joints to regress to pose (r6d)

        self.lower_in_joints = [0, 4, 5, 15, 7, 8, 9, 10, 1, 2]   # input joints
        self.lower_v_joints = [4, 5, 10, 11]  # joints to regress to velocity
        self.lower_p_joints = [4, 5, 7, 8, 1, 2]   # joints to regress to pose (r6d)

        self.torso_in_joints = [15, 12, 9, 6, 3, 0]  # input joints
        self.torso_v_joints = [15, 3]  # joints to regress to velocity
        self.torso_p_joints = [15, 12, 9, 6, 3, 0]  # joints to regress to pose (r6d)

        self.v_joints_all = self.upper_v_joints + self.lower_v_joints + self.torso_v_joints
        self.p_joints_all = self.upper_p_joints + self.lower_p_joints + self.torso_p_joints


        # pose regressors for upper, lower, and torso regions
        self.posers = nn.ModuleList([SubPoser(n_input=len(self.upper_in_joints) * joint_feature_dim + self.n_glb, v_output=3 * len(self.upper_v_joints), p_output=6 * len(self.upper_p_joints),
                                              n_hidden=n_hidden, num_layer=num_layer, dropout=0.2, n_glb=self.n_glb),  # upper regions
                                     SubPoser(n_input=len(self.lower_in_joints) * joint_feature_dim + self.n_glb, v_output=3 * len(self.lower_v_joints), p_output=6 * len(self.lower_p_joints),
                                              n_hidden=n_hidden, num_layer=num_layer, dropout=0.2, n_glb=self.n_glb),  # lower regions
                                     SubPoser(n_input=len(self.torso_in_joints) * joint_feature_dim + self.n_glb, v_output=3 * len(self.torso_v_joints), p_output=6 * len(self.torso_p_joints),
                                              n_hidden=n_hidden, num_layer=num_layer, dropout=0.2, n_glb=self.n_glb)])  # torso regions

        self.tran_b1_joints = [7, 8, 10, 11, 4, 5, 0, 12, 15, 20, 21]  # joints to regress to contact labels

        self.contact_reg = RNN(len(self.tran_b1_joints) * 9 + len(self.tran_b1_joints) * joint_feature_dim, 2, n_hidden // 2, n_rnn_layer=2, dropout=0)
        self.vel_reg = RNNWithInit(24 * 9 + 24 * joint_feature_dim, 3, n_hidden, n_rnn_layer=2, n_init=72, dropout=0)


        self.prob_threshold = (0.3, 0.6)
        self.gravity_velocity = torch.tensor([0, -0.018, 0])
        self.saved_previous_joint_pos = None
        self.body_model = body_model
        j0, _ = self.body_model.get_zero_pose_joint_and_vertex()
        self.floor_y = j0[10:12, 1].min().item()
        self.velocity_scale = 3

        self.ignored_joints = [10, 11, 20, 21, 22, 23]

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_poser(self):
        # freeze the pose part (but leave the translation part free)
        for param in self.posers.parameters():
            param.requires_grad = False
        for param in self.glb.parameters():
            param.requires_grad = False
        
    def _fill_reduced_glb_pose(self, glb_pose_pred):
        glb_pose_pred[:, :, 10, :] = glb_pose_pred[:, :, 7, :]
        glb_pose_pred[:, :, 11, :] = glb_pose_pred[:, :, 8, :]
        glb_pose_pred[:, :, 20, :] = glb_pose_pred[:, :, 18, :]
        glb_pose_pred[:, :, 21, :] = glb_pose_pred[:, :, 19, :]
        glb_pose_pred[:, :, 22, :] = glb_pose_pred[:, :, 20, :]
        glb_pose_pred[:, :, 23, :] = glb_pose_pred[:, :, 21, :]
        return glb_pose_pred
        
    def _fill_ignored_local_pose(self, local_pose_pred):
        neutral_pose = torch.eye(3, device=local_pose_pred.device).expand(local_pose_pred.shape[0], local_pose_pred.shape[1], len(self.ignored_joints), 3, 3)
        local_pose_pred[:, :, self.ignored_joints, :] = neutral_pose
        return local_pose_pred

    def pose_loss_func(self, p_out, v_out, local_pred, pos_pred, glb_pose_gt, local_pose_gt, vel_gt, seq_len=None):
        batch_size, T, _, _ = glb_pose_gt.shape
        if seq_len is None:
            seq_len = torch.full((batch_size,), T, dtype=torch.long, device=glb_pose_gt.device)
        else:
            seq_len = seq_len.to(glb_pose_gt.device)

        p_out_gt = glb_pose_gt[:, :, self.p_joints_all]
        v_out_gt = vel_gt[:, :, self.v_joints_all]
        pos_gt = self.body_model.forward_kinematics(local_pose_gt.view(-1, 24, 3, 3), calc_mesh=False)[1].view(batch_size, T, 24, 3)
        pos_gt = pos_gt[:, :, self.p_joints_all]
        pos_pred = pos_pred[:, :, self.p_joints_all]
        local_pred = local_pred[:, :, self.p_joints_all]
        local_pose_gt = local_pose_gt[:, :, self.p_joints_all]

        mask = torch.arange(T, device=p_out.device).expand(batch_size, T) < seq_len.unsqueeze(1)  # (batch_size, T)

        # glb pose loss
        pose_loss_glb = F.mse_loss(p_out[mask].flatten(1), p_out_gt[mask].flatten(1))

        # velocity loss
        vel_loss = F.mse_loss(v_out[mask][:, 1:].flatten(1), v_out_gt[mask][:, 1:].flatten(1))  # the loss from the velocity from pose part

        # position loss
        pos_loss = F.mse_loss(pos_pred[mask].flatten(1), pos_gt[mask].flatten(1))

        # jitter loss
        jerk = pos_pred[:, 3:, :] - 3 * pos_pred[:, 2:-1, :] + 3 * pos_pred[:, 1:-2, :] - pos_pred[:, :-3, :]
        jerk_masked = jerk[mask[:, T - 3]]
        jitter_loss = 1e-3 * torch.norm(jerk_masked, p=2, dim=3).mean()

        total_loss = (pose_loss_glb + vel_loss + pos_loss + jitter_loss)

        print(f"Total Pose Loss: {total_loss.item()}")

        return total_loss, pose_loss_glb.item()


    def tran_loss_func(self, contact_pred, contact_gt, rvel_pred, rvel_gt, tran_mask, seq_len=None):
        batch_size, T, _ = contact_gt.shape
        if seq_len is None:
            seq_len = torch.full((batch_size,), T, dtype=torch.long, device=contact_gt.device)
        else:
            seq_len = seq_len.to(contact_gt.device)

        mask = torch.arange(T, device=contact_gt.device).expand(batch_size, T) < seq_len.unsqueeze(1)  # (batch_size, T)

        root_velocity_loss = 0
        rvel_gt = rvel_gt / self.velocity_scale

        tran_mask_indices = torch.nonzero(tran_mask).squeeze()

        if tran_mask_indices.numel() > 0:
            for n in [1, 3, 9, 27]:
                # Stride across frames for the given interval
                start_indices = torch.arange(0, T, n, device=rvel_pred.device)
                end_indices = torch.minimum(start_indices + n, torch.tensor(rvel_pred.shape[1], device=rvel_pred.device))
                valid_slices = [
                    (end < seq_len[tran_mask_indices]).nonzero().squeeze()
                    for end in end_indices
                ]
                pred_slices = [rvel_pred[tran_mask_indices, start:end][valid].view(-1, 3) for valid, start, end in zip(valid_slices, start_indices, end_indices)]
                gt_slices = [rvel_gt[tran_mask_indices, start:end][valid].view(-1, 3) for valid, start, end in zip(valid_slices, start_indices, end_indices)]

                # Compute loss for this interval
                if len(pred_slices) > 0 and len(gt_slices) > 0:
                    pred_concat = torch.cat(pred_slices, dim=0)
                    gt_concat = torch.cat(gt_slices, dim=0)
                    root_velocity_loss += F.mse_loss(pred_concat, gt_concat)
        else:
            root_velocity_loss = 0

        if tran_mask_indices.numel() > 0:
            contact_loss = F.binary_cross_entropy_with_logits(contact_pred[mask].flatten(0), contact_gt[mask].flatten(0))
        else:
            contact_loss = 0
        total_loss = torch.tensor(0.0, device=contact_pred.device)
        total_loss += contact_loss
        total_loss += root_velocity_loss
        print(f"Total Tran Loss: {total_loss.item()}")

        return total_loss

    def forward(self, x, v_init, glb_init, seq_len, compute_tran=None):
        batch_size, T, n_joints, joint_feat_dim = x.shape
        s_glb = self.glb(x.flatten(2), seq_len)

        # upper body:
        x_upper = x[:, :, self.upper_in_joints].flatten(2)
        v_i_upper = v_init[:, self.upper_v_joints].flatten(1)
        p_i_upper = glb_init[:, self.upper_p_joints].flatten(1)
        v_upper, p_upper = self.posers[0](x_upper, s_glb, v_i_upper, p_i_upper, seq_len)
        v_upper = v_upper.view(batch_size, T, len(self.upper_v_joints), 3)
        p_upper = p_upper.view(batch_size, T, len(self.upper_p_joints), 6)

        # lower body:
        x_lower = x[:, :, self.lower_in_joints].flatten(2)
        v_i_lower = v_init[:, self.lower_v_joints].flatten(1)
        p_i_lower = glb_init[:, self.lower_p_joints].flatten(1)
        v_lower, p_lower = self.posers[1](x_lower, s_glb, v_i_lower, p_i_lower, seq_len)
        v_lower = v_lower.view(batch_size, T, len(self.lower_v_joints), 3)
        p_lower = p_lower.view(batch_size, T, len(self.lower_p_joints), 6)

        # torso:
        x_torso = x[:, :, self.torso_in_joints].flatten(2)
        v_i_torso = v_init[:, self.torso_v_joints].flatten(1)
        p_i_torso = glb_init[:, self.torso_p_joints].flatten(1)
        v_torso, p_torso = self.posers[2](x_torso, s_glb, v_i_torso, p_i_torso, seq_len)
        v_torso = v_torso.view(batch_size, T, len(self.torso_v_joints), 3)
        p_torso = p_torso.view(batch_size, T, len(self.torso_p_joints), 6)

        v_out = torch.cat([v_upper, v_lower, v_torso], dim=2)
        p_out = torch.cat([p_upper, p_lower, p_torso], dim=2)

        # sort the outputs by the smpl order
        glb_rotation_r6d = torch.zeros(batch_size, T, 24, 6, device=x.device)
        glb_rotation_r6d[:, :, self.upper_p_joints, :] = p_upper
        glb_rotation_r6d[:, :, self.lower_p_joints, :] = p_lower
        glb_rotation_r6d[:, :, self.torso_p_joints, :] = p_torso
        glb_rotation_r6d = self._fill_reduced_glb_pose(glb_rotation_r6d)

        glb_rotation = r6d_to_rotation_matrix(glb_rotation_r6d.reshape(-1, 6)).view(batch_size, T, 24, 3, 3)
        ik_rotation = self.body_model.inverse_kinematics_R(glb_rotation.view(batch_size * T, 24, 3, 3))
        pose_local_pred = ik_rotation.view(batch_size, T, 24, 3, 3)
        pose_local_pred = self._fill_ignored_local_pose(pose_local_pred)
        _, joint_pos_pred = self.body_model.forward_kinematics(pose_local_pred.view(-1, 24, 3, 3))
        joint_pos_pred = joint_pos_pred.view(batch_size, T, 24, 3)


        tran_input = torch.cat([x, joint_pos_pred, glb_rotation_r6d], dim=3)
        ft_contact_pred = self.contact_reg(tran_input[:, :, self.tran_b1_joints].flatten(2), seq_len)
        root_vel_pred = self.vel_reg(tran_input.flatten(2), v_init.flatten(1), seq_len)
        
        if compute_tran == 'transpose':
            # assume 1 batch size
            contact_probability = ft_contact_pred[0]
            velocity = root_vel_pred[0]
            joint_pos_pred = joint_pos_pred[0]

            # b1, velocity by contact: inverse of the foot velocity (i.e., pushing a foot backward meaning body is moving forward)  
            tran_b1_vel = -1 * art.math.lerp(
                torch.cat((torch.zeros(1, 3, device=joint_pos_pred.device), joint_pos_pred[1:, 10] - joint_pos_pred[:-1, 10]), dim=0),
                torch.cat((torch.zeros(1, 3, device=joint_pos_pred.device), joint_pos_pred[1:, 11] - joint_pos_pred[:-1, 11]), dim=0),
                (contact_probability.max(dim=1)).indices.view(-1, 1)
            ) + self.gravity_velocity.to(joint_pos_pred.device)

            # b2, velocity by direct estimation
            tran_b2_vel = velocity * self.velocity_scale / 60  # to world space
            weight = self._prob_to_weight(contact_probability.max(dim=1).values.sigmoid()).view(-1, 1)
            velocity = art.math.lerp(tran_b2_vel, tran_b1_vel, weight)

            # remove penetration
            current_root_y = 0
            for i in range(velocity.shape[0]):
                current_foot_y = current_root_y + joint_pos_pred[0, 10:11].min().item()
                if current_foot_y + velocity[i, 1].item() <= self.floor_y:
                    velocity[i, 1] = self.floor_y - current_foot_y
                current_root_y += velocity[i, 1].item()

            return glb_rotation, pose_local_pred, self.velocity_to_root_position(velocity)

        else:
            return p_out, v_out, pose_local_pred, joint_pos_pred, root_vel_pred, ft_contact_pred


    def init_hidden_states(self, v_init, glb_init):
        batch_size = v_init.shape[0]

        h_glb = self.glb.init_hidden(batch_size, device=v_init.device)

        h_upper_part = self.posers[0].rnn1.init_hidden_with_states(v_init[:, self.upper_v_joints].flatten(1))
        h_upper_glb = self.posers[0].rnn2.init_hidden_with_states(glb_init[:, self.upper_p_joints].flatten(1))

        h_lower_part = self.posers[1].rnn1.init_hidden_with_states(v_init[:, self.lower_v_joints].flatten(1))
        h_lower_glb = self.posers[1].rnn2.init_hidden_with_states(glb_init[:, self.lower_p_joints].flatten(1))

        h_torso_part = self.posers[2].rnn1.init_hidden_with_states(v_init[:, self.torso_v_joints].flatten(1))
        h_torso_glb = self.posers[2].rnn2.init_hidden_with_states(glb_init[:, self.torso_p_joints].flatten(1))

        h_tran_contact = self.contact_reg.init_hidden(batch_size, device=v_init.device)
        h_rvel = self.vel_reg.init_hidden_with_states(v_init.view(batch_size, -1))

        return h_glb, h_upper_glb, h_upper_part, h_lower_glb, h_lower_part, h_torso_part, h_torso_glb, h_tran_contact, h_rvel

    def forward_online(self, x, h_pose, current_tran=None, compute_tran=None):
        batch_size, T, n_joints, joint_feat_dim = x.shape
        h_glb, h_upper_glb, h_upper_part, h_lower_glb, h_lower_part, h_torso_part, h_torso_glb, h_tran_contact, h_rvel= h_pose
        s_glb, h_glb = self.glb.forward_online(x.flatten(2), h_glb)

        # upper body:
        x_upper = x[:, :, self.upper_in_joints].flatten(2)
        v_upper, p_upper, h_upper_part, h_upper_glb = self.posers[0].forward_online(x_upper, s_glb, h_upper_part, h_upper_glb)

        p_upper = p_upper.view(batch_size, T, len(self.upper_p_joints), 6)

        # lower body:
        x_lower = x[:, :, self.lower_in_joints].flatten(2)
        v_lower, p_lower, h_lower_part, h_lower_glb = self.posers[1].forward_online(x_lower, s_glb, h_lower_part, h_lower_glb)

        p_lower = p_lower.view(batch_size, T, len(self.lower_p_joints), 6)

        # torso:
        x_torso = x[:, :, self.torso_in_joints].flatten(2)
        v_torso, p_torso, h_torso_part, h_torso_glb = self.posers[2].forward_online(x_torso, s_glb, h_torso_part, h_torso_glb)

        p_torso = p_torso.view(batch_size, T, len(self.torso_p_joints), 6)


        glb_rotation_r6d = torch.zeros(batch_size, T, 24, 6, device=x.device)
        glb_rotation_r6d[:, :, self.upper_p_joints, :] = p_upper
        glb_rotation_r6d[:, :, self.lower_p_joints, :] = p_lower
        glb_rotation_r6d[:, :, self.torso_p_joints, :] = p_torso
        glb_rotation_r6d = self._fill_reduced_glb_pose(glb_rotation_r6d)

        glb_rotation = r6d_to_rotation_matrix(glb_rotation_r6d.reshape(-1, 6)).view(batch_size, T, 24, 3, 3)
        ik_rotation = self.body_model.inverse_kinematics_R(glb_rotation.view(batch_size * T, 24, 3, 3))
        pose_local_pred = ik_rotation.view(batch_size, T, 24, 3, 3)
        pose_local_pred = self._fill_ignored_local_pose(pose_local_pred)
        _, joint_pos_pred = self.body_model.forward_kinematics(pose_local_pred.view(-1, 24, 3, 3))
        joint_pos_pred = joint_pos_pred.view(batch_size, T, 24, 3)
      
        tran_input = torch.cat([x, joint_pos_pred, glb_rotation_r6d], dim=3)
        ft_contact_pred, h_tran_contact = self.contact_reg.forward_online(tran_input[:, :, self.tran_b1_joints].flatten(2), h_tran_contact)
        root_vel_pred, h_rvel = self.vel_reg.forward_online(tran_input.flatten(2), h_rvel)
        
        if compute_tran == 'transpose':
            # assume 1 batch size
            contact_probability = ft_contact_pred[0]
            velocity = root_vel_pred[0]
            joint_pos_pred = joint_pos_pred[0]


            if self.saved_previous_joint_pos is not None:
                joint_pos_pred_to_prev = joint_pos_pred - self.saved_previous_joint_pos
            else:
                joint_pos_pred_to_prev = joint_pos_pred - joint_pos_pred  # to make it 0
            self.saved_previous_joint_pos = joint_pos_pred

            # b1, velocity by contact: inverse of the foot velocity (i.e., pushing a foot backward meaning body is moving forward)  
            tran_b1_vel = (-1) * art.math.lerp(
                joint_pos_pred_to_prev[:1, 10],
                joint_pos_pred_to_prev[:1, 11],
                (contact_probability.max(dim=1)).indices.view(-1, 1)
                ) + self.gravity_velocity.to(joint_pos_pred.device)

            # b2, velocity by direct estimation
            tran_b2_vel = velocity * self.velocity_scale / 60  # to world space
            weight = self._prob_to_weight(contact_probability.max(dim=1).values.sigmoid()).view(-1, 1)
            velocity = art.math.lerp(tran_b2_vel, tran_b1_vel, weight)

            # remove penetration
            if current_tran is None:
                current_root_y = 0
            else:
                current_root_y = current_tran[0, 1].item()
            for i in range(velocity.shape[0]):
                current_foot_y = current_root_y + joint_pos_pred[0, 10:11].min().item()
                if current_foot_y + velocity[i, 1].item() <= self.floor_y:
                    velocity[i, 1] = self.floor_y - current_foot_y
                current_root_y += velocity[i, 1].item()
            tran = self.velocity_to_root_position(velocity, current_tran=current_tran)
            return pose_local_pred, glb_rotation, tran, (h_glb, h_upper_glb, h_upper_part, h_lower_glb, h_lower_part, h_torso_part, h_torso_glb, h_tran_contact, h_rvel)

        else:
            return pose_local_pred, glb_rotation, root_vel_pred, ft_contact_pred, (h_glb, h_upper_glb, h_upper_part, h_lower_glb, h_lower_part, h_torso_part, h_torso_glb, h_tran_contact, h_rvel)

    def _prob_to_weight(self, p):
        return (p.clamp(self.prob_threshold[0], self.prob_threshold[1]) - self.prob_threshold[0]) / \
            (self.prob_threshold[1] - self.prob_threshold[0])

    @staticmethod
    def velocity_to_root_position(velocity, current_tran=None):
        tran = torch.stack([velocity[:i + 1].sum(dim=0) for i in range(velocity.shape[0])])
        if current_tran is not None:
            tran = tran + current_tran
        return tran
