import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call
from torch._functorch.functional_call import stack_module_state

"""
We used below LSTM layers for online implementation with parallelization over joint nodes,
as pytorch's LSTM layer does not support vmap.
"""
class MyLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear_i = nn.Linear(input_size + hidden_size, 4 * hidden_size)

    def forward(self, x, hidden):
        h_prev, c_prev = hidden
        combined = torch.cat([x, h_prev], dim=-1)

        gates = self.linear_i(combined)
        i, f, g, o = gates.chunk(4, dim=-1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_t = f * c_prev + i * g
        h_t = o * torch.tanh(c_t)
        return h_t, c_t
class MyLSTMLayers(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        cells = []
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            cells.append(MyLSTMCell(layer_input_size, hidden_size))
        self.lstm_cells = nn.ModuleList(cells)

    def forward(self, x, hiddens):
        _, seq_len, _ = x.shape
        outputs = []
        for t in range(seq_len):
            input_t = x[:, t, :]
            new_hiddens = []
            for layer_idx, cell in enumerate(self.lstm_cells):
                h, c = hiddens[layer_idx][0], hiddens[layer_idx][1]
                h, c = cell(input_t, (h, c))
                new_hiddens.append(torch.stack([h, c], dim=0))
                input_t = h
            hiddens = torch.stack(new_hiddens, dim=0)
            outputs.append(hiddens[-1][0])
        outputs = torch.stack(outputs, dim=1)
        return outputs, hiddens

"""
Motion Feature Encoder (MFE) is a module that encodes the motion features of the input signal, just standard LSTM layers.
"""
class MotionFeatureEncoder(nn.Module):
    def __init__(self, n_channel, n_hidden=256, n_rnn_layer=2, online_mode=False):
        super().__init__()
        self.linear = nn.Linear(n_channel, n_hidden)
        if online_mode:
            self.rnn = MyLSTMLayers(n_hidden, n_hidden, n_rnn_layer)
        else:
            self.rnn = nn.LSTM(n_hidden, n_hidden, n_rnn_layer, bidirectional=False, batch_first=True, dropout=0)
        self.online_mode = online_mode
        self.n_rnn_layer = n_rnn_layer
        self.n_hidden = n_hidden

    def init_hidden(self, batch_size, device=torch.device('cpu')):
        if self.online_mode:
            return torch.zeros(self.n_rnn_layer, 2, batch_size, self.n_hidden, device=device)
        else:
            return (torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size, dtype=torch.float, device=device),
                    torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size, dtype=torch.float, device=device))

    def forward(self, x, h=None):
        if self.online_mode:
            return self._forward_online(x, h)
        else:
            return self._forward_offline(x, h)

    def _forward_offline(self, x, h=None):
        batch_size, T, D, n_channel = x.shape
        x = x.permute(0, 2, 1, 3).reshape(-1, T, n_channel)
        x = self.linear(x)
        if h is None:
            h = self.init_hidden(x.size(0), device=x.device)
        x, h = self.rnn(x, h)
        x = x.view(batch_size, D, T, -1).transpose(1, 2)
        return x

    def _forward_online(self, x, h=None):
        assert self.mode == 'online', "mode set the flag to be 'online' to use online mode"
        batch_size, T, D, n_channel = x.shape
        if h is None:
            h = self.init_hidden(batch_size * D, device=x.device)
        x = x.permute(0, 2, 1, 3).reshape(batch_size * D, T, n_channel)
        x = self.linear(x)
        x, h = self.rnn(x, h)
        x = x.view(batch_size, D, T, -1).permute(0, 2, 1, 3)
        return x, h


"""
Sensor Coordinate Encoder (SCE) is a module that encodes the sensor coordinates of the input signas, 
that informs the subsequent joint node modulator (JNM) about the location of the sensor.
"""
class SensorCoordinateEncoder(nn.Module):
    def __init__(self,
                 coordinate_origin,
                 coordinate_max,
                 coordinate_min,
                 n_freq=4,
                 n_emb=40,
                 n_hidden=256,
                 n_layers=3):
        super(SensorCoordinateEncoder, self).__init__()

        self.emb_layer = nn.Embedding(24, n_emb)
        self.n_freq_enc = 2 * n_freq * 3
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.freq_bands = 2.0 ** torch.linspace(0, n_freq - 1, n_freq).to(coordinate_origin.device)
        self.mlp_layers = nn.ModuleList([])

        for i in range(n_layers):
            if i == 0:
                in_dim = self.n_freq_enc + n_emb
            else:
                in_dim = 2 * n_hidden
            self.mlp_layers.append(
                nn.Sequential(
                    nn.Linear(in_dim, 2 * n_hidden),
                    nn.ReLU(),
                )
            )

        # Store the min and max values for normalization
        k = coordinate_origin[0].int()
        min_vals = coordinate_min - coordinate_origin[1:]
        max_vals = coordinate_max - coordinate_origin[1:]
        coordinate_origin = coordinate_origin[1:]

        self.register_buffer("min_vals", min_vals)
        self.register_buffer("max_vals", max_vals)
        self.register_buffer("coordinate_origin", coordinate_origin)
        self.register_buffer("k", k)

    def _standardize_coordinates(self, r):
        with torch.no_grad():
            return (r - self.coordinate_origin) / (self.max_vals - self.min_vals)

    def _get_freq_encoding(self, r):
        with torch.no_grad():
            freq = r.unsqueeze(-1) * self.freq_bands  # Multiply input by each frequency
            sin_val = torch.sin(2 * np.pi * freq)
            cos_val = torch.cos(2 * np.pi * freq)
            freq_enc = torch.cat([sin_val, cos_val], dim=-1)
            freq_enc = freq_enc.flatten(1)
            return freq_enc

    def init_weight_and_bias(self):
        # initialize bias weights so that the gamma is 1 and beta is 0 for at its joint origin
        with torch.no_grad():
            category_emb = self.emb_layer(self.k.unsqueeze(0))
            r_st = self._standardize_coordinates(self.coordinate_origin.unsqueeze(0))    # all zeros here
            assert torch.allclose(r_st, torch.zeros_like(r_st)), "r_st should be all zeros"
            freq_enc = self._get_freq_encoding(r_st)
            features = torch.cat([freq_enc, category_emb], dim=1)

            zero_feature = torch.zeros(2 * self.n_hidden, device=self.coordinate_origin.device)
            zero_feature[:self.n_hidden] = 1.0
            for _, layer in enumerate(self.mlp_layers):
                weight = torch.linalg.lstsq(features.unsqueeze(0), (zero_feature - layer[0].bias.data).unsqueeze(0)).solution.squeeze(0)
                layer[0].weight.data = weight.T
                # check forward
                features_out = layer(features)
                assert torch.allclose(features_out, zero_feature, atol=1e-5), "features out should be 1s and 0s for the joint origin"
                features = zero_feature

    def forward(self, sensor_coordinate):
        category_emb = self.emb_layer(sensor_coordinate[:, 0].long())
        r_st = self._standardize_coordinates(sensor_coordinate[:, 1:])
        freq_enc = self._get_freq_encoding(r_st)
        features_in = torch.cat([freq_enc, category_emb], dim=-1)
        features = features_in
        features_out = []
        # extract placement codes at each layer
        for _, layer in enumerate(self.mlp_layers):
            features = layer(features)
            features_out_layer_i = torch.stack([features[:, :self.n_hidden], features[:, self.n_hidden:]], dim=1)
            features_out.append(features_out_layer_i)
        features_out = torch.stack(features_out, dim=1)
        return features_out

"""
Joint Node Modulator (JNM) is a module that modulates the motion features of the input signal based on the placement code of extracted from the sensor coordinates.
"""
class JointNodeModulator(nn.Module):
    def __init__(self, n_hidden=128, n_hidden_layer=3, online_mode=False):
        super(JointNodeModulator, self).__init__()

        self.online_mode = online_mode
        self.rnn_layers = nn.ModuleList([])

        for i in range(n_hidden_layer):
            if online_mode:
                rnn = MyLSTMLayers(n_hidden, n_hidden, 1)
                self.rnn_layers.append(rnn)
            else:
                rnn = nn.LSTM(n_hidden, n_hidden, 1, bidirectional=False, batch_first=True, dropout=0)
                self.rnn_layers.append(rnn)

        self.n_hidden = n_hidden

    def init_hidden(self, batch_size, device=torch.device('cpu')):
        if self.online_mode:
            return torch.zeros(len(self.rnn_layers), 2, batch_size, self.n_hidden, device=device)
        else:
            return (torch.zeros(1, batch_size, self.n_hidden, dtype=torch.float, device=device),
                    torch.zeros(1, batch_size, self.n_hidden, dtype=torch.float, device=device))

    def forward(self, feat_m, placement_codes, h=None):
        if self.online_mode:
            return self._forward_online(feat_m, placement_codes, h)
        else:
            return self._forward_offline(feat_m, placement_codes, h)

    def _forward_offline(self, feat_m, placement_codes, h=None):
        batch_size, T, D, n_hidden = feat_m.shape
        feat_m2j_in = feat_m.permute(0, 2, 1, 3).contiguous().view(batch_size * D, T, -1)
        feat_m2j = feat_m2j_in
        if h is None:
            h = self.init_hidden(batch_size * D, device=feat_m.device)

        # fusing the placement codes and motion features at each layer
        for layer_idx, rnn in enumerate(self.rnn_layers):
            gamma_feat, beta_feat = placement_codes[:, :, layer_idx, 0], placement_codes[:, :, layer_idx, 1]
            gamma_feat = gamma_feat.unsqueeze(2).expand(batch_size, D, T, n_hidden).view(batch_size * D, T, n_hidden)
            beta_feat = beta_feat.unsqueeze(2).expand(batch_size, D, T, n_hidden).view(batch_size * D, T, n_hidden)
            feat_m2j = gamma_feat * feat_m2j + beta_feat 
            feat_m2j, _ = rnn(feat_m2j, h)

        feat_m2j = feat_m2j.view(batch_size, D, T, n_hidden).permute(0, 2, 1, 3)
        return feat_m2j

    def _forward_online(self, feat_m, placement_codes, h=None):
        assert self.mode == 'online', "mode must be set to 'online'"

        batch_size, T, D, n_feat = feat_m.shape
        assert D == 1, "Online mode expects D=1. That is you should only pass signal from 1 device to each joint node"

        feat_m2j = feat_m.permute(0, 2, 1, 3).reshape(batch_size * D, T, n_feat)
        if h is None:
            h = self.init_hidden_online(batch_size * D, device=feat_m2j.device)

        new_h = []
        for layer_idx, rnn in enumerate(self.rnn_layers):
            gamma_feat, beta_feat = placement_codes[:, layer_idx, 0], placement_codes[:, layer_idx, 1]
            hc_t_layer_idx = h[layer_idx: layer_idx + 1]
            feat_m2j = gamma_feat * feat_m2j + beta_feat
            feat_m2j, hc_t_layer_idx = rnn(feat_m2j, hc_t_layer_idx)
            new_h.append(hc_t_layer_idx)

        feat_m2j = feat_m2j.view(batch_size, D, T, self.n_hidden).permute(0, 2, 1, 3)
        new_h = torch.cat(new_h, dim=0)
        return feat_m2j, new_h


class KinematicsRegressor(nn.Module):
    def __init__(self, n_input, n_out, n_hidden):
        super().__init__()
        self.linear1 = nn.Linear(n_input, n_hidden)
        self.linear2 = nn.Linear(n_hidden, n_out)

    def forward(self, x, merge_device=False):
        batch_size, T, D, n_channel = x.shape
        if not merge_device:
            x = x.permute(0, 2, 1, 3).contiguous().view(batch_size * D, T, -1)
            x = F.relu(self.linear1(x))
            x = self.linear2(x)
            x = x.view(batch_size, D, T, -1).permute(0, 2, 1, 3)
        else:
            x = x.view(batch_size, T, D * n_channel)
            x = F.relu(self.linear1(x))
            x = self.linear2(x)
            x = x.view(batch_size, T, -1)
        return x


class IMUCoCo(nn.Module):
    def __init__(self,
                 coordinate_origins=None,
                 coordinate_max=None,
                 coordinate_min=None,
                 smpl_mesh_coordinates=None,
                 n_hidden=128,
                 n_kr_hidden=32,
                 n_mfe_layers=2,
                 n_jnm_layers=3,
                 n_sce_freq=4,
                 n_sce_emb=40,
                 online_mode=False,
                 joint_node_allocation_map=None,
                 joint_node_max_err_tolerance=-1
                 ):
        super().__init__()

        self.online_mode = online_mode

        self.n_joints = 24
        self.n_jnm_layers = n_jnm_layers
        self.n_hidden = n_hidden

        # modules declaration
        # sensor coordinate encoder x 24
        self.sces = nn.ModuleList([SensorCoordinateEncoder(
            coordinate_origin=coordinate_origins[joint_i],
            coordinate_max=coordinate_max,
            coordinate_min=coordinate_min,
            n_freq=n_sce_freq,
            n_emb=n_sce_emb,
            n_hidden=n_hidden,
            n_layers=n_jnm_layers) for joint_i in range(24)])

        # mesh feature encoders x 24
        self.mfes = nn.ModuleList([MotionFeatureEncoder(
            n_channel=9, n_hidden=n_hidden, n_rnn_layer=n_mfe_layers, online_mode=online_mode) for _ in range(24)])

        # mesh joint translators x 24
        self.jnms = nn.ModuleList([JointNodeModulator(n_hidden=n_hidden, n_hidden_layer=n_jnm_layers, online_mode=online_mode) for _ in range(24)])

        # joint velocity regressors x 24
        self.krvel = nn.ModuleList([KinematicsRegressor(n_input=n_hidden, n_hidden=n_kr_hidden, n_out=3) for _ in range(24)])
        # root velocity regressors x 24
        self.kr_rvel = nn.ModuleList([KinematicsRegressor(n_input=n_hidden, n_hidden=n_kr_hidden, n_out=3) for _ in range(24)])
        # joint position regressors x 24
        self.kr_pos = nn.ModuleList([KinematicsRegressor(n_input=n_hidden, n_hidden=n_kr_hidden, n_out=3) for _ in range(24)])
        # joint local orientation regressors x 24
        self.kr_lori = nn.ModuleList([KinematicsRegressor(n_input=n_hidden, n_hidden=n_kr_hidden, n_out=6) for _ in range(24)])
        # joint global orientation regressors x 24
        self.kr_gori = nn.ModuleList([KinematicsRegressor(n_input=n_hidden, n_hidden=n_kr_hidden, n_out=6) for _ in range(24)])

        # full-body pose regressor x 1
        self.pr = KinematicsRegressor(n_input=n_hidden * 24, n_hidden=n_kr_hidden * 24, n_out=24 * 6)

        # loss weighting coefficients
        self.lw_ph1_kr_joint = 0.5
        self.lw_ph1_pr_joint = 0.5

        self.lw_ph2_kr_mesh = 0.3
        self.lw_ph2_align = 0.35
        self.lw_ph2_pr_mesh = 0.35

        self.lw_kr_rvel = 0.09
        self.lw_kr_vel = 0.25
        self.lw_kr_pos = 0.36
        self.lw_kr_lori = 0.15
        self.lw_kr_gori = 0.15

        # for training
        self.align_loss = nn.CosineEmbeddingLoss()
        self.optim_ph1 = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.optim_ph2 = torch.optim.Adam(self.parameters(), lr=3e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim_ph2, T_max=100)


        # for IMU matching at test time
        current_device_2_joint_mapping = torch.zeros(24, dtype=torch.long, requires_grad=False)
        current_device_2_joint_error = torch.zeros(24, dtype=torch.float, requires_grad=False)
        mesh_positions = smpl_mesh_coordinates[:, 1:]  # drop the category
        mesh_categories = smpl_mesh_coordinates[:, :1]  # drop the category
        current_device_coordinates = torch.zeros((1, 4), dtype=torch.float, requires_grad=False)
        current_joint_node_coordinates = torch.zeros((24, 4), dtype=torch.float, requires_grad=False)

        self.register_buffer('current_device_2_joint_mapping', current_device_2_joint_mapping)
        self.register_buffer('current_device_2_joint_error', current_device_2_joint_error)
        self.register_buffer('mesh_positions', mesh_positions)
        self.register_buffer('mesh_categories', mesh_categories)
        self.register_buffer('coordinate_origins', coordinate_origins)
        self.register_buffer('current_device_coordinates', current_device_coordinates)
        self.register_buffer('current_joint_node_coordinates', current_joint_node_coordinates)

        self.current_device_nearest_mesh_idx = None

        self.joint_node_allocation_map = joint_node_allocation_map  # 'glb_ori_err' or 'distance'
        if self.joint_node_allocation_map is not None:
            mesh_transfer_loss = torch.load(joint_node_allocation_map, map_location=torch.device('cpu'), weights_only=True)
            self.register_buffer("mesh_transfer_loss", mesh_transfer_loss)

        self.joint_node_max_err_tolerance = joint_node_max_err_tolerance
        self.nodes_over_error_thres = []

        # for online implementation
        self.sce_params, self.sce_buffers = None, None
        self.jnm_params, self.jnm_buffers = None, None
        self.mfe_params, self.mfe_buffers = None, None
        self.jnm_base_model = None
        self.mfe_base_model = None
        self.sce_base_model = None
        self.parallel_joint_node_forward = None
        self.parallel_sce_forward = None
        self.placement_codes_buffered = None

    def init_sensor_coordinate_encoders(self):
        for sce in self.sces:
            sce.init_weight_and_bias()

    def prepare_parallel_joint_node_implementation(self):
        def functional_mfe_jnm_node_forward(mfe_params, mfe_buffers, jnm_params, jnm_buffers, imu_signal, placement_code, mfe_hidden, jnm_hidden):
            mfe_feat, mfe_hidden = functional_call(self.mfe_base_model, (mfe_params, mfe_buffers), (imu_signal, mfe_hidden))
            modulated_feat, jnm_hidden = functional_call(self.jnm_base_model, (jnm_params, jnm_buffers), (mfe_feat, placement_code, jnm_hidden))
            return modulated_feat, mfe_hidden, jnm_hidden

        # Use vmap to vectorize across the joint dimension
        parallel_joint_forward_vmap = torch.vmap(functional_mfe_jnm_node_forward, in_dims=(
            0,  # mfe_params dict
            0,  # mfe_buffers dict
            0,  # jnm_params dict
            0,  # jnm_buffers dict
            0,  # imu_signals:
            0,  # placement codes
            0,  # mfe_hidden
            0,  # jnm_hidden
        ))
        mfes_list = [mfe for mfe in self.mfes]
        jnms_list = [jnm for jnm in self.jnms]

        self.mfe_params, self.mfe_buffers = stack_module_state(mfes_list)
        self.jnm_params, self.jnm_buffers = stack_module_state(jnms_list)

        self.mfe_base_model = copy.deepcopy(self.mfes[0])
        self.mfe_base_model = self.mfe_base_model.to('meta')
        self.jnm_base_model = copy.deepcopy(self.jnms[0])
        self.jnm_base_model = self.jnm_base_model.to('meta')

        self.parallel_joint_node_forward = parallel_joint_forward_vmap

    def prepare_parallel_sce_implementation(self):
        def functional_sce_forward(sce_params, sce_buffers, imu_coordinates):
            placement_codes = functional_call(self.sce_base_model, (sce_params, sce_buffers), (imu_coordinates,))
            return placement_codes

        parallel_sce_forward_vmap = torch.vmap(functional_sce_forward, in_dims=(
            0,  # sce_params dict
            0,  # sce_buffers dict
            0,  # imu_coordinates: batching along joints dimension
        ))
        # Convert ModuleLists to regular lists for vmap
        sces_list = [sce for sce in self.sces]
        self.sce_params, self.sce_buffers = stack_module_state(sces_list)

        self.sce_base_model = copy.deepcopy(self.sces[0])
        self.sce_base_model = self.sce_base_model.to('meta')
        # Store the parallelized function
        self.parallel_sce_forward = parallel_sce_forward_vmap

    def _forward_kr(self, joint_idx, input_features):
        """
        :param input_features: (batch_size, T, D, n_inputs)
        :return:
        """
        vel_out = self.krvel[joint_idx](input_features)
        rvel_out = self.kr_rvel[joint_idx](input_features)
        pos_out = self.kr_pos[joint_idx](input_features)
        gori_out = self.kr_gori[joint_idx](input_features)
        lori_out = self.kr_lori[joint_idx](input_features)
        return vel_out, rvel_out, pos_out, gori_out, lori_out

    def pr_loss(self, pose_pred, pose_gt, is_for_mesh=False):
        pose_mse_loss = F.mse_loss(pose_pred, pose_gt)
        return pose_mse_loss

    def kr_loss(self, k_pred, k_gt, tran_mask, joint_idx=None, keep_n_mesh_dim=False):
        vel_pred, rvel_pred, pos_pred, gori_pred, lori_pred = k_pred
        vel_gt, rvel_gt, pos_gt, gori_gt, lori_gt = k_gt

        n_mesh = vel_pred.shape[2]

        rvel_gt = rvel_gt.unsqueeze(2).repeat(1, 1, n_mesh, 1)
        vel_gt = vel_gt.unsqueeze(2).repeat(1, 1, n_mesh, 1)
        pos_gt = pos_gt.unsqueeze(2).repeat(1, 1, n_mesh, 1)
        gori_gt = gori_gt.unsqueeze(2).repeat(1, 1, n_mesh, 1)
        lori_gt = lori_gt.unsqueeze(2).repeat(1, 1, n_mesh, 1)

        if joint_idx == 0:
            if tran_mask is not None:
                if not keep_n_mesh_dim:
                    vel_loss = F.mse_loss(vel_pred[tran_mask], vel_gt[tran_mask])
                else:
                    vel_loss = F.mse_loss(vel_pred[tran_mask], vel_gt[tran_mask], reduction='none')
                    vel_loss = vel_loss.mean(dim=(0, 1, 3))
            else:
                vel_loss = 0
        else:
            if not keep_n_mesh_dim:
                vel_loss = F.mse_loss(vel_pred, vel_gt)
            else:
                vel_loss = F.mse_loss(vel_pred, vel_gt, reduction='none')
                vel_loss = vel_loss.mean(dim=(0, 1, 3))
        if tran_mask is not None:
            rvel_loss = 0
            for n in [1, 3, 9, 27]:
                # Stride across frames for the given interval
                start_indices = torch.arange(0, vel_pred.shape[1], n, device=vel_pred.device)
                end_indices = torch.minimum(start_indices + n, torch.tensor(vel_pred.shape[1], device=vel_pred.device))

                # Collect sequences by slicing the root velocity
                pred_slices = [rvel_pred[tran_mask, start:end] for start, end in zip(start_indices, end_indices)]
                gt_slices = [rvel_gt[tran_mask, start:end] for start, end in zip(start_indices, end_indices)]

                # Concatenate into a single tensor for batched computation
                pred_concat = torch.cat(pred_slices, dim=1)
                gt_concat = torch.cat(gt_slices, dim=1)

                if not keep_n_mesh_dim:
                    loss_n = torch.mean((pred_concat - gt_concat).norm(dim=-1))
                else:
                    loss_n = torch.mean((pred_concat - gt_concat).norm(dim=-1), dim=(0, 1))
                rvel_loss += 0.25 * loss_n
        else:
            rvel_loss = 0

        if not keep_n_mesh_dim:
            pos_loss = F.mse_loss(pos_pred, pos_gt)
            gori_loss = F.mse_loss(gori_pred, gori_gt)
            lori_loss = F.mse_loss(lori_pred, lori_gt)
        else:
            pos_loss = F.mse_loss(pos_pred, pos_gt, reduction='none')
            gori_loss = F.mse_loss(gori_pred, gori_gt, reduction='none')
            lori_loss = F.mse_loss(lori_pred, lori_gt, reduction='none')
            pos_loss = pos_loss.mean(dim=(0, 1, 3))
            gori_loss = gori_loss.mean(dim=(0, 1, 3))
            lori_loss = lori_loss.mean(dim=(0, 1, 3))


        loss = vel_loss * self.lw_kr_vel + \
               rvel_loss * self.lw_kr_rvel + \
               pos_loss * self.lw_kr_pos + \
               gori_loss * self.lw_kr_gori + \
               lori_loss * self.lw_kr_lori

        return loss

    def forward_joint(self, imu_inputs_joints, coordinate_joint, pose_gt, train=False,
                      k_gt=None, tran_mask=None):
        """
        :param tran_mask:
        :param k_gt: vel_gt, pos_gt, gori_gt, lori_gt
        :param intermediate_reg:
        :param train:
        :param imu_inputs_joints: (batch_size, T, n_joints, n_input)
        :param coordinate_joint: (batch_size, n_joints, n_coordinate + 1)
        :param input_lengths: (batch_size,)
        :param pose_gt: (batch_size, T, n_joints, n_input)
        :return: loss
        """
        if train:
            self.train()
        else:
            self.eval()

        batch_size, T, n_joints, n_channel = imu_inputs_joints.shape
        self.optim_ph1.zero_grad()
        jfe_feat = torch.empty(batch_size, T, 24, self.n_hidden, device=imu_inputs_joints.device)  # (batch_size, T, n_joints, n_hidden)
        vel_gt, rvel_gt, pos_gt, gori_gt, lori_gt = k_gt
        total_kr_loss = 0
        for target_j in range(24):
            # feature from joint IMU
            feat_j = self.mfes[target_j](imu_inputs_joints[:, :, target_j:target_j + 1, :])
            q_j = self.sces[target_j](coordinate_joint[:, target_j, :]).unsqueeze(1)
            feat_j = self.jnms[target_j](feat_j, q_j)
            jfe_feat[:, :, target_j:target_j + 1, :] = feat_j

            j_vel_out, j_rvel_out, j_pos_out, j_gori_out, j_lori_out = self._forward_kr(target_j, feat_j)
            j_vel_gt, j_rvel_gt, j_pos_gt, j_gori_gt, j_lori_gt = vel_gt[:, :, target_j], rvel_gt, pos_gt[:, :, target_j], gori_gt[:, :, target_j], lori_gt[:, :, target_j]
            j_kr_loss = self.kr_loss((j_vel_out, j_rvel_out, j_pos_out, j_gori_out, j_lori_out), (j_vel_gt, j_rvel_gt, j_pos_gt, j_gori_gt, j_lori_gt), tran_mask)
            total_kr_loss += j_kr_loss

        pose_out = self.pr(jfe_feat, merge_device=True)
        total_kr_loss = total_kr_loss / 24
        total_pr_loss = self.pr_loss(pose_out, pose_gt.view(pose_gt.shape[0], pose_gt.shape[1], -1))


        loss = self.lw_ph1_pr_joint * total_pr_loss + self.lw_ph1_kr_joint * total_kr_loss
        if train:
            loss.backward()
            self.optim_ph1.step()
        return jfe_feat, loss.item(), total_pr_loss.item(), total_kr_loss.item()

    def forward_mesh(self, imu_inputs_mesh, coordinate_mesh,
                     joint_ref_feats, k_gt, pose_gt,
                     imu_inputs_mesh_masks=None, tran_mask=None,
                     train=False):

        if imu_inputs_mesh_masks is None:
            batch_size, T, D, _ = imu_inputs_mesh.shape
        else:
            batch_size, T, _, _ = imu_inputs_mesh.shape
            D = len(imu_inputs_mesh_masks[0])

        all_mesh2joint_feat = joint_ref_feats.permute(2, 0, 1, 3).unsqueeze(3).expand(24, batch_size, T, D, self.n_hidden).clone()

        cos_loss_target = torch.ones(batch_size * T * D, device=imu_inputs_mesh.device)

        joint_iteration_list = np.random.permutation(self.n_joints)

        total_loss = 0
        loss_by_joints = [0] * 24
        total_mesh_kr_loss = 0
        total_mesh_pose_loss = 0
        total_mesh_align_loss = 0

        vel_gt, rvel_gt, pos_gt, gori_gt, lori_gt = k_gt
        for target_j in joint_iteration_list:
            self.optim_ph2.zero_grad()

            if imu_inputs_mesh_masks is not None:
                imu_inputs_mesh_target_j = imu_inputs_mesh[:, :, imu_inputs_mesh_masks[target_j], :]
                coordinate_mesh_target_j = coordinate_mesh[:, imu_inputs_mesh_masks[target_j], :]
            else:
                imu_inputs_mesh_target_j = imu_inputs_mesh
                coordinate_mesh_target_j = coordinate_mesh

            # feature from mesh IMU
            feat_m = self.mfes[target_j](imu_inputs_mesh_target_j)
            q_m = self.sces[target_j](coordinate_mesh_target_j.view(-1, 4))
            q_m = q_m.view(batch_size, D, q_m.shape[1], q_m.shape[2], q_m.shape[3])
            feat_m2j = self.jnms[target_j](feat_m, q_m)

            # feature from recorded references
            feat_ref = joint_ref_feats[:, :, target_j:target_j + 1, :].expand_as(feat_m2j)
            feat_m2j_flattened = feat_m2j.permute(0, 2, 1, 3).reshape(batch_size * T * D, -1)
            feat_ref_flattened = feat_ref.permute(0, 2, 1, 3).reshape(batch_size * T * D, -1)
        
            # compute alignment
            loss_j_align_mesh = self.align_loss(feat_m2j_flattened, feat_ref_flattened, cos_loss_target)

            # compute kr
            m2j_vel_out, m2j_rvel_out, m2j_pos_out, m2j_gori_out, m2j_lori_out = self._forward_kr(target_j, feat_m2j)
            j_vel_gt, j_rvel_gt, j_pos_gt, j_gori_gt, j_lori_gt = vel_gt[:, :, target_j], rvel_gt, pos_gt[:, :, target_j], gori_gt[:, :, target_j], lori_gt[:, :, target_j]
            loss_j_kr_mesh = self.kr_loss((m2j_vel_out, m2j_rvel_out, m2j_pos_out, m2j_gori_out, m2j_lori_out), (j_vel_gt, j_rvel_gt, j_pos_gt, j_gori_gt, j_lori_gt), tran_mask)
            
            # compute pr
            all_mesh2joint_feat[target_j] = feat_m2j
            pose_gt_mesh = pose_gt.unsqueeze(1).expand(batch_size, D, T, 24, 6).reshape(batch_size * D, T, -1)
            pose_out_mesh = self.pr(all_mesh2joint_feat.permute(1, 3, 2, 0, 4).contiguous().view(batch_size * D, T, 24, -1), merge_device=True)  # (batch_size * D, T, n_output)
            loss_j_pose_mesh = self.pr_loss(pose_out_mesh, pose_gt_mesh)

            loss_j = self.lw_ph2_kr_mesh * loss_j_kr_mesh + self.lw_ph2_pr_mesh * loss_j_pose_mesh + self.lw_ph2_align * loss_j_align_mesh

            if train:
                loss_j.backward()
                self.optim_ph2.step()

            all_mesh2joint_feat = all_mesh2joint_feat.detach()

            total_loss += loss_j.item()
            loss_by_joints[target_j] = self.lw_ph2_kr_mesh * loss_j_kr_mesh.item() + self.lw_ph2_align * loss_j_align_mesh.item()
            total_mesh_align_loss += loss_j_align_mesh.item()
            total_mesh_pose_loss += loss_j_pose_mesh.item()
            total_mesh_kr_loss += loss_j_kr_mesh.item()

        total_mesh_align_loss /= 24
        total_mesh_kr_loss /= 24
        total_mesh_pose_loss /= 24
        total_loss /= 24
        loss_by_joints = [loss_by_joints[i] + self.lw_ph2_pr_mesh * total_mesh_pose_loss for i in range(24)]

        return total_loss, loss_by_joints, total_mesh_pose_loss, total_mesh_kr_loss, total_mesh_align_loss

    """
    Inference adaptive functions
    """

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
    def freeze_krpr(self):
        for param in self.krvel.parameters():
            param.requires_grad = False
        for param in self.kr_rvel.parameters():
            param.requires_grad = False
        for param in self.kr_pos.parameters():
            param.requires_grad = False
        for param in self.kr_gori.parameters():
            param.requires_grad = False
        for param in self.kr_lori.parameters():
            param.requires_grad = False
        for param in self.pr.parameters():
            param.requires_grad = False
    def freeze_sce(self):
        for param in self.sces.parameters():
            param.requires_grad = False
    def freeze_mfe(self):
        for param in self.mfes.parameters():
            param.requires_grad = False
    def unfreeze_mfe(self):
        for param in self.mfes.parameters():
            param.requires_grad = True
    def unfreeze_sce(self):
        for param in self.sces.parameters():
            param.requires_grad = True

    def set_current_device_coordinates(self, updated_device_coordinates):
        if len(self.current_device_coordinates) != len(updated_device_coordinates):
            # we are using a different number of devices, so we need to reset the current device coordinates to have a different shape
            self.current_device_coordinates = torch.zeros((len(updated_device_coordinates), 4), dtype=torch.float, requires_grad=False)
        if updated_device_coordinates.shape[1] == 4:    # if the provided coordinates are already have the category dimension
            self.current_device_coordinates = updated_device_coordinates
        else:   # if the provided coordinates do not have the category dimension, assign the numerical parts only
            self.current_device_coordinates[:, 1:] = updated_device_coordinates
        self.__update_min_loss_device_mapping()

    def buffer_placement_codes_with_current_devices(self):
        self.placement_codes_buffered = self.parallel_sce_forward(self.sce_params, self.sce_buffers, self.current_joint_node_coordinates.unsqueeze(1))[:, 0]

    def __update_min_loss_device_mapping(self):
        self.nodes_over_error_thres = []
        # find category by nearest mesh vertex
        nearest_mesh_vid = self.__get_nearest_mesh_idx(self.current_device_coordinates)
        mesh_vid_category = self.mesh_categories[nearest_mesh_vid]
        for joint_idx in range(24):
            loss_current_devices = self.mesh_transfer_loss[joint_idx, nearest_mesh_vid]
            best_device = torch.argmin(loss_current_devices)
            best_loss = loss_current_devices[best_device]
            self.current_device_coordinates[best_device][0] = mesh_vid_category[best_device].squeeze() # save the region category of the device
            self.current_device_2_joint_mapping[joint_idx] = best_device
            self.current_device_2_joint_error[joint_idx] = best_loss
            self.current_joint_node_coordinates[joint_idx] = self.current_device_coordinates[best_device]

            if self.joint_node_max_err_tolerance > 0 and best_loss > self.joint_node_max_err_tolerance:   # if the error is too large, set the node to be over error threshold
                self.nodes_over_error_thres.append(joint_idx)

    def __get_nearest_mesh_idx(self, device_coordinates):
        if device_coordinates.shape[-1] == 4:
            device_coordinates = device_coordinates[:, 1:]  # drop the category
        distance_mesh = torch.norm(device_coordinates.unsqueeze(1) - self.mesh_positions.unsqueeze(0), dim=2)
        nearest_mesh_idx = torch.argmin(distance_mesh, dim=1)
        return nearest_mesh_idx

    def inference_time_forward_mesh(self, imu_signals):
        batch_size, T, _, _ = imu_signals.shape
        encoded_signals = torch.zeros(batch_size, T, self.n_joints, self.n_hidden, device=imu_signals.device)
        for target_j in range(24):
            best_device_target_j = self.current_device_2_joint_mapping[target_j]
            buffered_placement_code_target_j = self.placement_codes_buffered[target_j].unsqueeze(0).repeat(batch_size, 1, 1, 1)
            buffered_placement_code_target_j = buffered_placement_code_target_j.unsqueeze(1)
            imu_signals_target_j = imu_signals[:, :, best_device_target_j, :].unsqueeze(2)
            feat_m = self.mfes[target_j](imu_signals_target_j)
            feat_m2j = self.jnms[target_j](feat_m, buffered_placement_code_target_j)
            encoded_signals[:, :, target_j, :] = feat_m2j.squeeze(2)

        encoded_signals[:, :, self.nodes_over_error_thres] = 0   # set the features of the nodes over error threshold to 0
        return encoded_signals


    def inference_time_forward_mesh_online(self, imu_signals, hidden_mfe=None, hidden_jnm=None):
        batch_size, T, _, _ = imu_signals.shape
        assert batch_size == 1, "only support batch size 1 for online"
        signal_lens = torch.tensor([T] * batch_size)
        remapped_imu_signal = imu_signals[:, :, self.current_device_2_joint_mapping, :].permute(2, 0, 1, 3).unsqueeze(3)
        batch_encoded_signals = []
        batch_mfe_hidden = []
        batch_jnm_hidden = []

        if hidden_mfe is None:
            hidden_mfe = self.mfes[0].init_hidden_online(batch_size, device=imu_signals.device).unsqueeze(0).repeat(self.n_joints, 1, 1, 1, 1)
        if hidden_jnm is None:
            hidden_jnm = self.jnms[0].init_hidden_online(batch_size, device=imu_signals.device).unsqueeze(0).repeat(self.n_joints, 1, 1, 1, 1)

        placement_codes = self.placement_codes_buffered.unsqueeze(1)
        for i in range(batch_size):
            per_batch_encoded, mfe_hidden, jnm_hidden = self.parallel_joint_node_forward(
                self.mfe_params,
                self.mfe_buffers,
                self.jnm_params,
                self.jnm_buffers,
                remapped_imu_signal[:, i:i + 1],
                signal_lens[i:i + 1],
                placement_codes[:, i:i+1],
                hidden_mfe,
                hidden_jnm
            )
            batch_encoded_signals.append(per_batch_encoded)
            batch_mfe_hidden.append(mfe_hidden)
            batch_jnm_hidden.append(jnm_hidden)

        encoded_signals = torch.cat(batch_encoded_signals, dim=1).squeeze(3).permute(1, 2, 0, 3)
        hidden_mfe = torch.cat(batch_mfe_hidden, dim=3)
        hidden_jnm = torch.cat(batch_jnm_hidden, dim=3)
        haencoded_signals[:, :, self.nodes_over_error_thres] = 0
        return encoded_signals, hidden_mfe, hidden_jnm

    def load_offline_state_dict_to_online_model(self, offline_state_dict):
        new_state_dict = {}
        for k, v in offline_state_dict.items():
            if k.startswith("mfes.") and "rnn.weight_ih_l" in k:
                joint_node_num = int(k.split(".")[1])
                layer_num = int(k.split("weight_ih_l")[-1])
                mfe_linear_weight_key = f"mfes.{joint_node_num}.rnn.lstm_cells.{layer_num}.linear_i.weight"
                ih_weight = offline_state_dict[k]
                hh_weight = offline_state_dict[k.replace("weight_ih_l", "weight_hh_l")]
                combined_weight = torch.cat([ih_weight, hh_weight], dim=1)
                mfe_linear_weight_value = combined_weight
                new_state_dict[mfe_linear_weight_key] = mfe_linear_weight_value
            elif k.startswith("mfes.") and "rnn.weight_hh_l" in k:
                # we already processed hh weight when processing the ih weight
                continue

            elif k.startswith("mfes.") and "rnn.bias_ih_l" in k:
                joint_node_num = int(k.split(".")[1])
                layer_num = int(k.split("bias_ih_l")[-1])
                mfe_linear_bias_key = f"mfes.{joint_node_num}.rnn.lstm_cells.{layer_num}.linear_i.bias"
                ih_bias = offline_state_dict[k]
                hh_bias = offline_state_dict[k.replace("bias_ih_l", "bias_hh_l")]
                combined_weight = ih_bias + hh_bias
                mfe_linear_weight_value = combined_weight
                new_state_dict[mfe_linear_bias_key] = mfe_linear_weight_value
            elif k.startswith("mfes.") and "rnn.bias_hh_l" in k:
                # we already processed hh bias when processing the ih bias
                continue

            elif k.startswith("jnms.") and "rnn_layers" in k and "weight_ih_l" in k:
                joint_node_num = int(k.split(".")[1])
                layer_num = int(k.split(".rnn_layers.")[1][0])
                jnm_linear_weight_key = f"jnms.{joint_node_num}.rnn_layers.{layer_num}.lstm_cells.0.linear_i.weight"
                ih_weight = offline_state_dict[k]
                hh_weight = offline_state_dict[k.replace("weight_ih_l", "weight_hh_l")]
                combined_weight = torch.cat([ih_weight, hh_weight], dim=1)
                mfe_linear_weight_value = combined_weight
                new_state_dict[jnm_linear_weight_key] = mfe_linear_weight_value
            elif k.startswith("jnms.") and "rnn_layers" in k and "weight_hh_l" in k:
                # we already processed hh weight when processing the ih weight
                continue

            elif k.startswith("jnms.") and "rnn_layers" in k and "bias_ih_l" in k:
                joint_node_num = int(k.split(".")[1])
                layer_num = int(k.split(".rnn_layers.")[1][0])
                jnm_linear_bias_key = f"jnms.{joint_node_num}.rnn_layers.{layer_num}.lstm_cells.0.linear_i.bias"
                ih_bias = offline_state_dict[k]
                hh_bias = offline_state_dict[k.replace("bias_ih_l", "bias_hh_l")]
                combined_weight = ih_bias + hh_bias
                mfe_linear_weight_value = combined_weight
                new_state_dict[jnm_linear_bias_key] = mfe_linear_weight_value
            elif k.startswith("jnms.") and "rnn_layers" in k and "bias_hh_l" in k:
                # we already processed hh bias when processing the ih bias
                continue
            else:
                new_state_dict[k] = v
        self.load_state_dict(new_state_dict)
