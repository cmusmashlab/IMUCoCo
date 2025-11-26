import torch
from collections import defaultdict
from articulate.evaluator import BasePoseEvaluator
from articulate.math.angular import radian_to_degree, angle_between, RotationRepresentation, r6d_to_rotation_matrix

def compute_mean_results_by_placement(case_all_results):
    # accumulate the values for each key
    temp_results = defaultdict(list)
    for results in case_all_results:
        for key, values in results.items():
            if key != 'test_sample_name' and key != 'input_names' and key != 'activity':
                temp_results[key].append(values)
    mean_results = {key: sum(values for values in value_list) / len(value_list) for key, value_list in temp_results.items()}
    return mean_results


class HPEEvaluator(BasePoseEvaluator):
    r"""
    Evaluator for full motions (pose sequences with global translations). Plenty of metrics.
    """

    def __init__(self, official_model_file: str, align_joint=None,
                 rep=RotationRepresentation.ROTATION_MATRIX,
                 use_pose_blendshape=False, fps=60,
                 device=torch.device('cpu')):
        r"""
        Init a full motion evaluator.

        :param official_model_file: Path to the official SMPL/MANO/SMPLH model to be loaded.
        :param align_joint: Which joint to align. (e.g. SMPLJoint.ROOT). By default the root.
        :param rep: The rotation representation used in the input poses.
        :param use_pose_blendshape: Whether to use pose blendshape or not.
        :param joint_mask: If not None, local angle error, global angle error, and joint position error
                           for these joints will be calculated additionally.
        :param fps: Motion fps, by default 60.
        :param device: torch.device, cpu or cuda.
        """
        super(HPEEvaluator, self).__init__(official_model_file, rep, use_pose_blendshape, device=device)
        self.align_joint = 0 if align_joint is None else align_joint.value
        self.fps = fps

        self.joint_ignore = [0, 10, 11, 20, 21, 22, 23]   # ignored root, wrist, finger, toe (see DiffusionPoser)
        self.joint_include = [x for x in range(24) if x not in self.joint_ignore]


    @torch.no_grad()
    def __call__(self, pose_p, pose_t, shape_p=None, shape_t=None, tran_p=None, tran_t=None,
                 is_pose_p_r6d=False, is_pose_t_r6d=False,
                 is_predicted_pose_local=True, instrumented_arm=None,
                 instrumented_leg=None, return_std=False):
        r"""
        Get the measured errors. The returned tensor in shape [10, 2] contains mean and std of:
          0.  Joint position error (align_joint position aligned).
          1.  Vertex position error (align_joint position aligned).
          2.  Joint local angle error (in degrees).
          3.  Joint global angle error (in degrees).
          4.  Predicted motion jerk (with global translation).
          5.  True motion jerk (with global translation).
          6.  Translation error (mean root translation error per second, using a time window size of 1s).
          7.  Masked joint position error (align_joint position aligned, zero if mask is None).
          8.  Masked joint local angle error. (in degrees, zero if mask is None).
          9.  Masked joint global angle error. (in degrees, zero if mask is None).

        :param pose_p: Predicted pose or the first pose in shape [batch_size, *] that can
                       reshape to [batch_size, num_joint, rep_dim].
        :param pose_t: True pose or the second pose in shape [batch_size, *] that can
                       reshape to [batch_size, num_joint, rep_dim].
        :param shape_p: Predicted shape that can expand to [batch_size, 10]. Use None for the mean(zero) shape.
        :param shape_t: True shape that can expand to [batch_size, 10]. Use None for the mean(zero) shape.
        :param tran_p: Predicted translations in shape [batch_size, 3]. Use None for zeros.
        :param tran_t: True translations in shape [batch_size, 3]. Use None for zeros.

        :return: Tensor in shape [10, 2] for the mean and std of all errors.
        """

        f = self.fps

        if is_pose_p_r6d:
            batch_size, T, n_joints, _ = pose_p.shape
            pose_p = r6d_to_rotation_matrix(pose_p.reshape(-1, 6)).view(-1, 24, 3, 3)
        if is_pose_t_r6d:
            pose_t = r6d_to_rotation_matrix(pose_t.reshape(-1, 6)).view(-1, 24, 3, 3)

        pose_global_t, joint_t, vertex_t = self.model.forward_kinematics(pose_t, shape_t, tran=tran_t, calc_mesh=True)

        if is_predicted_pose_local:    # if prediction is local, we align the root, and then fk to also get global.
            pose_p[:, self.align_joint] = pose_t[:, self.align_joint]
            pose_global_p, joint_p, vertex_p = self.model.forward_kinematics(pose_p, shape_p, tran=tran_p, calc_mesh=True)
        else:
            # if the prediction is global, we align the root, and ik to get local.
            # note align must happen before IK; otherwise aligning at local after IK, the joints other than root may be incorrectly moved away from predicted.
            pose_global_p = pose_p
            pose_global_p[:, self.align_joint] = pose_global_t[:, self.align_joint]
            pose_p = self.model.inverse_kinematics_R(pose_global_p.view(-1, 24, 3, 3)).view(-1, 24, 3, 3)  # get the local pose)
            _, joint_p, vertex_p = self.model.forward_kinematics(pose_p, shape_p, tran=tran_p, calc_mesh=True)

        pose_p_full = pose_p.clone()
        offset_from_p_to_t = (joint_t[:, self.align_joint] - joint_p[:, self.align_joint]).unsqueeze(1)
        
        # vertex position error (ignore end joints)
        ve = (vertex_p + offset_from_p_to_t - vertex_t).norm(dim=2)  # N, J
        
        # joint position error (ignore end joints)
        je = (joint_p + offset_from_p_to_t - joint_t).norm(dim=2)  # N, J
        je = je[:, self.joint_include]

        # local angular error
        lae = radian_to_degree(angle_between(pose_p, pose_t).view(pose_p.shape[0], -1))  # N, J
        lae = lae[:, self.joint_include]

        # global angular error
        gae = radian_to_degree(angle_between(pose_global_p, pose_global_t).view(pose_p.shape[0], -1))  # N, J
        gae = gae[:, self.joint_include]

        # jitter of prediction (km/s3)
        jkp = ((joint_p[3:] - 3 * joint_p[2:-1] + 3 * joint_p[1:-2] - joint_p[:-3]) * (f ** 3)).norm(dim=2) * 0.001  # N, J

        # jitter true (km/s3)
        jkt = ((joint_t[3:] - 3 * joint_t[2:-1] + 3 * joint_t[1:-2] - joint_t[:-3]) * (f ** 3)).norm(dim=2) * 0.001  # N, J

        # translation error (1s window)
        te_1 = ((joint_p[f:, :1] - joint_p[:-f, :1]) - (joint_t[f:, :1] - joint_t[:-f, :1])).norm(dim=2)  # N, 1
        # translation error (2s window)
        te_2 = ((joint_p[2 * f:, :1] - joint_p[:-2 * f, :1]) - (joint_t[2 * f:, :1] - joint_t[:-2 * f, :1])).norm(dim=2)  # N, 1
        # translation error (5s window)
        te_5 = ((joint_p[5 * f:, :1] - joint_p[:-5 * f, :1]) - (joint_t[5 * f:, :1] - joint_t[:-5 * f, :1])).norm(dim=2)  # N, 1
        # translation error (10s window)
        te_10 = ((joint_p[10 * f:, :1] - joint_p[:-10 * f, :1]) - (joint_t[10 * f:, :1] - joint_t[:-10 * f, :1])).norm(dim=2)  # N, 1

        results = {
            'GlobalAngularError': gae.mean().item(),
            'LocalAngularError': lae.mean().item(),
            'JointPositionError': je.mean().item(),
            'VertexPositionError': ve.mean().item(),
            'JitterPred': jkp.mean().item(),
            'JitterTrue': jkt.mean().item(),
            'TranslationError1s': te_1.mean().item(),
            'TranslationError2s': te_2.mean().item(),
            'TranslationError5s': te_5.mean().item(),
            'TranslationError10s': te_10.mean().item(),
        }
        return results
