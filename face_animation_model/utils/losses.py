import torch
from face_animation_model.utils.compute_lip_distributions import Compute_lip_metric
from face_animation_model.utils.ringnet_lip_static_embedding import RingNet_lip_embedding
import os
from face_animation_model.utils.dtw  import dtw
from numpy.linalg import norm
from face_animation_model.utils.mbp_loss import MBP_reconstruction_loss


def get_loss(type):
    if type == "mse":
        loss_ = torch.nn.MSELoss()
    elif type == "l1":
        loss_ = torch.nn.L1Loss()
    elif type == "cosine":
        loss_ = torch.nn.CosineSimilarity()
    else:
        raise("Enter valid reason")

    return loss_

class Custom_errors():
    """
    Input to the loss must be in mm
    """
    def __init__(self, vertice_dim, 
                        loss_creterion,
                        loss_dict={}):
        self.vertice_dim = vertice_dim
        self.loss = loss_creterion
        self.loss_dict = loss_dict

        # self lip metric
        self.lip_metric = Compute_lip_metric(vertice_dim, None, loss_dict)
        self.lip_static_embedding = RingNet_lip_embedding()
        self.mse_loss = torch.nn.MSELoss(reduction='none')
        self.l1_loss = torch.nn.L1Loss()
        self.mse_loss_with_reduction = torch.nn.MSELoss()
        self.ce_loss = torch.nn.CrossEntropyLoss()

        ### mbp weight reg loss type
        mbp_weight_reg_loss_type = self.loss_dict.get('mbp_weight_reg_loss_type', "l1")
        if mbp_weight_reg_loss_type == "l1":
            self.mbp_reg_loss_type = self.l1_loss
        else:
            self.mbp_reg_loss_type = self.mse_loss_with_reduction

        ### mbp_reconstruction loss
        if self.loss_dict.get('mbp_reconstruction_loss', None) is not None:
            self.mbp_reconstruction_loss = MBP_reconstruction_loss(vertice_dim, loss_dict)
        else:
            self.mbp_reconstruction_loss = None

        ### head
        self.head_reg_loss = get_loss(self.loss_dict.get('head_reg_type', "l1"))

    def compute_head_reg_loss(self, predict, real):
        """
        predict: B x Nf x nc
        real: B x Nf x nc
        """
        assert predict.shape == real.shape
        head_reg_weight = self.loss_dict.get('head_reg_weight', 0.0)
        if head_reg_weight > 0:
            hr_loss = self.head_reg_loss(predict, real)
            return hr_loss, head_reg_weight * hr_loss
        else:
            return (torch.tensor(0, dtype=predict.dtype, device=predict.device),
                    torch.tensor(0, dtype=predict.dtype, device=predict.device))

    @torch.no_grad()
    def error_in_mm(self, pred_verts, gt_verts):
        pred_verts_mm = pred_verts.view(-1, self.vertice_dim//3, 3) * 1000.0
        gt_verts_mm = gt_verts.view(-1, self.vertice_dim//3, 3) * 1000.0
        diff_in_mm = pred_verts_mm - gt_verts_mm
        diff_sum = torch.sum(diff_in_mm ** 2, dim=-1)  # Nf x 5023
        dist_in_mm = torch.sqrt(diff_sum)  # Nf x 5023 ### per frame x per vertices distance
        # dist_in_mm = torch.sqrt(torch.sum(diff_in_mm ** 2, dim=-1))
        return torch.mean(dist_in_mm)

    @torch.no_grad()
    def compute_masked_error_in_mm(self, pred_verts, gt_verts, mask):
        return self.error_in_mm(pred_verts * mask, gt_verts * mask)

    @torch.no_grad()
    def compute_max_diff_in_mm(self, pred_verts, gt_verts):
        """
        pred_verts : B x Nf x verice_dim
        """
        pred_verts_mm = pred_verts.view(-1, self.vertice_dim//3, 3) * 1000.0
        gt_verts_mm = gt_verts.view(-1, self.vertice_dim//3, 3) * 1000.0
        diff_in_mm = pred_verts_mm - gt_verts_mm  # Nf x 5023 x 3
        diff_sum = torch.sum(diff_in_mm ** 2, dim=-1)  # Nf x 5023
        dist_in_mm = torch.sqrt(diff_sum)  # Nf x 5023 ### per frame x per vertices distance
        # return torch.max(dist_in_mm)
        per_frame_max = torch.max(dist_in_mm, -1)[0]  # Nf x 1 # maximal vertex displacement per frame
        return torch.mean(per_frame_max)  # mean of per-frame maximal vertex displacement

    @torch.no_grad()
    def compute_masked_max_diff_in_mm(self, pred_verts, gt_verts, mask):
        return self.compute_max_diff_in_mm(pred_verts * mask, gt_verts * mask)

    def velocity_loss(self, predict, real, loss_fn=None):
        """
        predict: B x Nf x vertice_dim
        """
        velocity_weight = self.loss_dict.get('velocity_weight', 0.0)
        forward_velocity_weight = self.loss_dict.get('forward_velocity_weight', 0.0)

        if loss_fn is None:
            loss_fn = self.loss

        if velocity_weight > 0:
            velocity_pred = predict[:, 1:, :] - predict[:, :-1, :]
            velocity_real = real[:, 1:, :] - real[:, :-1, :]
            velocity_loss = loss_fn(velocity_pred, velocity_real)
            return velocity_loss, velocity_weight * velocity_loss
        elif forward_velocity_weight > 0:
            velocity_pred = predict[:, 1:, :] - predict[:, :-1, :].detach()
            velocity_real = real[:, 1:, :] - real[:, :-1, :]
            velocity_loss = loss_fn(velocity_pred, velocity_real)
            return velocity_loss, forward_velocity_weight * velocity_loss
        else:
            return torch.tensor(0, dtype=predict.dtype, device=predict.device), torch.tensor(0, dtype=predict.dtype, device=predict.device)


    def velocity_reg_loss(self, predict, real):
        """
        predict: B x Nf x vertice_dim
        """
        velocity_weight = self.loss_dict.get('velocity_weight', 0.0)
        forward_velocity_weight = self.loss_dict.get('forward_velocity_weight', 0.0)
        if velocity_weight > 0:
            velocity_pred = predict[:, 1:, :] - predict[:, :-1, :]
            velocity_real = real[:, 1:, :] - real[:, :-1, :]
            velocity_loss = self.loss(velocity_pred, velocity_real)
            return velocity_loss, velocity_weight * velocity_loss
        elif forward_velocity_weight > 0:
            velocity_pred = predict[:, 1:, :] - predict[:, :-1, :].detach()
            velocity_real = real[:, 1:, :] - real[:, :-1, :]
            velocity_loss = self.loss(velocity_pred, velocity_real)
            return velocity_loss, forward_velocity_weight * velocity_loss
        else:
            return torch.tensor(0, dtype=predict.dtype, device=predict.device), torch.tensor(0, dtype=predict.dtype, device=predict.device)


    def compute_ce_loss(self, predict, real):
        """
        predict: B x Nf x nc
        real: B x Nf x nc
        """
        # import pdb; pdb.set_trace()
        assert predict.shape == real.shape

        ce_weight = self.loss_dict.get('ce_weight', 0.0)
        if ce_weight > 0:
            ce_loss = self.ce_loss(predict, real)
            return ce_loss, ce_weight * ce_loss
        else:
            return torch.tensor(0, dtype=predict.dtype, device=predict.device), torch.tensor(0, dtype=predict.dtype, device=predict.device)


    def viseme_velocity_loss(self, predict_visemes, real):
            """
            predict: B x Nf x vertice_dim
            """
            viseme_velocity_weight = self.loss_dict.get('viseme_velocity_weight', 0.0)
            if viseme_velocity_weight > 0:
                velocity_pred = torch.mean(predict_visemes[:, 1:, :] - predict_visemes[:, :-1, :], dim=-1)  # 1 X T x 64 -> 1 x T x 1
                velocity_real = torch.mean(real[:, 1:, :] - real[:, :-1, :], dim=-1)  # 1 X T x 15069 -> 1 x T x 1

                velocity_loss = self.loss(velocity_pred, velocity_real)
                return velocity_loss, viseme_velocity_weight * velocity_loss
            else:
                return torch.tensor(0, dtype=predict_visemes.dtype, device=predict_visemes.device), torch.tensor(0, dtype=predict_visemes.dtype, device=predict_visemes.device)

    def accelaration_loss(self, predict, real):
        acc_weight = self.loss_dict.get('acceleration_weight', 0.0)
        if acc_weight > 0:
            x1_pred = predict[:, -1, :]
            x2_pred = predict[:, -2, :]
            x3_pred = predict[:, -3, :]
            acc_pred = x1_pred - 2 * x2_pred + x3_pred

            x1_real = real[:, -1, :]
            x2_real = real[:, -2, :]
            x3_real = real[:, -3, :]
            acc_real = x1_real - 2 * x2_real + x3_real
            acc_loss = self.loss(acc_pred, acc_real)
            return acc_loss, acc_weight * acc_loss
        else:
            return torch.tensor(0, dtype=predict.dtype, device=predict.device), torch.tensor(0, dtype=predict.dtype, device=predict.device)

    def VertsRegLoss(self, expression_offset):
        verts_regularizer_weight = self.loss_dict.get('verts_regularizer_weight', 0.0)
        if verts_regularizer_weight > 0.0:
            verts_reg_loss = verts_regularizer_weight * torch.mean(torch.sum(torch.abs(expression_offset), dim=2))
            return verts_reg_loss, verts_regularizer_weight * verts_reg_loss
        else:
            return torch.tensor(0, dtype=expression_offset.dtype, device=expression_offset.device), torch.tensor(0, dtype=expression_offset.dtype, device=expression_offset.device)


    def l1_rec_loss(self, predict, real):
        """
        predict: B x Nf x vertice_dim
        """
        l1_loss_weight = self.loss_dict.get('l1_loss_weight', 0.0)
        if l1_loss_weight > 0:
            loss = self.l1_loss(predict, real)
            return loss, loss * l1_loss_weight
        else:
            return torch.tensor(0, dtype=predict.dtype, device=predict.device), torch.tensor(0, dtype=predict.dtype, device=predict.device)


    def masked_rec_loss(self, predict, real, mask):
        """
        predict: B x Nf x vertice_dim
        """
        masked_rec_weight = self.loss_dict.get('masked_rec_weight', 0.0)
        if masked_rec_weight > 0:
            masked_pred = mask * predict
            masked_real = mask * real
            masked_loss = self.loss(masked_pred, masked_real)
            return masked_loss, masked_loss * masked_rec_weight
        else:
            return torch.tensor(0, dtype=predict.dtype, device=predict.device), torch.tensor(0, dtype=predict.dtype, device=predict.device)


    def velocity_regularizer_loss(self, predict):
        """
        predict: B x Nf x vertice_dim
        """
        velocity_reg_weight = self.loss_dict.get('velocity_reg_weight', 0.0)
        if velocity_reg_weight > 0:
            velocity_pred = predict[:, 1:, :] - predict[:, :-1, :]
            velocity_reg_loss = torch.norm(velocity_pred, p=2)
            return velocity_reg_loss, velocity_reg_weight * velocity_reg_loss
        else:
            return torch.tensor(0, dtype=predict.dtype, device=predict.device), torch.tensor(0, dtype=predict.dtype, device=predict.device)

    def accl_regularizer_loss(self, predict):
        """
        predict: B x Nf x vertice_dim
        """
        accl_reg_weight = self.loss_dict.get('accl_reg_weight', 0.0)
        if accl_reg_weight > 0:
            x1_pred = predict[:, -1, :]
            x2_pred = predict[:, -2, :]
            x3_pred = predict[:, -3, :]
            accl_pred = x1_pred - 2 * x2_pred + x3_pred
            accl_reg_loss = torch.norm(accl_pred, p=2)
            return accl_reg_loss, accl_reg_weight * accl_reg_loss
        else:
            return torch.tensor(0, dtype=predict.dtype, device=predict.device), torch.tensor(0, dtype=predict.dtype, device=predict.device)

    def lip_distance_loss(self, predict, real):

        lip_distance_loss_weight = self.loss_dict.get('lip_distance_loss_weight', 0.0)
        lip_distance_loss_threshold = self.loss_dict.get('lip_distance_loss_threshold', 1000.0)
        if lip_distance_loss_weight > 0.0:
            gt_distance, pred_distance = self.lip_metric.lip_differences_using_triangle(predict, real, in_mm=False)
            gt_distance_in_mm, _ = self.lip_metric.lip_differences_using_triangle(predict, real, in_mm=True)
            per_frame_weights = (gt_distance_in_mm < lip_distance_loss_threshold) * 1.0
            # import pdb; pdb.set_trace()
            lip_distance_loss = self.loss(pred_distance * per_frame_weights, gt_distance * per_frame_weights)
            return lip_distance_loss, lip_distance_loss_weight * lip_distance_loss
        else:
            return torch.tensor(0, dtype=predict.dtype, device=predict.device), torch.tensor(0, dtype=predict.dtype, device=predict.device)

    def lip_reconstruction_loss(self, predict, real, mask):
        lip_reconstruction_loss_weight = self.loss_dict.get('lip_reconstruction_loss_weight', 0.0)
        if lip_reconstruction_loss_weight > 0.0:
            lip_reconstruction_loss = self.loss(predict * mask,
                                                real * mask)
            return lip_reconstruction_loss, lip_reconstruction_loss_weight * lip_reconstruction_loss
        else:
            return torch.tensor(0, dtype=predict.dtype, device=predict.device), torch.tensor(0, dtype=predict.dtype, device=predict.device)

    def lip_static_embedding_distance_loss(self, predict, real):

        lip_static_embedding_distance_loss_weight = self.loss_dict.get('lip_static_embedding_distance_loss_weight', 0.0)
        lip_static_embedding_distance_loss_threshold = self.loss_dict.get('lip_static_embedding_distance_loss_threshold', 1000.0)
        if lip_static_embedding_distance_loss_weight > 0.0:

            pred_upper_lip, pred_lower_lip = self.lip_static_embedding.extract_upper_and_lower_lip(predict)
            pred_lip_distance = self.lip_metric.get_distance(pred_upper_lip-pred_lower_lip)
            pred_lip_distance = torch.mean(pred_lip_distance, dim=-1)

            gt_upper_lip, gt_lower_lip = self.lip_static_embedding.extract_upper_and_lower_lip(real)
            gt_lip_distance = self.lip_metric.get_distance(gt_upper_lip - gt_lower_lip)
            gt_lip_distance = torch.mean(gt_lip_distance, dim=-1)

            gt_distance_in_mm, _ = self.lip_metric.lip_differences_using_triangle(predict, real, in_mm=True)
            per_frame_weights = (gt_distance_in_mm < lip_static_embedding_distance_loss_threshold) * 1.0

            lip_static_embedding_distance_loss = self.loss(pred_lip_distance * per_frame_weights, gt_lip_distance * per_frame_weights)
            return lip_static_embedding_distance_loss, lip_static_embedding_distance_loss_weight * lip_static_embedding_distance_loss
        else:
            return torch.tensor(0, dtype=predict.dtype, device=predict.device), torch.tensor(0, dtype=predict.dtype, device=predict.device)

    def biased_full_reconstruction_loss(self, predict, real):

        biased_full_reconstruction_loss_dict = self.loss_dict.get('biased_full_reconstruction_loss', None)
        if biased_full_reconstruction_loss_dict is not None:
            lip_closure_threshold = biased_full_reconstruction_loss_dict.get('lip_closure_threshold', 1000)
            face_keypoints = self.lip_static_embedding.extract_lip_keypoints(real.view(-1, self.vertice_dim//3,3) * 1000.0)

            upper_lip_midpoint = face_keypoints[:, 45]  # this is fixed for the ring embedding
            lower_lip_midpoint = face_keypoints[:, 49]  # this is fixed for the ring embedding
            diff = upper_lip_midpoint - lower_lip_midpoint
            gt_distance_in_mm = torch.sqrt(torch.sum(diff ** 2, dim=-1))

            # closed frames are set the closed frame weights
            per_frame_weights = (gt_distance_in_mm < lip_closure_threshold) * 1.0
            per_frame_weights = per_frame_weights * biased_full_reconstruction_loss_dict.get('closed_lip_frames_weight')
            per_frame_weights[per_frame_weights < 0.1] = biased_full_reconstruction_loss_dict.get('other_frames_weight')

            biased_full_reconstruction_loss = self.mse_loss(predict, real)
            per_frame_weights = per_frame_weights.view(1, -1, 1)
            biased_full_reconstruction_loss = torch.mean(per_frame_weights * biased_full_reconstruction_loss)
            return biased_full_reconstruction_loss, biased_full_reconstruction_loss
        else:
            return torch.tensor(0, dtype=predict.dtype, device=predict.device), torch.tensor(0, dtype=predict.dtype, device=predict.device)

    @torch.no_grad()
    def lip_distance_metric(self, predict, real):

        real_face_keypoints = self.lip_static_embedding.extract_lip_keypoints(real.view(-1, self.vertice_dim//3,3) * 1000.0)

        upper_lip_midpoint = real_face_keypoints[:, 45]  # this is fixed for the ring embedding
        lower_lip_midpoint = real_face_keypoints[:, 49]  # this is fixed for the ring embedding
        diff = upper_lip_midpoint - lower_lip_midpoint
        gt_distance_in_mm = torch.sqrt(torch.sum(diff ** 2, dim=-1))

        predict_face_keypoints = self.lip_static_embedding.extract_lip_keypoints(predict.view(-1, self.vertice_dim//3,3) * 1000.0)
        upper_lip_midpoint = predict_face_keypoints[:, 45]  # this is fixed for the ring embedding
        lower_lip_midpoint = predict_face_keypoints[:, 49]  # this is fixed for the ring embedding
        diff = upper_lip_midpoint - lower_lip_midpoint
        pred_distance_in_mm = torch.sqrt(torch.sum(diff ** 2, dim=-1))

        dist_diff = gt_distance_in_mm - pred_distance_in_mm
        return torch.mean(dist_diff)

    @torch.no_grad()
    def compute_dtw_w_lip_dist(self, predict, real):

        real_face_keypoints = self.lip_static_embedding.extract_lip_keypoints(real.view(-1, self.vertice_dim//3,3) * 1000.0)

        upper_lip_midpoint = real_face_keypoints[:, 45]  # this is fixed for the ring embedding
        lower_lip_midpoint = real_face_keypoints[:, 49]  # this is fixed for the ring embedding
        diff = upper_lip_midpoint - lower_lip_midpoint
        gt_distance_in_mm = torch.sqrt(torch.sum(diff ** 2, dim=-1))

        predict_face_keypoints = self.lip_static_embedding.extract_lip_keypoints(predict.view(-1, self.vertice_dim//3,3) * 1000.0)
        upper_lip_midpoint = predict_face_keypoints[:, 45]  # this is fixed for the ring embedding
        lower_lip_midpoint = predict_face_keypoints[:, 49]  # this is fixed for the ring embedding
        diff = upper_lip_midpoint - lower_lip_midpoint
        pred_distance_in_mm = torch.sqrt(torch.sum(diff ** 2, dim=-1))

        dist, cost, acc_cost, path = dtw(gt_distance_in_mm.view(-1,1).cpu().numpy(), pred_distance_in_mm.view(-1,1).cpu().numpy(),
                                         dist=lambda x, y: norm(x - y, ord=1))
        dist = dist / acc_cost.shape[0]
        # import pdb; pdb.set_trace()
        return dist

    @torch.no_grad()
    def compute_dtw_full_seq(self, predict, real):
        """
        compute dtw full seq
        """
        real = real.view(-1, self.vertice_dim) * 1000.0
        predict = predict.view(-1, self.vertice_dim) * 1000.0
        dist, cost, acc_cost, path = dtw(real.cpu().numpy(), predict.cpu().numpy(),
                                         dist=lambda x, y: norm(x - y, ord=1))
        dist = dist / acc_cost.shape[0]
        dist = dist / (self.vertice_dim //3) # normalize with the number of vertices

        return dist

    # compute the mbp reconstruction metric
    def compute_mbp_reconstruction_loss(self, predict, real, file_name):
        if self.mbp_reconstruction_loss is not None:
            loss = self.mbp_reconstruction_loss.compute_loss(predict, real, file_name)
        else:
             loss = torch.tensor(0, dtype=predict.dtype, device=predict.device)
        return loss


    def pred_based_compute_mbp_reconstruction(self, predict, real, file_name):
        if self.mbp_reconstruction_loss is not None:
            loss = self.mbp_reconstruction_loss.compute_loss(predict, real, file_name)
        else:
            loss = torch.tensor(0, dtype=predict.dtype, device=predict.device)
        return loss


    @torch.no_grad()
    def compute_mbp_reconstruction_metric(self, predict, real, file_name):
        if self.mbp_reconstruction_loss is not None:
            loss = self.mbp_reconstruction_loss.compute_metric_in_mm(predict, real, file_name)
        else:
             loss = torch.tensor(0, dtype=predict.dtype, device=predict.device)
        return loss

    @torch.no_grad()
    def lip_sync_metric(self, predict, real, mask):
        """
        This is the lip sync metric used in the faceformer paper
        """
        mask = mask.to(real.device)
        lip_pred = predict * mask
        lip_real = real * mask

        pred_verts_mm = lip_pred.view(-1, self.vertice_dim//3, 3) * 1000.0
        gt_verts_mm = lip_real.view(-1, self.vertice_dim//3, 3) * 1000.0

        l2_diff_all_lip_vert = torch.mean(self.mse_loss(pred_verts_mm, gt_verts_mm), dim=-1)
        max_l2_error_lip_vert, idx = torch.max(l2_diff_all_lip_vert, dim=-1)
        mean_max_l2_error_lip_vert = torch.mean(max_l2_error_lip_vert)
        return mean_max_l2_error_lip_vert


    @torch.no_grad()
    def lip_sync_metric_l2(self, predict, real, mask):
        """
        This is the lip sync metric used in the faceformer paper
        """
        mask = mask.to(real.device)
        lip_pred = predict * mask
        lip_real = real * mask

        pred_verts_mm = lip_pred.view(-1, self.vertice_dim//3, 3) * 1000.0
        gt_verts_mm = lip_real.view(-1, self.vertice_dim//3, 3) * 1000.0

        diff_in_mm = pred_verts_mm - gt_verts_mm
        l2_dist_in_mm = torch.sqrt(torch.sum(diff_in_mm ** 2, dim=-1)) # Nf x 5023
        max_l2_error_lip_vert, idx = torch.max(l2_dist_in_mm, dim=-1) # Nf x 1
        mean_max_l2_error_lip_vert = torch.mean(max_l2_error_lip_vert)  # 1
        return mean_max_l2_error_lip_vert

    def mbp_weight_reg_loss(self, pred_mbp_weight, filename):
        mbp_weight_reg_loss_weight = self.loss_dict.get('mbp_weight_reg_loss', 0.0)
        if mbp_weight_reg_loss_weight > 0.0:
            gt_per_frame_weights = self.mbp_reconstruction_loss.normalized_weight_dict[filename][:pred_mbp_weight.shape[1]].to(pred_mbp_weight.device)
            gt_per_frame_weights = gt_per_frame_weights.view(1, -1, 1)
            mbp_weight_reg_loss = self.mbp_reg_loss_type(gt_per_frame_weights, pred_mbp_weight)

            # print("\nngt normalized weights", ", ".join([str(gt_per_frame_weights[0, x, 0].item()) for x in range(60)]))
            # print("pref normalized weights", ", ".join([str(pred_mbp_weight[0, x, 0].item()) for x in range(60)]))
            #
            # print("gt normalized weights mean", torch.mean(gt_per_frame_weights).item())
            # print("pref normalized weights mean",torch.mean(pred_mbp_weight).item())
            #
            #
            # print("mbp_weight_reg_loss", mbp_weight_reg_loss.item())
            # print("mbp_weight_reg_loss weighted",mbp_weight_reg_loss.item() * mbp_weight_reg_loss_weight)
            return mbp_weight_reg_loss, mbp_weight_reg_loss_weight * mbp_weight_reg_loss
        else:
            return torch.tensor(0, dtype=pred_mbp_weight.dtype, device=pred_mbp_weight.device), torch.tensor(0, dtype=pred_mbp_weight.dtype, device=pred_mbp_weight.device)

    def pred_based_mbp_reconstruction_loss(self, pred, real, pred_per_frame_mbp_weights):
        pred_based_mbp_reconstruction_loss_weight = self.loss_dict.get('pred_based_mbp_reconstruction_loss', 0.0)
        if pred_based_mbp_reconstruction_loss_weight > 0.0:
            biased_full_reconstruction_loss = self.mse_loss(pred, real)
            pred_per_frame_mbp_weights = pred_per_frame_mbp_weights.view(1, -1, 1)
            pred_based_mbp_reconstruction_loss = torch.mean(pred_per_frame_mbp_weights * biased_full_reconstruction_loss)
            return pred_based_mbp_reconstruction_loss, pred_based_mbp_reconstruction_loss_weight * pred_based_mbp_reconstruction_loss
        else:
            return torch.tensor(0, dtype=pred.dtype, device=pred.device), torch.tensor(0, dtype=pred.dtype, device=pred.device)




