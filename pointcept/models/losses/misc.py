"""
Misc Losses

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .builder import LOSSES
from pointcept.utils.logger import get_logger
from pointcept.extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2


def ignore_label(scores, labels, ignore=None, valid=None, scene_id=None):
    if ignore is None:
        return scores, labels
    if(valid is None):
        valid = labels != ignore
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels

@LOSSES.register_module('MSELoss')
class MSELoss(nn.Module):
    def __init__(
        self,
        size_average=None,
        reduce=None,
        reduction="mean",
        loss_weight=1.0,
    ):
        super(MSELoss, self).__init__()
        self.loss_weight = loss_weight
        # self.loss = nn.MSELoss(
        #     size_average=size_average,
        #     reduce=reduce,
        #     reduction=reduction,
        # )

    def forward(self, pred, target):
        N, C = target.shape
        mask = torch.zeros((N, C))
        mask[target >= 0.7] = 1

        zero_pred = pred.clone()
        one_pred = pred.clone()
        zero_target = target.clone()
        one_target = target.clone()

        zero_pred[mask == 1] = 0
        zero_target[mask == 1] = 0
        one_pred[mask == 0] = 0
        one_target[mask == 0] = 0

        loss = torch.sum(torch.pow(zero_pred - zero_target, 2)) / (N * C - (mask == 1).sum().item() + 1e-6) + torch.sum(
            torch.pow(one_pred - one_target, 2)) / (N * C - (mask == 0).sum().item() + 1e-6)
        loss = loss * self.loss_weight
        return loss

@LOSSES.register_module()
class CosineLoss(nn.Module):
    def __init__(
        self,
        size_average=None,
        reduce=None,
        reduction="mean",
        loss_weight=1.0,
    ):
        super(CosineLoss, self).__init__()
        self.loss = nn.CosineEmbeddingLoss(
            size_average=size_average,
            reduce=reduce,
            reduction=reduction
        )
        self.loss_weight = loss_weight
        self.logger = get_logger(name="pointcept")

    def forward(self, pred, target):
        label = torch.ones(target.shape[0]).cuda()
        inp = torch.sqrt(torch.sum(pred.mul(pred), dim=-1) + 0.00000001).view(pred.shape[0], 1)
        pred_norm = pred / (inp + 0.0000000001)
        loss = torch.min(self.loss(pred_norm, target, label), self.loss(-pred_norm, target, label))
        if torch.isnan(loss).any():
            self.logger.info(f'Nan value occurs: {torch.isnan(pred).any()}, {torch.isnan(pred_norm).any()}')
            self.logger.info(f'{torch.min(pred)}, {torch.max(pred)}')
        loss = loss.mean() * self.loss_weight
        return loss

@LOSSES.register_module()
class CosineLossV2(nn.Module):
    def __init__(
        self,
        size_average=None,
        reduce=None,
        reduction="mean",
        loss_weight=1.0,
        edge_weight=0.5,
    ):
        super(CosineLossV2, self).__init__()
        self.loss = nn.CosineEmbeddingLoss(
            size_average=size_average,
            reduce=reduce,
            reduction=reduction
        )
        self.loss_weight = loss_weight
        self.edge_weight = edge_weight
        self.logger = get_logger(name="pointcept")

    def forward(self, pred, target):
        assert len(target) == 2
        target_, cls_label = target
        inp = torch.sqrt(torch.sum(pred.mul(pred), dim=-1) + 0.00000001).view(pred.shape[0], 1)
        pred_norm = pred / (inp + 0.0000000001)
        pred_norm_edge = pred_norm[cls_label == 1]
        pred_norm_non_edge = pred_norm[cls_label == 0]
        target_edge = target_[cls_label == 1]
        target_non_edge = target_[cls_label == 0]
        label_edge = torch.ones(target_edge.shape[0]).cuda()
        label_non_edge = torch.ones(target_non_edge.shape[0]).cuda()
        loss_edge = torch.min(self.loss(pred_norm_edge, target_edge, label_edge), self.loss(-pred_norm_edge, target_edge, label_edge))
        loss_non_edge = torch.min(self.loss(pred_norm_non_edge, target_non_edge, label_non_edge), self.loss(-pred_norm_non_edge, target_non_edge, label_non_edge))
        if torch.isnan(loss_edge).any() or torch.isnan(loss_non_edge).any():
            self.logger.info(f'Nan value occurs: {torch.isnan(pred).any()}, {torch.isnan(pred_norm).any()}')
            self.logger.info(f'{torch.min(pred)}, {torch.max(pred)}')
        loss = loss_non_edge.mean() * (1-self.edge_weight) + loss_edge.mean() * self.edge_weight
        return loss * self.loss_weight


@LOSSES.register_module()
class SmoothLoss(nn.Module):
    def __init__(
        self,
        size_average=None,
        reduce=None,
        reduction="mean",
        loss_weight=1.0,
        k=3
    ):
        super(SmoothLoss, self).__init__()
        self.k = k
        self.loss_weight = loss_weight
        self.logger = get_logger(name="pointcept")

    def compute_knn(self, points, k):
        """
        Compute the k nearest neighbor of input points
        input: points: (n, 3),
        output: idx: (n, k)
        """
        distance_matrix = torch.cdist(points, points, p=2)  # [N, N]
        knn_indices = torch.topk(distance_matrix, k=k + 1, largest=False).indices  # [N, k+1]
        return knn_indices[:, 1:]  # [N, k]

    def forward(self, pred, target):
        assert len(pred) == 3 and len(target) == 2
        pred_, coord, nums = pred
        target_, cls_label = target
        # self.logger.info(f'{coord.shape[0]}, {coord.shape[0] % 16384}, {nums}')

        inp = torch.sqrt(torch.sum(pred_.mul(pred_), dim=-1) + 0.00000001).view(pred_.shape[0], 1)
        pred_norm = pred_ / (inp + 0.0000000001)

        num = 0
        loss = 0.0
        for i in range(len(nums)):
            coord_part = coord[num : num + nums[i]]
            pred_norm_part = pred_norm[num : num + nums[i]]
            cls_label_part = cls_label[num : num + nums[i]]
            target_part = target_[num : num + nums[i]]
            pred_norm_edge = pred_norm_part[cls_label_part == 1]
            coord_edge = coord_part[cls_label_part == 1]

            knn_indices = self.compute_knn(coord_edge, self.k)  # [N, k]
            neighbor_directions = pred_norm_edge[knn_indices]  # [N, k, 3]
            central_directions = pred_norm_edge.unsqueeze(1)  # [N, 1, 3]
            similarity = torch.abs(torch.sum(neighbor_directions * central_directions, dim=2))  # [N, k]
            similarity = torch.clamp(similarity, -1.0, 1.0)

            target_edge = target_part[cls_label_part == 1]
            target_neighbor_directions = target_edge[knn_indices]  # [N, k, 3]
            target_central_directions = target_edge.unsqueeze(1)  # [N, 1, 3]
            target_similarity = torch.abs(torch.sum(target_neighbor_directions * target_central_directions, dim=2))  # [N, k]
            target_similarity = torch.clamp(target_similarity, -1.0, 1.0)

            loss += F.mse_loss(similarity, target_similarity)

            num += nums[i]

        smooth_loss = loss / len(nums)

        return smooth_loss * self.loss_weight


@LOSSES.register_module()
class SmoothLossV2(nn.Module):
    def __init__(
        self,
        size_average=None,
        reduce=None,
        reduction="mean",
        loss_weight=1.0,
        k=3
    ):
        super(SmoothLossV2, self).__init__()
        self.k = k
        self.loss_weight = loss_weight
        self.logger = get_logger(name="pointcept")

    def compute_knn(self, points, k):
        """
        Compute the k nearest neighbor of input points
        input: points: (n, 3),
        output: idx: (n, k)
        """
        distance_matrix = torch.cdist(points, points, p=2)  # [N, N]
        knn_indices = torch.topk(distance_matrix, k=k + 1, largest=False).indices  # [N, k+1]
        return knn_indices[:, 1:]  # [N, k]

    def forward(self, pred, target):
        assert len(pred) == 3
        pred_, coord, nums = pred

        inp = torch.sqrt(torch.sum(pred_.mul(pred_), dim=-1) + 0.00000001).view(pred_.shape[0], 1)
        pred_norm = pred_ / (inp + 0.0000000001)
        
        num = 0
        loss = 0.0
        for i in range(len(nums)):
            coord_part = coord[num : num + nums[i]]
            pred_norm_part = pred_norm[num : num + nums[i]]
            target_part = target[num : num + nums[i]]

            knn_indices = self.compute_knn(coord_part, self.k)  # [N, k]
            neighbor_directions = pred_norm_part[knn_indices]  # [N, k, 3]
            central_directions = pred_norm_part.unsqueeze(1)  # [N, 1, 3]
            similarity = torch.abs(torch.sum(neighbor_directions * central_directions, dim=2))  # [N, k]
            similarity = torch.clamp(similarity, -1.0, 1.0)

            target_neighbor_directions = target_part[knn_indices]  # [N, k, 3]
            target_central_directions = target_part.unsqueeze(1)  # [N, 1, 3]
            target_similarity = torch.abs(
                torch.sum(target_neighbor_directions * target_central_directions, dim=2))  # [N, k]
            target_similarity = torch.clamp(target_similarity, -1.0, 1.0)

            loss += F.mse_loss(similarity, target_similarity)

            num += nums[i]

        smooth_loss = loss / len(nums)

        return smooth_loss * self.loss_weight


@LOSSES.register_module()
class ChamferLoss(nn.Module):
    def __init__(
        self,
        cd_type='L1',
        loss_weight=1.0,
    ):
        super(ChamferLoss, self).__init__()
        assert cd_type in ['L1', 'L2']
        self.loss_weight = loss_weight
        if cd_type == 'L1':
            self.loss = ChamferDistanceL1()
        else:
            self.loss = ChamferDistanceL2()

    def forward(self, pred, target):
        assert len(pred) == 3
        pred_pc, nums, dense_nums = pred
        
        num = 0
        dense_num = 0
        loss = 0.0
        for i in range(len(nums)):
            pred_pc_part = pred_pc[num : num + nums[i]]
            target_pc_part = target[dense_num : dense_num + dense_nums[i]]
            loss += self.loss(pred_pc_part.unsqueeze(0), target_pc_part.unsqueeze(0))

            num += nums[i]
            dense_num += dense_nums[i]

        cd_loss = loss / len(nums)
        
        return cd_loss * self.loss_weight


@LOSSES.register_module()
class ChamferLossV2(nn.Module):
    def __init__(
            self,
            cd_type='L1',
            loss_weight=1.0,
    ):
        super(ChamferLossV2, self).__init__()
        assert cd_type in ['L1', 'L2']
        self.loss_weight = loss_weight
        if cd_type == 'L1':
            self.loss = ChamferDistanceL1()
        else:
            self.loss = ChamferDistanceL2()

    def forward(self, pred, target):
        assert len(pred) == 4
        pred_pc, nums, dense_nums, label = pred

        num = 0
        dense_num = 0
        loss = 0.0
        for i in range(len(nums)):
            pred_pc_part = pred_pc[num: num + nums[i]]
            label_part = label[num: num + nums[i]]
            target_pc_part = target[dense_num: dense_num + dense_nums[i]]
            loss += self.loss(pred_pc_part[label_part == 1].unsqueeze(0), target_pc_part.unsqueeze(0))

            num += nums[i]
            dense_num += dense_nums[i]

        cd_loss = loss / len(nums)

        return cd_loss * self.loss_weight


@LOSSES.register_module()
class CrossEntropyLoss(nn.Module):
    def __init__(
        self,
        weight=None,
        size_average=None,
        reduce=None,
        reduction="mean",
        label_smoothing=0.0,
        loss_weight=1.0,
        ignore_index=-1,
    ):
        super(CrossEntropyLoss, self).__init__()
        weight = torch.tensor(weight).cuda() if weight is not None else None
        self.loss_weight = loss_weight
        self.loss = nn.CrossEntropyLoss(
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )

    def forward(self, pred, target):
        return self.loss(pred, target) * self.loss_weight


@LOSSES.register_module()
class SmoothCELoss(nn.Module):
    def __init__(self, smoothing_ratio=0.1):
        super(SmoothCELoss, self).__init__()
        self.smoothing_ratio = smoothing_ratio

    def forward(self, pred, target):
        eps = self.smoothing_ratio
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prb).total(dim=1)
        loss = loss[torch.isfinite(loss)].mean()
        return loss


@LOSSES.register_module()
class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.5, logits=True, reduce=True, loss_weight=1.0):
        """Binary Focal Loss
        <https://arxiv.org/abs/1708.02002>`
        """
        super(BinaryFocalLoss, self).__init__()
        assert 0 < alpha < 1
        self.gamma = gamma
        self.alpha = alpha
        self.logits = logits
        self.reduce = reduce
        self.loss_weight = loss_weight

    def forward(self, pred, target, **kwargs):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction with shape (N)
            target (torch.Tensor): The ground truth. If containing class
                indices, shape (N) where each value is 0≤targets[i]≤1, If containing class probabilities,
                same shape as the input.
        Returns:
            torch.Tensor: The calculated loss
        """
        if self.logits:
            bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        else:
            bce = F.binary_cross_entropy(pred, target, reduction="none")
        pt = torch.exp(-bce)
        alpha = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_loss = alpha * (1 - pt) ** self.gamma * bce

        if self.reduce:
            focal_loss = torch.mean(focal_loss)
        return focal_loss * self.loss_weight


@LOSSES.register_module()
class FocalLoss(nn.Module):
    def __init__(
        self, gamma=2.0, alpha=0.5, reduction="mean", loss_weight=1.0, ignore_index=-1
    ):
        """Focal Loss
        <https://arxiv.org/abs/1708.02002>`
        """
        super(FocalLoss, self).__init__()
        assert reduction in (
            "mean",
            "sum",
        ), "AssertionError: reduction should be 'mean' or 'sum'"
        assert isinstance(
            alpha, (float, list)
        ), "AssertionError: alpha should be of type float"
        assert isinstance(gamma, float), "AssertionError: gamma should be of type float"
        assert isinstance(
            loss_weight, float
        ), "AssertionError: loss_weight should be of type float"
        assert isinstance(ignore_index, int), "ignore_index must be of type int"
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, pred, target, **kwargs):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction with shape (N, C) where C = number of classes.
            target (torch.Tensor): The ground truth. If containing class
                indices, shape (N) where each value is 0≤targets[i]≤C−1, If containing class probabilities,
                same shape as the input.
        Returns:
            torch.Tensor: The calculated loss
        """
        # [B, C, d_1, d_2, ..., d_k] -> [C, B, d_1, d_2, ..., d_k]
        pred = pred.transpose(0, 1)
        # [C, B, d_1, d_2, ..., d_k] -> [C, N]
        pred = pred.reshape(pred.size(0), -1)
        # [C, N] -> [N, C]
        pred = pred.transpose(0, 1).contiguous()
        # (B, d_1, d_2, ..., d_k) --> (B * d_1 * d_2 * ... * d_k,)
        target = target.view(-1).contiguous()
        assert pred.size(0) == target.size(
            0
        ), "The shape of pred doesn't match the shape of target"
        valid_mask = target != self.ignore_index
        target = target[valid_mask]
        pred = pred[valid_mask]

        if len(target) == 0:
            return 0.0

        num_classes = pred.size(1)
        target = F.one_hot(target, num_classes=num_classes)

        alpha = self.alpha
        if isinstance(alpha, list):
            alpha = pred.new_tensor(alpha)
        pred_sigmoid = pred.sigmoid()
        target = target.type_as(pred)
        one_minus_pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
        focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * one_minus_pt.pow(
            self.gamma
        )

        loss = (
            F.binary_cross_entropy_with_logits(pred, target, reduction="none")
            * focal_weight
        )
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.total()
        return self.loss_weight * loss


@LOSSES.register_module()
class DiceLoss(nn.Module):
    def __init__(self, smooth=1, exponent=2, loss_weight=1.0, ignore_index=-1):
        """DiceLoss.
        This loss is proposed in `V-Net: Fully Convolutional Neural Networks for
        Volumetric Medical Image Segmentation <https://arxiv.org/abs/1606.04797>`_.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.exponent = exponent
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, pred, target, **kwargs):
        # [B, C, d_1, d_2, ..., d_k] -> [C, B, d_1, d_2, ..., d_k]
        pred = pred.transpose(0, 1)
        # [C, B, d_1, d_2, ..., d_k] -> [C, N]
        pred = pred.reshape(pred.size(0), -1)
        # [C, N] -> [N, C]
        pred = pred.transpose(0, 1).contiguous()
        # (B, d_1, d_2, ..., d_k) --> (B * d_1 * d_2 * ... * d_k,)
        target = target.view(-1).contiguous()
        assert pred.size(0) == target.size(
            0
        ), "The shape of pred doesn't match the shape of target"
        valid_mask = target != self.ignore_index
        target = target[valid_mask]
        pred = pred[valid_mask]

        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        target = F.one_hot(
            torch.clamp(target.long(), 0, num_classes - 1), num_classes=num_classes
        )

        total_loss = 0
        for i in range(num_classes):
            if i != self.ignore_index:
                num = torch.sum(torch.mul(pred[:, i], target[:, i])) * 2 + self.smooth
                den = (
                    torch.sum(
                        pred[:, i].pow(self.exponent) + target[:, i].pow(self.exponent)
                    )
                    + self.smooth
                )
                dice_loss = 1 - num / den
                total_loss += dice_loss
        loss = total_loss / num_classes
        return self.loss_weight * loss
