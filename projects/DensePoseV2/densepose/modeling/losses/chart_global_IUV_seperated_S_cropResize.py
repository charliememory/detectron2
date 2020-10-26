# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Any, List
import torch, pdb
from torch.nn import functional as F

from detectron2.config import CfgNode
from detectron2.structures import Instances

from .. import DensePoseConfidenceModelConfig, DensePoseUVConfidenceType
# from .chart import DensePoseChartLoss
from .chart_global_IUV_seperated_S import DensePoseChartGlobalIUVSeparatedSLoss
from .registry import DENSEPOSE_LOSS_REGISTRY
from .utils import BilinearInterpolationHelper, LossDict, SingleTensorsHelper


def dice_coefficient(x, target):
    eps = 1e-5
    n_inst = x.size(0)
    x = x.reshape(n_inst, -1)
    target = target.reshape(n_inst, -1)
    intersection = (x * target).sum(dim=1)
    union = (x ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1) + eps
    loss = 1. - (2 * intersection / union)
    return loss


@DENSEPOSE_LOSS_REGISTRY.register()
class DensePoseChartGlobalIUVSeparatedSCropResizeLoss(DensePoseChartGlobalIUVSeparatedSLoss):
    """"""


    def produce_densepose_losses(
        self, proposals_with_gt: List[Instances], densepose_predictor_outputs: Any,
        gt_bitmasks: Any,
    ) -> LossDict:
        """
        Losses for segmentation and U/V coordinates computed as cross-entropy
        for segmentation unnormalized scores given ground truth labels at
        annotated points and smooth L1 loss for U and V coordinate estimates at
        annotated points.

        Args:
            proposals_with_gt (list of Instances): detections with associated ground truth data
            densepose_predictor_outputs: DensePose predictor outputs, an object
                of a dataclass that is assumed to have the following attributes:
             * fine_segm - fine segmentation estimates, tensor of shape [N, C, S, S]
             * u - U coordinate estimates per fine labels, tensor of shape [N, C, S, S]
             * v - V coordinate estimates per fine labels, tensor of shape [N, C, S, S]
        Return:
            dict: str -> tensor: dict of losses with the following entries:
             * `loss_densepose_U`: smooth L1 loss for U coordinate estimates
             * `loss_densepose_V`: smooth L1 loss for V coordinate estimates
             * `loss_densepose_I`: cross entropy for raw unnormalized scores for fine
                 segmentation estimates given ground truth labels
             * `loss_densepose_S`: cross entropy for raw unnormalized scores for coarse
                 segmentation estimates given ground truth labels;
                 may be included if coarse segmentation is only trained
                 using DensePose ground truth; if additional supervision through
                 instance segmentation data is performed (`segm_trained_by_masks` is True),
                 this loss is handled by `produce_mask_losses` instead
        """
        # densepose outputs are computed for all images and all bounding boxes;
        # i.e. if a batch has 4 images with (3, 1, 2, 1) proposals respectively,
        # the outputs will have size(0) == 3+1+2+1 == 7
        # densepose_outputs_size = densepose_predictor_outputs.u.size()

        if not len(proposals_with_gt):
            return self.produce_fake_densepose_losses(densepose_predictor_outputs)

        tensors_helper = SingleTensorsHelper(proposals_with_gt)
        n_batch = len(tensors_helper.index_with_dp)

        # NOTE: we need to keep the same computation graph on all the GPUs to
        # perform reduction properly. Hence even if we have no data on one
        # of the GPUs, we still need to generate the computation graph.
        # Add fake (zero) loss in the form Tensor.sum() * 0
        if not n_batch:
            return self.produce_fake_densepose_losses(densepose_predictor_outputs)

        interpolator = BilinearInterpolationHelper.from_matches(
            tensors_helper, [None,None,256,256]
        )

        j_valid_fg = interpolator.j_valid * (tensors_helper.fine_segm_labels_gt > 0)

        # self.zero_training_loss = False
        losses_uv = self.produce_densepose_losses_uv(
            proposals_with_gt, densepose_predictor_outputs, tensors_helper, interpolator, j_valid_fg
        )

        # if not self.segm_trained_by_masks:
        losses_segm = self.produce_densepose_losses_segm(
            proposals_with_gt, densepose_predictor_outputs, tensors_helper, interpolator, j_valid_fg,
            gt_bitmasks,
        )

        return {**losses_uv, **losses_segm}
        # else:
        #     return {**losses_uv}

    def produce_densepose_losses_uv(
        self,
        proposals_with_gt: List[Instances],
        densepose_predictor_outputs: Any,
        tensors_helper: SingleTensorsHelper,
        interpolator: BilinearInterpolationHelper,
        j_valid_fg: torch.Tensor,
    ) -> LossDict:
        """
        Compute losses for U/V coordinates: smooth L1 loss between
        estimated coordinates and the ground truth.

        Args:
            proposals_with_gt (list of Instances): detections with associated ground truth data
            densepose_predictor_outputs: DensePose predictor outputs, an object
                of a dataclass that is assumed to have the following attributes:
             * u - U coordinate estimates per fine labels, tensor of shape [N, C, S, S]
             * v - V coordinate estimates per fine labels, tensor of shape [N, C, S, S]
        Return:
            dict: str -> tensor: dict of losses with the following entries:
             * `loss_densepose_U`: smooth L1 loss for U coordinate estimates
             * `loss_densepose_V`: smooth L1 loss for V coordinate estimates
        """
        u_gt = tensors_helper.u_gt[j_valid_fg]
        u_est = interpolator.extract_at_points_globalIUV_crop_resize(
            densepose_predictor_outputs.u
        )[j_valid_fg]

        v_gt = tensors_helper.v_gt[j_valid_fg]
        v_est = interpolator.extract_at_points_globalIUV_crop_resize(
            densepose_predictor_outputs.v
        )[j_valid_fg]

        # u_gt = tensors_helper.u_gt[j_valid_fg]
        # u_est = interpolator.extract_at_points(
        #     densepose_predictor_outputs.u[tensors_helper.index_with_dp]
        # )[j_valid_fg]

        # v_gt = tensors_helper.v_gt[j_valid_fg]
        # v_est = interpolator.extract_at_points(
        #     densepose_predictor_outputs.v[tensors_helper.index_with_dp]
        # )[j_valid_fg]
        # pdb.set_trace()

        losses = {
            "loss_densepose_U": F.smooth_l1_loss(u_est, u_gt, reduction="sum") * self.w_points,
            "loss_densepose_V": F.smooth_l1_loss(v_est, v_gt, reduction="sum") * self.w_points,
        }
        # losses = {
        #     "loss_densepose_U": F.smooth_l1_loss(u_est, u_gt, reduction="mean") * self.w_points,
        #     "loss_densepose_V": F.smooth_l1_loss(v_est, v_gt, reduction="mean") * self.w_points,
        # }
        # if hasattr(self,'prev_loss_U'):
        #     if losses["loss_densepose_U"]>self.prev_loss_U and losses["loss_densepose_U"]>0.5:
        #         print(losses)
        #         losses["loss_densepose_U"] = losses["loss_densepose_U"]*0
        #         losses["loss_densepose_V"] = losses["loss_densepose_V"]*0
        #         print("==> pass loss")
        #     else:
        #         self.prev_loss_U = losses["loss_densepose_U"]
        # else:
        #     self.prev_loss_U = losses["loss_densepose_U"]

        # if losses["loss_densepose_V"]>=0.5:
        #     print(losses)
        #     pdb.set_trace()
        for k,v in losses.items():
            if torch.isnan(v):
                print(k,v)
                pdb.set_trace()

        return losses
        # return {
        #     "loss_densepose_U": F.smooth_l1_loss(u_est, u_gt, reduction="mean") * self.w_points,
        #     "loss_densepose_V": F.smooth_l1_loss(v_est, v_gt, reduction="mean") * self.w_points,
        # }

    def produce_densepose_losses_segm(
        self,
        proposals_with_gt: List[Instances],
        densepose_predictor_outputs: Any,
        tensors_helper: SingleTensorsHelper,
        interpolator: BilinearInterpolationHelper,
        j_valid_fg: torch.Tensor,
        gt_bitmasks: Any,
    ) -> LossDict:
        """
        Losses for fine / coarse segmentation: cross-entropy
        for segmentation unnormalized scores given ground truth labels at
        annotated points for fine segmentation and dense mask annotations
        for coarse segmentation.

        Args:
            proposals_with_gt (list of Instances): detections with associated ground truth data
            densepose_predictor_outputs: DensePose predictor outputs, an object
                of a dataclass that is assumed to have the following attributes:
             * fine_segm - fine segmentation estimates, tensor of shape [N, C, S, S]
             * coarse_segm - coarse segmentation estimates, tensor of shape [N, D, S, S]
        Return:
            dict: str -> tensor: dict of losses with the following entries:
             * `loss_densepose_I`: cross entropy for raw unnormalized scores for fine
                 segmentation estimates given ground truth labels
             * `loss_densepose_S`: cross entropy for raw unnormalized scores for coarse
                 segmentation estimates given ground truth labels;
                 may be included if coarse segmentation is only trained
                 using DensePose ground truth; if additional supervision through
                 instance segmentation data is performed (`segm_trained_by_masks` is True),
                 this loss is handled by `produce_mask_losses` instead
        """

        fine_segm_gt = tensors_helper.fine_segm_labels_gt[interpolator.j_valid]
        fine_segm_est = interpolator.extract_at_points_globalIUV_crop_resize(
            densepose_predictor_outputs.fine_segm,
            slice_fine_segm=slice(None),
            w_ylo_xlo=interpolator.w_ylo_xlo[:, None],
            w_ylo_xhi=interpolator.w_ylo_xhi[:, None],
            w_yhi_xlo=interpolator.w_yhi_xlo[:, None],
            w_yhi_xhi=interpolator.w_yhi_xhi[:, None],
            mode='bilinear', #'nearest'
        )[interpolator.j_valid, :]

        losses = {
            "loss_densepose_I": F.cross_entropy(fine_segm_est, fine_segm_gt.long()) * self.w_part
        }
        # print(fine_segm_gt)
        # print(torch.argmax(fine_segm_est,dim=-1))
        # print(losses)
        # import imageio
        # pdb.set_trace()
        # imageio.imwrite("tmp/fine_segm_gt.png", (fine_segm_gt/24.0)[0].detach().cpu().numpy())
        # imageio.imwrite("tmp/fine_segm_est.png", (torch.argmax(fine_segm_est,dim=-1)/24.0)[0].detach().cpu().numpy())

        # m = fine_segm_gt.long()>0
        # loss = F.cross_entropy(fine_segm_est, fine_segm_gt.long(), reduction='none')
        # pdb.set_trace()
        # import imageio
        # est = densepose_predictor_outputs.coarse_segm
        # imageio.imwrite("tmp/est.png", (fine_segm_gt/24.0)[0,0].detach().cpu().numpy())

        # loss = (loss*m).mean()
        # losses = {
        #     "loss_densepose_I": loss * self.w_part
        # }

        # Resample everything to the estimated data size, no need to resample
        # S_est then:
        if not self.segm_trained_by_masks:
            assert self.n_segm_chan==2
            coarse_segm_gt = tensors_helper.coarse_segm_gt
            coarse_segm_est = densepose_predictor_outputs.coarse_segm[tensors_helper.index_with_dp]
            coarse_segm_est = interpolator.extract_at_points_separatedS_crop_resize(
                coarse_segm_est,
                slice_index_uv=slice(None),
                mode='bilinear', #'nearest'
            )
            if self.n_segm_chan == 2:
                coarse_segm_gt = coarse_segm_gt > 0
            losses["loss_densepose_S"] = (
                F.cross_entropy(coarse_segm_est, coarse_segm_gt.long()) * self.w_segm
            )
        else:
            assert self.n_segm_chan==1
            loss_mask = dice_coefficient(densepose_predictor_outputs.coarse_segm, gt_bitmasks)
            losses["loss_densepose_S"] = loss_mask.mean() * self.w_segm


        return losses


