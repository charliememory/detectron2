# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Any, List
import torch, pdb, copy
from torch.nn import functional as F

from detectron2.config import CfgNode
from detectron2.structures import Instances

from densepose.utils.comm import compute_locations, compute_grid, aligned_bilinear
from .. import DensePoseConfidenceModelConfig, DensePoseUVConfidenceType
from .chart import DensePoseChartLoss
from .registry import DENSEPOSE_LOSS_REGISTRY

from .utils import (
    BilinearInterpolationHelper,
    ChartBasedAnnotationsAccumulator,
    LossDict,
    extract_packed_annotations_from_matches,
    dice_coefficient, FocalLoss, smooth_loss, tv_loss
)

@DENSEPOSE_LOSS_REGISTRY.register()
class DensePoseChartGlobalIUVSeparatedSPoolerLoss(DensePoseChartLoss):
    def __call__(
        self, proposals_with_gt: List[Instances], densepose_predictor_outputs: Any,
        gt_bitmasks: Any, images=None, skeleton_feats_gt=None, body_semantics_gt=None,
        features_dp_ori=None, gt_bitmasks_body=None
    ) -> LossDict:
        """
        Produce chart-based DensePose losses

        Args:
            proposals_with_gt (list of Instances): detections with associated ground truth data
            densepose_predictor_outputs: an object of a dataclass that contains predictor outputs
                with estimated values; assumed to have the following attributes:
                * coarse_segm - coarse segmentation estimates, tensor of shape [N, D, S, S]
                * fine_segm - fine segmentation estimates, tensor of shape [N, C, S, S]
                * u - U coordinate estimates per fine labels, tensor of shape [N, C, S, S]
                * v - V coordinate estimates per fine labels, tensor of shape [N, C, S, S]
            where N is the number of detections, C is the number of fine segmentation
            labels, S is the estimate size ( = width = height) and D is the number of
            coarse segmentation channels.

        Return:
            (dict: str -> tensor): dict of losses with the following entries:
                * loss_densepose_I: fine segmentation loss (cross-entropy)
                * loss_densepose_S: coarse segmentation loss (cross-entropy)
                * loss_densepose_U: loss for U coordinates (smooth L1)
                * loss_densepose_V: loss for V coordinates (smooth L1)
        """
        if not len(proposals_with_gt):
            return self.produce_fake_densepose_losses(densepose_predictor_outputs)

        accumulator = ChartBasedAnnotationsAccumulator()
        packed_annotations = extract_packed_annotations_from_matches(proposals_with_gt, accumulator)

        # NOTE: we need to keep the same computation graph on all the GPUs to
        # perform reduction properly. Hence even if we have no data on one
        # of the GPUs, we still need to generate the computation graph.
        # Add fake (zero) loss in the form Tensor.sum() * 0
        if packed_annotations is None:
            return self.produce_fake_densepose_losses(densepose_predictor_outputs)

        interpolator = BilinearInterpolationHelper.from_matches(
            packed_annotations, tuple(densepose_predictor_outputs.u.shape[2:])
        )

        j_valid_fg = interpolator.j_valid * (packed_annotations.fine_segm_labels_gt > 0)

        losses_uv = self.produce_densepose_losses_uv(
            proposals_with_gt,
            densepose_predictor_outputs,
            packed_annotations,
            interpolator,
            j_valid_fg,
        )

        losses_segm = self.produce_densepose_losses_segm(
            proposals_with_gt,
            densepose_predictor_outputs,
            packed_annotations,
            interpolator,
            j_valid_fg,
            gt_bitmasks,
            body_semantics_gt,
            gt_bitmasks_body,
        )

        losses_smooth = self.produce_densepose_losses_smooth(
            proposals_with_gt,
            features_dp_ori,
        )

        losses_tv = self.produce_densepose_losses_tv(
            proposals_with_gt,
            features_dp_ori,
        )

        # return {**losses_uv, **losses_segm}
        if self.use_aux_global_skeleton:
            losses_skeleton = self.produce_densepose_losses_skeleton(
                proposals_with_gt, densepose_predictor_outputs, skeleton_feats_gt
            )
        else:
            losses_skeleton = {}

        return {**losses_uv, **losses_segm, **losses_smooth, **losses_tv, **losses_skeleton}

                        
    def _torch_dilate(self, binary_img, kernel_size=3, mode='nearest'):
        if kernel_size==0:
            return binary_img
        B,C,H,W = binary_img.shape
        binary_img = binary_img.reshape([B*C,1,H,W])
        if not hasattr(self, 'dilate_kernel'):
            # self.dilate_kernel = torch.Tensor(torch.ones([kernel_size,kernel_size]), device=binary_img.device)[None,None,...]
            self.dilate_kernel = torch.ones([1,1,kernel_size,kernel_size], device=binary_img.device)
        # pdb.set_trace()
        pad = torch.nn.ReflectionPad2d(int(kernel_size//2))

        out = torch.clamp(torch.nn.functional.conv2d(pad(binary_img), self.dilate_kernel, padding=0), 0, 1)
        out = F.interpolate(out, size=binary_img.shape[2:], mode=mode)
        return out.reshape([B,C,H,W])

    "TODO: 1) use dice_coefficient loss instead of simple mes loss"
    def produce_densepose_losses_skeleton(
        self,
        proposals_with_gt: List[Instances],
        densepose_predictor_outputs: Any,
        skeleton_feats_gt: Any,
    ) -> LossDict:
        global_s_gt = [torch.clamp(gt.gt_bitmasks.float().sum(dim=0), 0, 1) for gt in proposals_with_gt]
        global_s_gt = torch.stack(global_s_gt, dim=0)[:,None,...]

        # import imageio
        # imageio.imwrite("output/tmp/skeleton_feats_gt.png", skeleton_feats_gt[0].sum(0).detach().cpu().numpy())
        # pdb.set_trace()

        # skeleton_feats_gt = skeleton_feats_gt * global_s_gt
        skeleton_feats_est = densepose_predictor_outputs.aux_supervision[:,-55:]

        ## standard mse loss
        # loss_skeleton = F.mse_loss(skeleton_feats_est, skeleton_feats_gt, reduction="none").mean(dim=1,keepdim=True)
        # loss_skeleton = (loss_skeleton * global_s_gt).mean() * self.w_aux_global_skeleton
        
        ## dice_coefficient loss
        ## dilate & binarize gt, and then apply dice_coefficient loss to balance FG and BG
        # skeleton_feats_gt = self._torch_dilate(skeleton_feats_gt, kernel_size=3)
        # skeleton_feats_gt = (skeleton_feats_gt>0).float()
        # loss_skeleton = dice_coefficient(skeleton_feats_est*global_s_gt, skeleton_feats_gt*global_s_gt)
        # loss_skeleton = loss_skeleton.mean()  * self.w_aux_global_skeleton

        ## focal-like l2 loss
        loss_skeleton = focal_l2_loss(skeleton_feats_est[None,...], skeleton_feats_gt[None,...], nstack_weight=[1])
        loss_skeleton = loss_skeleton  * self.w_aux_global_skeleton


        losses = {}
        losses["loss_densepose_aux_skeleton"] = loss_skeleton

        return losses


    def produce_densepose_losses_smooth(
        self,
        proposals_with_gt: List[Instances],
        features_dp_ori: Any,
    ) -> LossDict:

        ins_mask_list = [torch.clamp(gt.gt_bitmasks.float().sum(dim=0), 0, 1) for gt in proposals_with_gt]
        if self.use_mean_uv:
            loss = []
            for m in ins_mask_list:
                m = m[None,None,...]
                loss.append(smooth_loss(features_dp_ori*m, m, reduction="mean"))
            return {"loss_densepose_smooth": torch.stack(loss).mean() * self.w_smooth,}
        else:
            loss_x, loss_y = [], []
            for m in ins_mask_list:
                m = m[None,None,...]
                d_x, d_y = smooth_loss(features_dp_ori*m, m, reduction="none")
                loss_x.append(d_x)
                loss_y.append(d_y)
                # pdb.set_trace()
            loss = torch.stack(loss_x).sum(0).mean() + torch.stack(loss_y).sum(0).mean()
            loss *= self.w_smooth
            return {"loss_densepose_smooth": loss}

    def produce_densepose_losses_tv(
        self,
        proposals_with_gt: List[Instances],
        features_dp_ori: Any,
    ) -> LossDict:

        # ins_mask_list = [torch.clamp(gt.gt_bitmasks.float().sum(dim=0), 0, 1) for gt in proposals_with_gt]
        # loss = []
        # for m in ins_mask_list:
        #     # pdb.set_trace()
        #     # fine_segm = m * densepose_predictor_outputs.fine_segm
        #     # u = m * densepose_predictor_outputs.u
        #     # v = m * densepose_predictor_outputs.v
        #     m = m[None,None,...]
        #     loss.append(tv_loss(features_dp_ori))

        return {
            "loss_densepose_tv": tv_loss(features_dp_ori) * self.w_tv,
        }

    def produce_densepose_losses_segm(
        self,
        proposals_with_gt: List[Instances],
        densepose_predictor_outputs: Any,
        packed_annotations: Any,
        interpolator: BilinearInterpolationHelper,
        j_valid_fg: torch.Tensor,
        gt_bitmasks: Any,
        body_semantics_gt: Any,
        gt_bitmasks_body: Any,
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
        fine_segm_gt = packed_annotations.fine_segm_labels_gt[interpolator.j_valid]
        fine_segm_est = interpolator.extract_at_points(
            densepose_predictor_outputs.fine_segm,
            slice_fine_segm=slice(None),
            w_ylo_xlo=interpolator.w_ylo_xlo[:, None],
            w_ylo_xhi=interpolator.w_ylo_xhi[:, None],
            w_yhi_xlo=interpolator.w_yhi_xlo[:, None],
            w_yhi_xhi=interpolator.w_yhi_xhi[:, None],
        )[interpolator.j_valid, :]
        # return {
        #     "loss_densepose_I": F.cross_entropy(fine_segm_est, fine_segm_gt.long()) * self.w_part,
        #     "loss_densepose_S": self.segm_loss(
        #         proposals_with_gt, densepose_predictor_outputs, packed_annotations
        #     )
        #     * self.w_segm,
        # }

        losses = {}
        if self.use_part_focal_loss:
            losses["loss_densepose_I"] = self.focal_loss(fine_segm_est, fine_segm_gt.long()).mean() * self.w_part
        else:
            losses["loss_densepose_I"] = F.cross_entropy(fine_segm_est, fine_segm_gt.long()) * self.w_part
        assert self.n_segm_chan==1
        valid_idxs = gt_bitmasks.abs().mean([1,2,3])>0
        # print(gt_bitmasks.abs().mean([1,2,3]))
        # pdb.set_trace()
        if self.pred_ins_body:
            # pdb.set_trace()
            loss_mask = dice_coefficient(densepose_predictor_outputs.coarse_segm[valid_idxs][:,0], gt_bitmasks[valid_idxs]).mean()
            losses["loss_densepose_S"] = loss_mask * self.w_segm
            loss_body = []
            for ii in  range(gt_bitmasks.shape[0]):
                if gt_bitmasks[ii].max()>0:
                    # pdb.set_trace()
                    loss_body.append(dice_coefficient(densepose_predictor_outputs.coarse_segm[ii,1], gt_bitmasks_body[ii]))
            if loss_body!=[]:
                loss_body = torch.stack(loss_body).mean()
            else:
                loss_body = 0.
            losses["loss_densepose_S_body"] = loss_body * self.w_body
        else:
            loss_mask = dice_coefficient(densepose_predictor_outputs.coarse_segm[valid_idxs], gt_bitmasks[valid_idxs])
            losses["loss_densepose_S"] = loss_mask.mean() * self.w_segm

        if self.use_aux_global_s:
            global_s_gt = [torch.clamp(gt.gt_bitmasks.float().sum(dim=0), 0, 1) for gt in proposals_with_gt]
            global_s_gt = torch.stack(global_s_gt, dim=0)[:,None,...]
            global_s_est = densepose_predictor_outputs.aux_supervision[:,0:1]
            losses["loss_densepose_aux_global_S"] = dice_coefficient(global_s_est, global_s_gt).mean() * self.w_aux_global_s
            
        if self.use_aux_body_semantics:
            # pdb.set_trace()
            body_s_est = densepose_predictor_outputs.aux_supervision[:,0:15] 
            body_s_gt = body_semantics_gt.long() 
            losses["loss_densepose_body_semantics"] = F.cross_entropy(body_s_est, body_s_gt) * self.w_aux_body_semantics
            
            # # remove bg category
            # fg = (body_s_gt>0).float()
            # loss = F.cross_entropy(body_s_est, body_s_gt, reduction='none')
            # loss = (loss*fg).sum()/fg.sum()
            # losses["loss_densepose_body_semantics"] = loss * self.w_aux_body_semantics


        return losses


    def produce_densepose_losses_uv(
        self,
        proposals_with_gt: List[Instances],
        densepose_predictor_outputs: Any,
        packed_annotations: Any,
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
        u_gt = packed_annotations.u_gt[j_valid_fg]
        u_est = interpolator.extract_at_points(densepose_predictor_outputs.u)[j_valid_fg]
        v_gt = packed_annotations.v_gt[j_valid_fg]
        v_est = interpolator.extract_at_points(densepose_predictor_outputs.v)[j_valid_fg]
        if self.use_mean_uv:
            return {
                "loss_densepose_U": F.smooth_l1_loss(u_est, u_gt, reduction="mean") * self.w_points,
                "loss_densepose_V": F.smooth_l1_loss(v_est, v_gt, reduction="mean") * self.w_points,
            }
        else:
            return {
                "loss_densepose_U": F.smooth_l1_loss(u_est, u_gt, reduction="sum") * self.w_points,
                "loss_densepose_V": F.smooth_l1_loss(v_est, v_gt, reduction="sum") * self.w_points,
            }
