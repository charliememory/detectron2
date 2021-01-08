# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Any, List
import torch, pdb, copy
from torch.nn import functional as F

from detectron2.config import CfgNode
from detectron2.structures import Instances

from densepose.utils.comm import compute_locations, compute_grid, aligned_bilinear
from .. import DensePoseConfidenceModelConfig, DensePoseUVConfidenceType
from .chart import DensePoseChartLoss
# from .chart_global_IUV_seperated_S import DensePoseChartGlobalIUVSeparatedSLoss
from .registry import DENSEPOSE_LOSS_REGISTRY
# from .utils import BilinearInterpolationHelper, LossDict #, SingleTensorsHelper
# from .utils import dice_coefficient, FocalLoss, focal_l2_loss

from .utils import (
    BilinearInterpolationHelper,
    ChartBasedAnnotationsAccumulator,
    LossDict,
    extract_packed_annotations_from_matches,
    dice_coefficient, FocalLoss,
)

# def dice_coefficient(x, target):
#     eps = 1e-5
#     n_inst = x.size(0)
#     x = x.reshape(n_inst, -1)
#     target = target.reshape(n_inst, -1)
#     intersection = (x * target).sum(dim=1)
#     union = (x ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1) + eps
#     loss = 1. - (2 * intersection / union)
#     return loss


@DENSEPOSE_LOSS_REGISTRY.register()
class DensePoseChartGlobalIUVSeparatedSPoolerLoss(DensePoseChartLoss):
    def __call__(
        self, proposals_with_gt: List[Instances], densepose_predictor_outputs: Any,
        gt_bitmasks: Any, images=None, skeleton_feats_gt=None
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
        # if self.use_teacher_student:
        #     with torch.no_grad():
        #         teacher_features = self.teacher_model.backbone(images.tensor)
        #         teacher_instances = copy.copy(proposals_with_gt)
        #         for i in range(len(teacher_instances)):
        #             teacher_instances[i].set('pred_boxes', teacher_instances[i].get('gt_boxes').clone())
        #             teacher_instances[i].set('pred_classes', teacher_instances[i].get('gt_classes').clone())
        #         self.teacher_instances = self.teacher_model.roi_heads.forward_with_given_boxes(teacher_features, teacher_instances)

        #     for i in range(len(proposals_with_gt)):
        #         for j in range(len(proposals_with_gt[i].gt_densepose)):
        #             if proposals_with_gt[i].gt_densepose[j] is not None:
        #                 x = proposals_with_gt[i].gt_densepose[j].x
        #                 y = proposals_with_gt[i].gt_densepose[j].y
        #                 x = torch.clamp(x/256.*self.heatmap_size, 0, self.heatmap_size-2)
        #                 y = torch.clamp(y/256.*self.heatmap_size, 0, self.heatmap_size-2)
        #                 self.teacher_instances[i].gt_densepose[j].segm = F.interpolate(self.teacher_instances[i].gt_densepose[j].segm[None,None,...], 
        #                                                 [self.heatmap_size,self.heatmap_size], mode="nearest")[0,0]

        #                 # import imageio
        #                 # imageio.imwrite("output/tmp/segm.png", self.teacher_instances[i].gt_densepose[j].segm.detach().cpu().numpy())
        #                 # imageio.imwrite("output/tmp/segm_pred.png", self.teacher_instances[i].pred_densepose[j].coarse_segm[0,1].detach().cpu().numpy())
        #                 self.teacher_instances[i].gt_densepose[j].segm[y.long(),x.long()] = 0
        #                 self.teacher_instances[i].gt_densepose[j].segm[y.long(),x.long()+1] = 0
        #                 # pdb.set_trace()
        #                 self.teacher_instances[i].gt_densepose[j].segm[y.long()+1,x.long()] = 0
        #                 self.teacher_instances[i].gt_densepose[j].segm[y.long()+1,x.long()+1] = 0
        #                 # imageio.imwrite("output/tmp/segm2.png", self.teacher_instances[i].gt_densepose[j].segm.detach().cpu().numpy())
        #                 # pdb.set_trace()

        #         "TODO"
        #         if self.teach_ins_wo_gt_dp:
        #             pass

        # # if not self.segm_trained_by_masks:
        # return self.produce_densepose_losses(proposals_with_gt, densepose_predictor_outputs, 
        # 	gt_bitmasks, skeleton_feats_gt)
        # # else:
        # #     losses_densepose = self.produce_densepose_losses(
        # #         proposals_with_gt, densepose_predictor_outputs, gt_bitmasks
        # #     )
        # #     losses_mask = self.produce_mask_losses(proposals_with_gt, densepose_predictor_outputs)
        # #     return {**losses_densepose, **losses_mask}




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

# <<<<<<< HEAD
# =======
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
        )

        # return {**losses_uv, **losses_segm}
        if self.use_aux_global_skeleton:
            losses_skeleton = self.produce_densepose_losses_skeleton(
                proposals_with_gt, densepose_predictor_outputs, skeleton_feats_gt
            )
        else:
            losses_skeleton = {}

        return {**losses_uv, **losses_segm, **losses_skeleton}






    # def produce_densepose_losses(
    #     self, proposals_with_gt: List[Instances], densepose_predictor_outputs: Any,
    #     gt_bitmasks: Any, skeleton_feats_gt: Any
    # ) -> LossDict:
    #     """
    #     Losses for segmentation and U/V coordinates computed as cross-entropy
    #     for segmentation unnormalized scores given ground truth labels at
    #     annotated points and smooth L1 loss for U and V coordinate estimates at
    #     annotated points.

    #     Args:
    #         proposals_with_gt (list of Instances): detections with associated ground truth data
    #         densepose_predictor_outputs: DensePose predictor outputs, an object
    #             of a dataclass that is assumed to have the following attributes:
    #          * fine_segm - fine segmentation estimates, tensor of shape [N, C, S, S]
    #          * u - U coordinate estimates per fine labels, tensor of shape [N, C, S, S]
    #          * v - V coordinate estimates per fine labels, tensor of shape [N, C, S, S]
    #     Return:
    #         dict: str -> tensor: dict of losses with the following entries:
    #          * `loss_densepose_U`: smooth L1 loss for U coordinate estimates
    #          * `loss_densepose_V`: smooth L1 loss for V coordinate estimates
    #          * `loss_densepose_I`: cross entropy for raw unnormalized scores for fine
    #              segmentation estimates given ground truth labels
    #          * `loss_densepose_S`: cross entropy for raw unnormalized scores for coarse
    #              segmentation estimates given ground truth labels;
    #              may be included if coarse segmentation is only trained
    #              using DensePose ground truth; if additional supervision through
    #              instance segmentation data is performed (`segm_trained_by_masks` is True),
    #              this loss is handled by `produce_mask_losses` instead
    #     """
    #     # densepose outputs are computed for all images and all bounding boxes;
    #     # i.e. if a batch has 4 images with (3, 1, 2, 1) proposals respectively,
    #     # the outputs will have size(0) == 3+1+2+1 == 7
    #     densepose_outputs_size = densepose_predictor_outputs.u.size()

    #     if not len(proposals_with_gt):
    #         return self.produce_fake_densepose_losses(densepose_predictor_outputs)

    #     tensors_helper = SingleTensorsHelper(proposals_with_gt)
    #     n_batch = len(tensors_helper.index_with_dp)

    #     # NOTE: we need to keep the same computation graph on all the GPUs to
    #     # perform reduction properly. Hence even if we have no data on one
    #     # of the GPUs, we still need to generate the computation graph.
    #     # Add fake (zero) loss in the form Tensor.sum() * 0
    #     if not n_batch:
    #         return self.produce_fake_densepose_losses(densepose_predictor_outputs)

    #     interpolator = BilinearInterpolationHelper.from_matches(
    #         tensors_helper, densepose_outputs_size
    #     )

    #     j_valid_fg = interpolator.j_valid * (tensors_helper.fine_segm_labels_gt > 0)

    #     losses_uv = self.produce_densepose_losses_uv(
    #         proposals_with_gt, densepose_predictor_outputs, tensors_helper, interpolator, j_valid_fg
    #     )

    #     losses_segm = self.produce_densepose_losses_segm(
    #         proposals_with_gt, densepose_predictor_outputs, tensors_helper, interpolator, j_valid_fg,
    #         gt_bitmasks
    #     )

    #     if self.use_aux_global_skeleton:
    #         losses_skeleton = self.produce_densepose_losses_skeleton(
    #             proposals_with_gt, densepose_predictor_outputs, skeleton_feats_gt
    #         )
    #     else:
    #         losses_skeleton = {}

    #     return {**losses_uv, **losses_segm, **losses_skeleton}

                        
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


    # def produce_densepose_losses_segm(
    #     self,
    #     proposals_with_gt: List[Instances],
    #     densepose_predictor_outputs: Any,
    #     tensors_helper: SingleTensorsHelper,
    #     interpolator: BilinearInterpolationHelper,
    #     j_valid_fg: torch.Tensor,
    #     gt_bitmasks: Any,
    # ) -> LossDict:
    #     """
    #     Losses for fine / coarse segmentation: cross-entropy
    #     for segmentation unnormalized scores given ground truth labels at
    #     annotated points for fine segmentation and dense mask annotations
    #     for coarse segmentation.

    #     Args:
    #         proposals_with_gt (list of Instances): detections with associated ground truth data
    #         densepose_predictor_outputs: DensePose predictor outputs, an object
    #             of a dataclass that is assumed to have the following attributes:
    #          * fine_segm - fine segmentation estimates, tensor of shape [N, C, S, S]
    #          * coarse_segm - coarse segmentation estimates, tensor of shape [N, D, S, S]
    #     Return:
    #         dict: str -> tensor: dict of losses with the following entries:
    #          * `loss_densepose_I`: cross entropy for raw unnormalized scores for fine
    #              segmentation estimates given ground truth labels
    #          * `loss_densepose_S`: cross entropy for raw unnormalized scores for coarse
    #              segmentation estimates given ground truth labels;
    #              may be included if coarse segmentation is only trained
    #              using DensePose ground truth; if additional supervision through
    #              instance segmentation data is performed (`segm_trained_by_masks` is True),
    #              this loss is handled by `produce_mask_losses` instead
    #     """
    #     losses = {}

    #     if self.use_teacher_student:
    #         loss_fine_segm_teach = 0.
    #         m_sum = 0.
    #         cnt = 0
    #         for i in range(len(proposals_with_gt)):
    #             for j in range(len(proposals_with_gt[i].gt_densepose)):
    #                 if proposals_with_gt[i].gt_densepose[j] is not None:
    #                     m = (self.teacher_instances[i].gt_densepose[j].segm>0).float()[None,...]
    #                     # pdb.set_trace()
    #                     fine_segm_gt = torch.argmax(self.teacher_instances[i].pred_densepose[j].fine_segm, dim=1)
    #                     fine_segm_est = densepose_predictor_outputs.fine_segm[cnt:cnt+1]
    #                     m_sum += m.sum()
    #                     # if self.use_mean_uv:
    #                     #     loss_u_teach += (F.smooth_l1_loss(u_est, u_gt, reduction="none") * m).sum()/m.sum() * self.w_points_teach
    #                     #     loss_v_teach += (F.smooth_l1_loss(v_est, v_gt, reduction="none") * m).sum()/m.sum() * self.w_points_teach
    #                     # else:

    #                     if self.use_part_focal_loss:
    #                         l = self.focal_loss(fine_segm_est, fine_segm_gt.long())
    #                     else:
    #                         l = F.cross_entropy(fine_segm_est, fine_segm_gt.long(), reduction="none")
    #                     # pdb.set_trace()
    #                     loss_fine_segm_teach += (l * m).sum()
    #                 cnt += 1
    #         losses["loss_densepose_I_teach"] = loss_fine_segm_teach/m_sum * self.w_part_teach

    #     fine_segm_gt = tensors_helper.fine_segm_labels_gt[interpolator.j_valid]
    #     fine_segm_est = interpolator.extract_at_points(
    #         densepose_predictor_outputs.fine_segm[tensors_helper.index_with_dp],
    #         slice_fine_segm=slice(None),
    #         w_ylo_xlo=interpolator.w_ylo_xlo[:, None],
    #         w_ylo_xhi=interpolator.w_ylo_xhi[:, None],
    #         w_yhi_xlo=interpolator.w_yhi_xlo[:, None],
    #         w_yhi_xhi=interpolator.w_yhi_xhi[:, None],
    #     )[interpolator.j_valid, :]
    #     if self.use_part_focal_loss:
    #         # pdb.set_trace()
    #         losses["loss_densepose_I"] = self.focal_loss(fine_segm_est, fine_segm_gt.long()).mean() * self.w_part
    #     else:
    #         losses["loss_densepose_I"] = F.cross_entropy(fine_segm_est, fine_segm_gt.long()) * self.w_part
    #     fine_segm_est = torch.argmax(fine_segm_est,dim=1)

    #     "TODO: add global S, skeleton_feat loss"
    #     assert self.n_segm_chan==1
    #     # pdb.set_trace()
    #     valid_idxs = gt_bitmasks.abs().mean([1,2,3])>0
    #     loss_mask = dice_coefficient(densepose_predictor_outputs.coarse_segm[valid_idxs], gt_bitmasks[valid_idxs])
    #     losses["loss_densepose_S"] = loss_mask.mean() * self.w_segm

    #     if self.use_aux_global_s:
    #         global_s_gt = [torch.clamp(gt.gt_bitmasks.float().sum(dim=0), 0, 1) for gt in proposals_with_gt]
    #         global_s_gt = torch.stack(global_s_gt, dim=0)[:,None,...]
    #         global_s_est = densepose_predictor_outputs.aux_supervision[:,0:1]

    #         losses["loss_densepose_aux_global_S"] = dice_coefficient(global_s_est, global_s_gt).mean() * self.w_aux_global_s
            
    #     return losses

    def produce_densepose_losses_segm(
        self,
        proposals_with_gt: List[Instances],
        densepose_predictor_outputs: Any,
        packed_annotations: Any,
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
        # pdb.set_trace()
        valid_idxs = gt_bitmasks.abs().mean([1,2,3])>0
        loss_mask = dice_coefficient(densepose_predictor_outputs.coarse_segm[valid_idxs], gt_bitmasks[valid_idxs])
        losses["loss_densepose_S"] = loss_mask.mean() * self.w_segm

        if self.use_aux_global_s:
            global_s_gt = [torch.clamp(gt.gt_bitmasks.float().sum(dim=0), 0, 1) for gt in proposals_with_gt]
            global_s_gt = torch.stack(global_s_gt, dim=0)[:,None,...]
            global_s_est = densepose_predictor_outputs.aux_supervision[:,0:1]
            losses["loss_densepose_aux_global_S"] = dice_coefficient(global_s_est, global_s_gt).mean() * self.w_aux_global_s
            
        return losses


    # "TODO: 2) linear interpolate UV labels"
    # def produce_densepose_losses_uv(
    #     self,
    #     proposals_with_gt: List[Instances],
    #     densepose_predictor_outputs: Any,
    #     tensors_helper: SingleTensorsHelper,
    #     interpolator: BilinearInterpolationHelper,
    #     j_valid_fg: torch.Tensor,
    # ) -> LossDict:
    #     """
    #     Compute losses for U/V coordinates: smooth L1 loss between
    #     estimated coordinates and the ground truth.

    #     Args:
    #         proposals_with_gt (list of Instances): detections with associated ground truth data
    #         densepose_predictor_outputs: DensePose predictor outputs, an object
    #             of a dataclass that is assumed to have the following attributes:
    #          * u - U coordinate estimates per fine labels, tensor of shape [N, C, S, S]
    #          * v - V coordinate estimates per fine labels, tensor of shape [N, C, S, S]
    #     Return:
    #         dict: str -> tensor: dict of losses with the following entries:
    #          * `loss_densepose_U`: smooth L1 loss for U coordinate estimates
    #          * `loss_densepose_V`: smooth L1 loss for V coordinate estimates
    #     """
    #     # pdb.set_trace()
    #     # if not j_valid_fg.any():
    #         # pdb.set_trace()

    #     u_gt = tensors_helper.u_gt[j_valid_fg]
    #     u_est = interpolator.extract_at_points(
    #         densepose_predictor_outputs.u[tensors_helper.index_with_dp]
    #     )[j_valid_fg]

    #     v_gt = tensors_helper.v_gt[j_valid_fg]
    #     v_est = interpolator.extract_at_points(
    #         densepose_predictor_outputs.v[tensors_helper.index_with_dp]
    #     )[j_valid_fg]

    #     if self.use_teacher_student:
    #         loss_u_teach = 0.
    #         loss_v_teach = 0.
    #         m_sum = 0.
    #         cnt = 0
    #         for i in range(len(proposals_with_gt)):
    #             for j in range(len(proposals_with_gt[i].gt_densepose)):
    #                 if proposals_with_gt[i].gt_densepose[j] is not None:
    #                     m = (self.teacher_instances[i].gt_densepose[j].segm>0).float()[None,None,...]
    #                     u_gt = self.teacher_instances[i].pred_densepose[j].u
    #                     u_est = densepose_predictor_outputs.u[cnt:cnt+1]
    #                     v_gt = self.teacher_instances[i].pred_densepose[j].v
    #                     v_est = densepose_predictor_outputs.v[cnt:cnt+1]
    #                     m_sum += m.sum()
    #                     # if self.use_mean_uv:
    #                     #     loss_u_teach += (F.smooth_l1_loss(u_est, u_gt, reduction="none") * m).sum()/m.sum() * self.w_points_teach
    #                     #     loss_v_teach += (F.smooth_l1_loss(v_est, v_gt, reduction="none") * m).sum()/m.sum() * self.w_points_teach
    #                     # else:
    #                     # pdb.set_trace()
    #                     l_u = F.smooth_l1_loss(u_est, u_gt, reduction="none").mean(dim=1,keepdim=True)
    #                     loss_u_teach += (l_u * m).sum() * self.w_points_teach
    #                     l_v = F.smooth_l1_loss(v_est, v_gt, reduction="none").mean(dim=1,keepdim=True)
    #                     loss_v_teach += (l_v * m).sum() * self.w_points_teach
    #                 cnt += 1
    #         if self.use_mean_uv:
    #             return {
    #                 "loss_densepose_U": F.smooth_l1_loss(u_est, u_gt, reduction="mean") * self.w_points,
    #                 "loss_densepose_V": F.smooth_l1_loss(v_est, v_gt, reduction="mean") * self.w_points,
    #                 "loss_densepose_U_teach": loss_u_teach/m_sum,
    #                 "loss_densepose_V_teach": loss_v_teach/m_sum,
    #             }
    #         else:
    #             return {
    #                 "loss_densepose_U": F.smooth_l1_loss(u_est, u_gt, reduction="sum") * self.w_points,
    #                 "loss_densepose_V": F.smooth_l1_loss(v_est, v_gt, reduction="sum") * self.w_points,
    #                 "loss_densepose_U_teach": loss_u_teach,
    #                 "loss_densepose_V_teach": loss_v_teach,
    #             }
    #     else:
    #         if self.use_mean_uv:
    #             return {
    #                 "loss_densepose_U": F.smooth_l1_loss(u_est, u_gt, reduction="mean") * self.w_points,
    #                 "loss_densepose_V": F.smooth_l1_loss(v_est, v_gt, reduction="mean") * self.w_points,
    #             }
    #         else:
    #             return {
    #                 "loss_densepose_U": F.smooth_l1_loss(u_est, u_gt, reduction="sum") * self.w_points,
    #                 "loss_densepose_V": F.smooth_l1_loss(v_est, v_gt, reduction="sum") * self.w_points,
    #             }

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
