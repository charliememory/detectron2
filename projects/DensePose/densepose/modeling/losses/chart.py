# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Any, List
import torch
from torch.nn import functional as F

from detectron2.config import CfgNode
from detectron2.structures import Instances

# <<<<<<< HEAD

# from .registry import DENSEPOSE_LOSS_REGISTRY
# from .utils import BilinearInterpolationHelper, LossDict, SingleTensorsHelper, resample_data
# from .utils import dice_coefficient, FocalLoss


# @dataclass
# class DataForMaskLoss:
#     """
#     Contains mask GT and estimated data for proposals from multiple images:
#     """

#     # tensor of size (K, H, W) containing GT labels
#     masks_gt: Optional[torch.Tensor] = None
#     # tensor of size (K, C, H, W) containing estimated scores
#     masks_est: Optional[torch.Tensor] = None


# def extract_data_for_mask_loss_from_matches(
#     proposals_targets: Iterable[Instances], estimated_segm: torch.Tensor
# ) -> DataForMaskLoss:
#     """
#     Extract data for mask loss from instances that contain matched GT and
#     estimated bounding boxes.
#     Args:
#         proposals_targets: Iterable[Instances]
#             matched GT and estimated results, each item in the iterable
#             corresponds to data in 1 image
#         estimated_segm: tensor(K, C, S, S) of float - raw unnormalized
#             segmentation scores, here S is the size to which GT masks are
#             to be resized
#     Return:
#         masks_est: tensor(K, C, S, S) of float - class scores
#         masks_gt: tensor(K, S, S) of int64 - labels
#     """
#     data = DataForMaskLoss()
#     masks_gt = []
#     offset = 0
#     assert estimated_segm.shape[2] == estimated_segm.shape[3], (
#         f"Expected estimated segmentation to have a square shape, "
#         f"but the actual shape is {estimated_segm.shape[2:]}"
#     )
#     mask_size = estimated_segm.shape[2]
#     num_proposals = sum(inst.proposal_boxes.tensor.size(0) for inst in proposals_targets)
#     num_estimated = estimated_segm.shape[0]
#     assert (
#         num_proposals == num_estimated
#     ), "The number of proposals {} must be equal to the number of estimates {}".format(
#         num_proposals, num_estimated
#     )

#     for proposals_targets_per_image in proposals_targets:
#         n_i = proposals_targets_per_image.proposal_boxes.tensor.size(0)
#         if not n_i:
#             continue
#         gt_masks_per_image = proposals_targets_per_image.gt_masks.crop_and_resize(
#             proposals_targets_per_image.proposal_boxes.tensor, mask_size
#         ).to(device=estimated_segm.device)
#         masks_gt.append(gt_masks_per_image)
#         offset += n_i

#     if masks_gt:
#         data.masks_est = estimated_segm
#         data.masks_gt = torch.cat(masks_gt, dim=0)
#     return data
# =======
from .mask_or_segm import MaskOrSegmentationLoss
from .registry import DENSEPOSE_LOSS_REGISTRY
from .utils import (
    BilinearInterpolationHelper,
    ChartBasedAnnotationsAccumulator,
    LossDict,
    extract_packed_annotations_from_matches,
    dice_coefficient, FocalLoss,
)
# >>>>>>> upstream/master


@DENSEPOSE_LOSS_REGISTRY.register()
class DensePoseChartLoss:
    """
    DensePose loss for chart-based training. A mesh is split into charts,
    each chart is given a label (I) and parametrized by 2 coordinates referred to
    as U and V. Ground truth consists of a number of points annotated with
    I, U and V values and coarse segmentation S defined for all pixels of the
    object bounding box. In some cases (see `COARSE_SEGM_TRAINED_BY_MASKS`),
    semantic segmentation annotations can be used as ground truth inputs as well.

    Estimated values are tensors:
     * U coordinates, tensor of shape [N, C, S, S]
     * V coordinates, tensor of shape [N, C, S, S]
     * fine segmentation estimates, tensor of shape [N, C, S, S] with raw unnormalized
       scores for each fine segmentation label at each location
     * coarse segmentation estimates, tensor of shape [N, D, S, S] with raw unnormalized
       scores for each coarse segmentation label at each location
    where N is the number of detections, C is the number of fine segmentation
    labels, S is the estimate size ( = width = height) and D is the number of
    coarse segmentation channels.

    The losses are:
    * regression (smooth L1) loss for U and V coordinates
    * cross entropy loss for fine (I) and coarse (S) segmentations
    Each loss has an associated weight
    """

    def __init__(self, cfg: CfgNode):
        """
        Initialize chart-based loss from configuration options

        Args:
            cfg (CfgNode): configuration options
        """
        # fmt: off
        self.heatmap_size = cfg.MODEL.ROI_DENSEPOSE_HEAD.HEATMAP_SIZE
        self.w_points     = cfg.MODEL.ROI_DENSEPOSE_HEAD.POINT_REGRESSION_WEIGHTS
        self.w_part       = cfg.MODEL.ROI_DENSEPOSE_HEAD.PART_WEIGHTS
        self.use_part_focal_loss = cfg.MODEL.ROI_DENSEPOSE_HEAD.PART_FOCAL_LOSS
        self.w_segm       = cfg.MODEL.ROI_DENSEPOSE_HEAD.INDEX_WEIGHTS
        self.w_body       = cfg.MODEL.ROI_DENSEPOSE_HEAD.BODY_WEIGHTS
        self.n_segm_chan  = cfg.MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS
        self.w_smooth       = cfg.MODEL.ROI_DENSEPOSE_HEAD.SMOOTH_WEIGHTS
        self.w_tv      = cfg.MODEL.ROI_DENSEPOSE_HEAD.TV_WEIGHTS
        # fmt: on
        self.segm_trained_by_masks = cfg.MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS
# <<<<<<< HEAD
        self.use_mean_uv = cfg.MODEL.ROI_DENSEPOSE_HEAD.MEAN_UV_LOSS

        if self.use_part_focal_loss:
            gamma = cfg.MODEL.ROI_DENSEPOSE_HEAD.PART_FOCAL_GAMMA
            self.focal_loss = FocalLoss(gamma=gamma, alpha=None, size_average=False)

        self.use_teacher_student = cfg.MODEL.TEACHER_STUDENT
        self.teacher_cfg = cfg.MODEL.TEACHER_CFG_FILE
        self.teacher_weights = cfg.MODEL.TEACHER_WEIGHTS
        self.teach_ins_wo_gt_dp = cfg.MODEL.TEACH_INS_WO_GT_DP
        self.w_part_teach     = cfg.MODEL.TEACH_PART_WEIGHTS
        self.w_points_teach     = cfg.MODEL.TEACH_POINT_REGRESSION_WEIGHTS
        if self.use_teacher_student:
            from densepose.engine import Trainer
            from densepose.modeling.densepose_checkpoint import DensePoseCheckpointer
            from densepose.config import get_cfg, add_densepose_config
            self.teacher_cfg = get_cfg()
            add_densepose_config(self.teacher_cfg)
            self.teacher_cfg.merge_from_file(cfg.MODEL.TEACHER_CFG_FILE)
            self.teacher_model = Trainer.build_model(self.teacher_cfg)
            # pdb.set_trace()
            DensePoseCheckpointer(self.teacher_model).load(cfg.MODEL.TEACHER_WEIGHTS)
            self.teacher_model.eval()

        self.use_aux_global_s = cfg.MODEL.CONDINST.AUX_SUPERVISION_GLOBAL_S
        self.use_aux_global_skeleton = cfg.MODEL.CONDINST.AUX_SUPERVISION_GLOBAL_SKELETON
        self.use_aux_body_semantics = cfg.MODEL.CONDINST.AUX_SUPERVISION_BODY_SEMANTICS
        self.w_aux_global_s = cfg.MODEL.CONDINST.AUX_SUPERVISION_GLOBAL_S_WEIGHTS
        self.w_aux_global_skeleton = cfg.MODEL.CONDINST.AUX_SUPERVISION_GLOBAL_SKELETON_WEIGHTS
        self.w_aux_body_semantics = cfg.MODEL.CONDINST.AUX_SUPERVISION_BODY_SEMANTICS
        
        self.pred_ins_body = cfg.MODEL.CONDINST.PREDICT_INSTANCE_BODY

#     def __call__(
#         self, proposals_with_gt: List[Instances], densepose_predictor_outputs: Any, images=None
# =======
        self.segm_loss = MaskOrSegmentationLoss(cfg)

    def __call__(
        self, proposals_with_gt: List[Instances], densepose_predictor_outputs: Any, **kwargs
# >>>>>>> upstream/master
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
            dict: str -> tensor: dict of losses with the following entries:
             * `loss_densepose_U`: smooth L1 loss for U coordinate estimates
             * `loss_densepose_V`: smooth L1 loss for V coordinate estimates
             * `loss_densepose_I`: cross entropy for raw unnormalized scores for fine
                 segmentation estimates given ground truth labels;
             * `loss_densepose_S`: cross entropy for raw unnormalized scores for coarse
                 segmentation estimates given ground truth labels;
        """
# <<<<<<< HEAD
#         # if self.use_teacher_student:
#         #     with torch.no_grad():
#         #         pdb.set_trace()
#         #         outputs = self.teacher_model(inputs)

#         if not self.segm_trained_by_masks:
#             return self.produce_densepose_losses(proposals_with_gt, densepose_predictor_outputs)
#         else:
#             losses_densepose = self.produce_densepose_losses(
#                 proposals_with_gt, densepose_predictor_outputs
#             )
#             losses_mask = self.produce_mask_losses(proposals_with_gt, densepose_predictor_outputs)
#             return {**losses_densepose, **losses_mask}
# =======
        # densepose outputs are computed for all images and all bounding boxes;
        # i.e. if a batch has 4 images with (3, 1, 2, 1) proposals respectively,
        # the outputs will have size(0) == 3+1+2+1 == 7
# >>>>>>> upstream/master

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
        )

        return {**losses_uv, **losses_segm}
# >>>>>>> upstream/master

    def produce_fake_mask_losses(self, densepose_predictor_outputs: Any) -> LossDict:
        """
        Fake coarse segmentation loss used when no suitable ground truth data
        was found in a batch. The loss has a value 0 and is primarily used to
        construct the computation graph, so that `DistributedDataParallel`
        has similar graphs on all GPUs and can perform reduction properly.
        Args:
            densepose_predictor_outputs: DensePose predictor outputs, an object
                of a dataclass that is assumed to have `coarse_segm`
                attribute
        Return:
            dict: str -> tensor: dict of losses with a single entry
                `loss_densepose_S` with 0 value
        """
        return {"loss_densepose_S": densepose_predictor_outputs.coarse_segm.sum() * 0}

    def produce_mask_losses(
        self, proposals_with_gt: List[Instances], densepose_predictor_outputs: Any
    ) -> LossDict:
        """
        Computes coarse segmentation loss as cross-entropy for raw unnormalized
        scores given ground truth labels.
        Args:
            proposals_with_gt (list of Instances): detections with associated ground truth data
            densepose_predictor_outputs: an object of a dataclass that contains predictor outputs
                with estimated values; assumed to have the following attribute:
                * coarse_segm (tensor of shape [N, D, S, S]): coarse segmentation estimates
                    as raw unnormalized scores
            where N is the number of detections, S is the estimate size ( = width = height) and
            D is the number of coarse segmentation channels.
        Return:
            dict: str -> tensor: dict of losses with a single entry:
            * `loss_densepose_S`: cross entropy for raw unnormalized scores for coarse
                segmentation given ground truth labels
        """
        if not len(proposals_with_gt):
            return self.produce_fake_mask_losses(densepose_predictor_outputs)
        # densepose outputs are computed for all images and all bounding boxes;
        # i.e. if a batch has 4 images with (3, 1, 2, 1) proposals respectively,
        # the outputs will have size(0) == 3+1+2+1 == 7
        with torch.no_grad():
            mask_loss_data = extract_data_for_mask_loss_from_matches(
                proposals_with_gt, densepose_predictor_outputs.coarse_segm
            )
        if (mask_loss_data.masks_gt is None) or (mask_loss_data.masks_est is None):
            return self.produce_fake_mask_losses(densepose_predictor_outputs)
        return {
            "loss_densepose_S": F.cross_entropy(
                mask_loss_data.masks_est, mask_loss_data.masks_gt.long()
            )
            * self.w_segm
        }

    def produce_fake_densepose_losses(self, densepose_predictor_outputs: Any) -> LossDict:
        """
        Fake losses for fine segmentation and U/V coordinates. These are used when
        no suitable ground truth data was found in a batch. The loss has a value 0
        and is primarily used to construct the computation graph, so that
        `DistributedDataParallel` has similar graphs on all GPUs and can perform
        reduction properly.

        Args:
            densepose_predictor_outputs: DensePose predictor outputs, an object
                of a dataclass that is assumed to have the following attributes:
             * fine_segm - fine segmentation estimates, tensor of shape [N, C, S, S]
             * u - U coordinate estimates per fine labels, tensor of shape [N, C, S, S]
             * v - V coordinate estimates per fine labels, tensor of shape [N, C, S, S]
        Return:
            dict: str -> tensor: dict of losses with the following entries:
             * `loss_densepose_U`: has value 0
             * `loss_densepose_V`: has value 0
             * `loss_densepose_I`: has value 0
             * `loss_densepose_S`: has value 0
        """
        losses_uv = self.produce_fake_densepose_losses_uv(densepose_predictor_outputs)
        losses_segm = self.produce_fake_densepose_losses_segm(densepose_predictor_outputs)
        return {**losses_uv, **losses_segm}

    def produce_fake_densepose_losses_uv(self, densepose_predictor_outputs: Any) -> LossDict:
        """
        Fake losses for U/V coordinates. These are used when no suitable ground
        truth data was found in a batch. The loss has a value 0
        and is primarily used to construct the computation graph, so that
        `DistributedDataParallel` has similar graphs on all GPUs and can perform
        reduction properly.

        Args:
            densepose_predictor_outputs: DensePose predictor outputs, an object
                of a dataclass that is assumed to have the following attributes:
             * u - U coordinate estimates per fine labels, tensor of shape [N, C, S, S]
             * v - V coordinate estimates per fine labels, tensor of shape [N, C, S, S]
        Return:
            dict: str -> tensor: dict of losses with the following entries:
             * `loss_densepose_U`: has value 0
             * `loss_densepose_V`: has value 0
        """
        return {
            "loss_densepose_U": densepose_predictor_outputs.u.sum() * 0,
            "loss_densepose_V": densepose_predictor_outputs.v.sum() * 0,
        }

    def produce_fake_densepose_losses_segm(self, densepose_predictor_outputs: Any) -> LossDict:
        """
        Fake losses for fine / coarse segmentation. These are used when
        no suitable ground truth data was found in a batch. The loss has a value 0
        and is primarily used to construct the computation graph, so that
        `DistributedDataParallel` has similar graphs on all GPUs and can perform
        reduction properly.

        Args:
            densepose_predictor_outputs: DensePose predictor outputs, an object
                of a dataclass that is assumed to have the following attributes:
             * fine_segm - fine segmentation estimates, tensor of shape [N, C, S, S]
             * coarse_segm - coarse segmentation estimates, tensor of shape [N, D, S, S]
        Return:
            dict: str -> tensor: dict of losses with the following entries:
             * `loss_densepose_I`: has value 0
             * `loss_densepose_S`: has value 0, added only if `segm_trained_by_masks` is False
        """
        losses = {
            "loss_densepose_I": densepose_predictor_outputs.fine_segm.sum() * 0,
            "loss_densepose_S": self.segm_loss.fake_value(densepose_predictor_outputs),
        }
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
        return {
            "loss_densepose_U": F.smooth_l1_loss(u_est, u_gt, reduction="sum") * self.w_points,
            "loss_densepose_V": F.smooth_l1_loss(v_est, v_gt, reduction="sum") * self.w_points,
        }

    def produce_densepose_losses_segm(
        self,
        proposals_with_gt: List[Instances],
        densepose_predictor_outputs: Any,
        packed_annotations: Any,
        interpolator: BilinearInterpolationHelper,
        j_valid_fg: torch.Tensor,
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
        return {
            "loss_densepose_I": F.cross_entropy(fine_segm_est, fine_segm_gt.long()) * self.w_part,
            "loss_densepose_S": self.segm_loss(
                proposals_with_gt, densepose_predictor_outputs, packed_annotations
            )
            * self.w_segm,
        }
