# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Any, List
import torch, pdb
from torch.nn import functional as F

from detectron2.config import CfgNode
from detectron2.structures import Instances

from densepose.utils.comm import compute_locations, compute_grid, aligned_bilinear
from .. import DensePoseConfidenceModelConfig, DensePoseUVConfidenceType
from .chart import DensePoseChartLoss
# from .chart_global_IUV_seperated_S import DensePoseChartGlobalIUVSeparatedSLoss
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
class DensePoseChartGlobalIUVSeparatedSCropResizeLoss(DensePoseChartLoss):
    """"""


    def __init__(self, cfg: CfgNode):
        """
        Initialize chart-based loss from configuration options

        Args:
            cfg (CfgNode): configuration options
        """
        # fmt: off
        # self.heatmap_size = cfg.MODEL.ROI_DENSEPOSE_HEAD.HEATMAP_SIZE
        self.w_points     = cfg.MODEL.ROI_DENSEPOSE_HEAD.POINT_REGRESSION_WEIGHTS
        self.w_part       = cfg.MODEL.ROI_DENSEPOSE_HEAD.PART_WEIGHTS
        self.w_segm       = cfg.MODEL.ROI_DENSEPOSE_HEAD.INDEX_WEIGHTS
        self.n_segm_chan  = cfg.MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS
        # fmt: on
        self.segm_trained_by_masks = cfg.MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS

        self.soi = cfg.MODEL.FCOS.SIZES_OF_INTEREST[-1]
        self.w_aux_rel_coord       = cfg.MODEL.ROI_DENSEPOSE_HEAD.AUX_REL_COORDS_WEIGHTS
        self.w_aux_fg_segm       = cfg.MODEL.ROI_DENSEPOSE_HEAD.AUX_FG_SEGM_WEIGHTS
        self.norm_coord_boxHW = cfg.MODEL.CONDINST.IUVHead.NORM_COORD_BOXHW

        self.densepose_size = 256 ## the size of annotation is 256x256

    def __call__(
        self, proposals_with_gt: List[Instances], densepose_predictor_outputs: Any,
        gt_bitmasks: Any,
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
            gt_bitmasks: instance mask ground truth

        Return:
            (dict: str -> tensor): dict of losses with the following entries:
                * loss_densepose_I: fine segmentation loss (cross-entropy)
                * loss_densepose_S: coarse segmentation loss (cross-entropy)
                * loss_densepose_U: loss for U coordinates (smooth L1)
                * loss_densepose_V: loss for V coordinates (smooth L1)
        """
        return self.produce_densepose_losses(proposals_with_gt, densepose_predictor_outputs, gt_bitmasks)

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
             * `loss_densepose_S`: has value 0, added only if `segm_trained_by_masks` is False
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
        losses = {"loss_densepose_I": densepose_predictor_outputs.fine_segm.sum() * 0}
        # if not self.segm_trained_by_masks:
        losses["loss_densepose_S"] = densepose_predictor_outputs.coarse_segm.sum() * 0
        return losses



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

        # losses = {
        #     "loss_densepose_U": F.smooth_l1_loss(u_est, u_gt, reduction="sum") * self.w_points,
        #     "loss_densepose_V": F.smooth_l1_loss(v_est, v_gt, reduction="sum") * self.w_points,
        # }
        losses = {
            "loss_densepose_U": F.smooth_l1_loss(u_est, u_gt, reduction="mean") * self.w_points,
            "loss_densepose_V": F.smooth_l1_loss(v_est, v_gt, reduction="mean") * self.w_points,
        }
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

        "TODO: 1) add dense segm supervision to IUV head; 2) make the supervision instance-aware"
        "Add dense segm supervision to IUV head"
        pred_aux_sup = densepose_predictor_outputs.aux_supervision
        if self.w_aux_rel_coord > 0:
            assert pred_aux_sup.shape[1]>0
            rel_coord_est = pred_aux_sup[:,:2]
            H, W = rel_coord_est.shape[-2:]
            N = len(proposals_with_gt)
            # N = gt_bitmasks.shape[0]
            rel_coord_gt = self._create_rel_coord_gt(proposals_with_gt, H, W, 
                            densepose_predictor_outputs.stride, rel_coord_est.device, 
                            norm_coord_boxHW=self.norm_coord_boxHW)
            # gt_locations_list = []
            # for idx in range(N):
            #     boxes = proposals_with_gt[idx].gt_boxes.tensor.clone()
            #     imgH, imgW = proposals_with_gt[idx].image_size
            #     boxes[:,0] = boxes[:,0]/imgW
            #     boxes[:,1] = boxes[:,1]/imgH
            #     boxes[:,2] = boxes[:,2]/imgW
            #     boxes[:,3] = boxes[:,3]/imgH
            #     gt_locations_list.append(
            #         torch.stack([(boxes[:,0]+boxes[:,2])*0.5, (boxes[:,1]+boxes[:,3])*0.5], dim=1)
            #     )
            # instance_locations = torch.cat(gt_locations_list, dim=0)

            # gt_bitmasks = torch.cat([per_im.gt_bitmasks for per_im in proposals_with_gt])
            
            # # if self.use_rel_coords: 
            # # n_inst = len(instances)

            # # im_inds = instances.im_inds

            # locations = compute_locations(
            #     H, W, 
            #     stride=densepose_predictor_outputs.stride, 
            #     device=rel_coord_est.device,
            #     norm=True,
            # )

            # # instance_locations = instances.locations
            # relative_coords = instance_locations.reshape(-1, 1, 2) - locations.reshape(1, -1, 2)
            # relative_coords = relative_coords.permute(0, 2, 1).float()
            # # relative_coords = relative_coords.float() / self.soi
            # relative_coords = relative_coords.to(dtype=rel_coord_est.dtype)
            # # rel_coord_list = []

            # rel_coord_gt = torch.zeros([N,2,H,W], device=rel_coord_est.device).to(dtype=rel_coord_est.dtype)
            
            # coord_all = relative_coords.reshape(-1, 2, H, W) * gt_bitmasks[:,None,...]
            # cnt = 0
            # for idx in range(N):
            #     # pdb.set_trace()
            #     # proposals_with_gt[idx].num_instances
            #     # if idx in im_inds:
            #     # cc = relative_coords[idx:idx+1].reshape(-1, 2, H, W)
            #     # # assert s_logits.shape[1]==1
            #     # ss = gt_bitmasks[idx:idx+1]
            #     num = proposals_with_gt[idx].gt_bitmasks.shape[0]
            #     # coord = torch.sum(cc*ss, dim=0, keepdim=True) \
            #     #       / (torch.sum(ss, dim=0, keepdim=True)+1e-7)
            #     coord = torch.mean(coord_all[cnt:cnt+num], dim=0, keepdim=True) 
            #     cnt += num
            #     rel_coord_gt[idx:idx+1] = coord #.reshape(1, 2, H, W)
            # pdb.set_trace()
            # import imageio
            # imageio.imwrite('tmp/rel_coord_est.png', rel_coord_est[0,0].detach().cpu().numpy())
            # imageio.imwrite('tmp/rel_coord_gt.png', rel_coord_gt[0,0].detach().cpu().numpy())

            losses["loss_densepose_aux_rel_coord"] = F.smooth_l1_loss(rel_coord_est, rel_coord_gt, reduction="sum")/(rel_coord_gt!=0.).float().sum() \
                                                    * self.w_aux_rel_coord

        if self.w_aux_fg_segm > 0:
            fg_segm_est = pred_aux_sup[:,2:3]
            H, W = fg_segm_est.shape[-2:]
            N = len(proposals_with_gt)
            fg_segm_gt = torch.zeros([N,1,H,W], device=fg_segm_est.device).to(dtype=fg_segm_est.dtype)
            gt_bitmasks = torch.cat([per_im.gt_bitmasks for per_im in proposals_with_gt])
            cnt = 0
            for idx in range(N):
                num = proposals_with_gt[idx].gt_bitmasks.shape[0]
                # coord = torch.mean(gt_bitmasks[cnt:cnt+num].max(dim=0), dim=0, keepdim=True) 
                # pdb.set_trace()
                fg_segm_gt[idx:idx+1] = torch.max(gt_bitmasks[cnt:cnt+num],dim=0)[0][None,None,...]
                cnt += num
            loss_mask = dice_coefficient(fg_segm_est, fg_segm_gt)
            # pdb.set_trace()
            losses["loss_densepose_aux_fg_segm"] = loss_mask.mean() * self.w_aux_fg_segm


            # loss_mask = dice_coefficient(densepose_predictor_outputs.coarse_segm, gt_bitmasks)
            # losses["loss_densepose_S"] = loss_mask.mean() * self.w_segm

        return losses

    def _create_rel_coord_gt(self, gt_instances, H, W, stride, device, norm_coord_boxHW=True):
        
        N = len(gt_instances)
        gt_locations_list = []
        for idx in range(N):
            boxes = gt_instances[idx].gt_boxes.tensor.clone()
            imgH, imgW = gt_instances[idx].image_size
            boxes[:,0] = boxes[:,0]/imgW
            boxes[:,1] = boxes[:,1]/imgH
            boxes[:,2] = boxes[:,2]/imgW
            boxes[:,3] = boxes[:,3]/imgH
            gt_locations_list.append(
                torch.stack([(boxes[:,0]+boxes[:,2])*0.5, (boxes[:,1]+boxes[:,3])*0.5], dim=1)
            )
        instance_locations = torch.cat(gt_locations_list, dim=0)

        gt_bitmasks = torch.cat([per_im.gt_bitmasks for per_im in gt_instances])

        locations = compute_locations(
            H, W, 
            stride=stride, 
            device=device,
            norm=True,
        )

        # instance_locations = instances.locations
        relative_coords = instance_locations.reshape(-1, 1, 2) - locations.reshape(1, -1, 2)
        relative_coords = relative_coords.permute(0, 2, 1).float()
        # relative_coords = relative_coords.to(dtype=rel_coord_est.dtype)
        if norm_coord_boxHW:
            # pdb.set_trace()
            cnt = 0
            for idx in range(N):
                imgH, imgW = gt_instances[idx].image_size
                boxes = gt_instances[idx].gt_boxes.tensor
                for i in range(boxes.shape[0]):
                    boxH, boxW = boxes[i,3]-boxes[i,1], boxes[i,2]-boxes[i,0]
                    relative_coords[cnt+i,0:1] = relative_coords[cnt+i,0:1]*imgW/boxW
                    relative_coords[cnt+i,1:2] = relative_coords[cnt+i,1:2]*imgH/boxH
                cnt += boxes.shape[0]

        rel_coord_gt = torch.zeros([N,2,H,W], device=device).float()
        gt_bitmasks = F.interpolate(gt_bitmasks[:,None,...].float(), size=(H,W), mode='nearest')
        coord_all = relative_coords.reshape(-1, 2, H, W) * gt_bitmasks
        cnt = 0
        for idx in range(N):
            num = gt_instances[idx].gt_bitmasks.shape[0]
            coord = torch.mean(coord_all[cnt:cnt+num], dim=0, keepdim=True) 
            cnt += num
            rel_coord_gt[idx:idx+1] = coord #.reshape(1, 2, H, W)
        return rel_coord_gt



