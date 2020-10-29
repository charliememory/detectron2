# -*- coding: utf-8 -*-
import logging
from typing import Dict, List, Optional

import torch
from torch import nn
import torch.nn.functional as F
import pdb

from detectron2.structures import ImageList, Boxes
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.structures.instances import Instances
from detectron2.structures.masks import PolygonMasks, polygons_to_bitmask

from .dynamic_mask_head import build_dynamic_mask_head
from .mask_branch import build_mask_branch
from .iuv_head import build_iuv_head
from .iuv_deeplab_head import build_iuv_deeplab_head
from .iuv_unet_head import build_iuv_unet_head
from .iuv_multiscale_head import build_iuv_multiscale_head
from .iuv_multilayermask_head import build_iuv_multilayermask_head
from .iuv_multilayercoord_head import build_iuv_multilayercoord_head

# from adet.utils.comm import aligned_bilinear
from densepose.utils.comm import aligned_bilinear

from .. import (
    build_densepose_data_filter,
    build_densepose_head,
    build_densepose_losses,
    build_densepose_predictor,
    densepose_inference,
)

__all__ = ["CondInst"]


logger = logging.getLogger(__name__)


@META_ARCH_REGISTRY.register()
class CondInst(nn.Module):
    """
    Main class for CondInst architectures (see https://arxiv.org/abs/2003.05664).
    """

    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())
        self.mask_head = build_dynamic_mask_head(cfg)
        self.mask_branch = build_mask_branch(cfg, self.backbone.output_shape())
        ##
        if cfg.MODEL.ROI_DENSEPOSE_HEAD.NAME=="DensePoseDeepLabHead":
            self.iuv_head = build_iuv_deeplab_head(cfg)
        elif cfg.MODEL.ROI_DENSEPOSE_HEAD.NAME=="DensePoseUnetHead":
            self.iuv_head = build_iuv_unet_head(cfg)
        elif cfg.MODEL.ROI_DENSEPOSE_HEAD.NAME=="DensePoseMultiscaleHead":
            self.iuv_head = build_iuv_multiscale_head(cfg)
        elif cfg.MODEL.ROI_DENSEPOSE_HEAD.NAME=="DensePoseMultilayermaskHead":
            self.iuv_head = build_iuv_multilayermask_head(cfg)
        elif cfg.MODEL.ROI_DENSEPOSE_HEAD.NAME=="DensePoseMultilayercoordHead":
            self.iuv_head = build_iuv_multilayercoord_head(cfg)
        else:
            self.iuv_head = build_iuv_head(cfg)
        self.iuv_fea_dim = cfg.MODEL.CONDINST.IUVHead.CHANNELS
        self.s_ins_fea_dim = cfg.MODEL.CONDINST.MASK_HEAD.CHANNELS
        self.use_mask_feats_iuvhead = cfg.MODEL.CONDINST.IUVHead.USE_MASK_FEATURES
        self.mask_out_bg_feats = cfg.MODEL.CONDINST.IUVHead.MASK_OUT_BG_FEATURES
        # assert self.iuv_fea_dim+self.s_ins_fea_dim == cfg.MODEL.CONDINST.MASK_BRANCH.OUT_CHANNELS
        # assert cfg.MODEL.CONDINST.IUVHead.CHANNELS==cfg.MODEL.CONDINST.MASK_BRANCH.CHANNELS
        ##
        self.mask_out_stride = cfg.MODEL.CONDINST.MASK_OUT_STRIDE
        self.max_proposals = cfg.MODEL.CONDINST.MAX_PROPOSALS

        # build top module
        in_channels = self.proposal_generator.in_channels_to_top_module

        self.controller = nn.Conv2d(
            in_channels, self.mask_head.num_gen_params,
            kernel_size=3, stride=1, padding=1
        )
        torch.nn.init.normal_(self.controller.weight, std=0.01)
        torch.nn.init.constant_(self.controller.bias, 0)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        self._init_densepose_head(cfg)

        self.to(self.device)

    def _init_densepose_head(self, cfg):
        # fmt: off
        self.densepose_on          = cfg.MODEL.DENSEPOSE_ON
        if not self.densepose_on:
            return
        self.densepose_data_filter = build_densepose_data_filter(cfg)
        # dp_pooler_resolution       = cfg.MODEL.ROI_DENSEPOSE_HEAD.POOLER_RESOLUTION
        # dp_pooler_sampling_ratio   = cfg.MODEL.ROI_DENSEPOSE_HEAD.POOLER_SAMPLING_RATIO
        # dp_pooler_type             = cfg.MODEL.ROI_DENSEPOSE_HEAD.POOLER_TYPE
        # self.use_decoder           = cfg.MODEL.ROI_DENSEPOSE_HEAD.DECODER_ON
        # fmt: on
        # if self.use_decoder:
        #     dp_pooler_scales = (1.0 / input_shape[self.in_features[0]].stride,)
        # else:
        #     dp_pooler_scales = tuple(1.0 / input_shape[k].stride for k in self.in_features)
        # in_channels = [input_shape[f].channels for f in self.in_features][0]

        # if self.use_decoder:
        #     self.decoder = Decoder(cfg, input_shape, self.in_features)

        # self.densepose_pooler = ROIPooler(
        #     output_size=dp_pooler_resolution,
        #     scales=dp_pooler_scales,
        #     sampling_ratio=dp_pooler_sampling_ratio,
        #     pooler_type=dp_pooler_type,
        # )
        # self.densepose_head = build_densepose_head(cfg, in_channels)
        # self.densepose_predictor = build_densepose_predictor(
        #     cfg, self.densepose_head.n_out_channels
        # )
        # self.densepose_losses = build_densepose_losses(cfg)

    "TODO: convert raw input to detectron2.structures.instances.Instances to reuse densepose code"
    def forward(self, batched_inputs):
        # pdb.set_trace()
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        features = self.backbone(images.tensor)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            # if batched_inputs[0]['image'].shape[-1]!=images[0].shape[-1]:
            #     pdb.set_trace()
            self.add_bitmasks(gt_instances, images.tensor.size(-2), images.tensor.size(-1))
        else:
            gt_instances = None

        agg_feats, mask_feats, sem_losses = self.mask_branch(features, gt_instances)
        # iuv_feats, s_ins_feats = mask_feats[:,:self.iuv_fea_dim], mask_feats[:,self.iuv_fea_dim:]
        if self.use_mask_feats_iuvhead:
            iuv_feats, s_ins_feats = mask_feats, mask_feats
        else:
            iuv_feats, s_ins_feats = agg_feats, mask_feats

        # iuv_logits = self.iuv_head(iuv_feats, self.mask_branch.out_stride)
        # pdb.set_trace()

        # if torch.isnan(iuv_logits.mean()):
        #     pdb.set_trace()

        proposals, proposal_losses = self.proposal_generator(
            images, features, gt_instances, self.controller
        )

        if self.training:
            densepose_loss_dict = self._forward_mask_heads_train(proposals, s_ins_feats, iuv_feats, gt_instances=gt_instances)

            # # instances = 
            # loss_densepose = self._forward_densepose_train(mask_feats, gt_instances)

            losses = {}
            losses.update(sem_losses)
            losses.update(proposal_losses)
            # losses.update({"loss_mask": loss_mask})
            losses.update(densepose_loss_dict)

            # print(losses)
            # for k,v in losses.items():
            #     if torch.isnan(v):
            #         print(k,v)
            #         pdb.set_trace()

            return losses
        else:
            # "TODO add densepose inference"
            assert len(batched_inputs)==1
            imgsize = (batched_inputs[0]["height"],batched_inputs[0]["width"])
            densepose_instances, densepose_outputs = self._forward_mask_heads_test(proposals, s_ins_feats, iuv_feats, imgsize=imgsize)
            # pdb.set_trace()
            # import imageio
            # im = batched_inputs[0]["image"]/255.
            # im = F.interpolate(im.unsqueeze(0), size=imgsize)
            # dp = densepose_instances[0]['instances']
            # boxes = dp.pred_boxes.tensor.detach().cpu()
            # segms = dp.pred_densepose.coarse_segm.detach().cpu()
            # segms_comb = torch.zeros_like(im)
            # for idx in range(boxes.shape[0]):
            #     x1,y1,x2,y2 = boxes[idx].floor().int()
            #     segms_comb[:,:,y1:y2,x1:x2] += F.interpolate(segms[idx:idx+1,1:2], (y2-y1,x2-x1))
            # im = ((segms_comb+im)/2)[0].permute([1,2,0]).numpy()
            # imageio.imwrite('tmp/im_ins_bbox.png', im)
            # pdb.set_trace()

            # import imageio
            # im = batched_inputs[0]["image"]/255.
            # H, W = im.shape[-2:]
            # S = densepose_outputs.coarse_segm.detach().cpu()
            # ins_mask = F.interpolate(torch.sum(S,dim=0,keepdim=True), size=(H,W))[0,1:2]
            # # S = F.interpolate(S, size=(H,W))
            # # S = (S[:,0:1]<S[:,1:2]).float()
            # # ins_mask = torch.sum(S,dim=0,keepdim=True)[0]\
            # im = ((im+ins_mask)/2).permute([1,2,0]).numpy()
            # imageio.imwrite('tmp/im_ins.png', im)
            # pdb.set_trace()


            return densepose_instances


            # padded_im_h, padded_im_w = images.tensor.size()[-2:]
            # processed_results = []
            # for im_id, (input_per_image, image_size) in enumerate(zip(batched_inputs, images.image_sizes)):
            #     height = input_per_image.get("height", image_size[0])
            #     width = input_per_image.get("width", image_size[1])

            #     instances_per_im = pred_instances_w_masks[pred_instances_w_masks.im_inds == im_id]
            #     instances_per_im = self.postprocess(
            #         instances_per_im, height, width,
            #         padded_im_h, padded_im_w
            #     )

            #     processed_results.append({
            #         "instances": instances_per_im
            #     })

            # return processed_results

    def _forward_mask_heads_train(self, proposals, mask_feats, iuv_feats, gt_instances: List[Instances]):
        # prepare the inputs for mask heads
        pred_instances = proposals["instances"]
        # iuv_logits = self.iuv_head(iuv_feats, self.mask_branch.out_stride, pred_instances)

        if 0 <= self.max_proposals < len(pred_instances):
            inds = torch.randperm(len(pred_instances), device=mask_feats.device).long()
            logger.info("clipping proposals from {} to {}".format(
                len(pred_instances), self.max_proposals
            ))
            pred_instances = pred_instances[inds[:self.max_proposals]]

        pred_instances.mask_head_params = pred_instances.top_feats

        # loss_mask = self.mask_head(
        #     mask_feats, iuv_logits, self.mask_branch.out_stride,
        #     pred_instances, gt_instances
        # )
        loss_mask = self.mask_head(self.iuv_head, iuv_feats,
            mask_feats, self.mask_branch.out_stride,
            pred_instances, gt_instances=gt_instances, mask_out_bg_feats=self.mask_out_bg_feats
        )

        return loss_mask

    def _forward_mask_heads_test(self, proposals, mask_feats, iuv_feats, imgsize):
        # prepare the inputs for mask heads
        for im_id, per_im in enumerate(proposals):
            per_im.im_inds = per_im.locations.new_ones(len(per_im), dtype=torch.long) * im_id
        pred_instances = Instances.cat(proposals)
        pred_instances.mask_head_params = pred_instances.top_feat
        # iuv_logits = self.iuv_head(iuv_feats, self.mask_branch.out_stride, pred_instances)
        # densepose_instances, densepose_outputs = self.mask_head(
        #     mask_feats, iuv_logits, self.mask_branch.out_stride, pred_instances
        # )

        densepose_instances, densepose_outputs = self.mask_head(self.iuv_head, iuv_feats,
            mask_feats, self.mask_branch.out_stride, 
            pred_instances, mask_out_bg_feats=self.mask_out_bg_feats
        )

        # im_inds = densepose_instances.get('im_inds')
        boxes = densepose_instances.pred_boxes.tensor
        boxes = boxes/densepose_instances.image_size[0]*imgsize[0]
        # boxes_new = []
        # for idx in range(boxes.shape[0]):
        #     bb = boxes[idx:idx+1]
        #     x1, y1, x2, y2 = bb[0]
        #     im_idx = im_inds[idx].item()
        #     boxes_new.append(bb/(y2-y1)*imgsize_ori_list[im_idx][0])
        # pdb.set_trace()
        densepose_instances.set('pred_boxes', Boxes(boxes))

        # pdb.set_trace()
        # from ...utils.comm import SIUV_logit_to_iuv_batch
        # import imageio
        # S = densepose_outputs.coarse_segm[:1]
        # I = densepose_outputs.fine_segm[:1]
        # U = densepose_outputs.u[:1]
        # V = densepose_outputs.v[:1]
        # siuv_logit = torch.cat([S,I,U,V], dim=1)
        # iuv = SIUV_logit_to_iuv_batch(siuv_logit, norm=False, use_numpy=False)


        return [{'instances': densepose_instances}], densepose_outputs
        # return [densepose_instances], densepose_outputs
        # return {'instances': densepose_instances}, densepose_outputs

    def _forward_mask_heads_test_global(self, proposals, mask_feats, iuv_logits, imgsize):
        # prepare the inputs for mask heads
        for im_id, per_im in enumerate(proposals):
            per_im.im_inds = per_im.locations.new_ones(len(per_im), dtype=torch.long) * im_id
        pred_instances = Instances.cat(proposals)
        pred_instances.mask_head_params = pred_instances.top_feat

        densepose_instances = self.mask_head(
            mask_feats, iuv_logits, self.mask_branch.out_stride, pred_instances
        )

        return [{'instances': densepose_instances}]

            # padded_im_h, padded_im_w = images.tensor.size()[-2:]
            # processed_results = []
            # for im_id, (input_per_image, image_size) in enumerate(zip(batched_inputs, images.image_sizes)):
            #     height = input_per_image.get("height", image_size[0])
            #     width = input_per_image.get("width", image_size[1])

            #     instances_per_im = pred_instances_w_masks[pred_instances_w_masks.im_inds == im_id]
            #     instances_per_im = self.postprocess(
            #         instances_per_im, height, width,
            #         padded_im_h, padded_im_w
            #     )

            #     processed_results.append({
            #         "instances": instances_per_im
            #     })

            # return processed_results

    def add_bitmasks(self, instances, im_h, im_w):
        for per_im_gt_inst in instances:
            if not per_im_gt_inst.has("gt_masks"):
                continue
            start = int(self.mask_out_stride // 2)
            if isinstance(per_im_gt_inst.get("gt_masks"), PolygonMasks):
                polygons = per_im_gt_inst.get("gt_masks").polygons
                per_im_bitmasks = []
                per_im_bitmasks_full = []
                for per_polygons in polygons:
                    bitmask = polygons_to_bitmask(per_polygons, im_h, im_w)
                    bitmask = torch.from_numpy(bitmask).to(self.device).float()
                    start = int(self.mask_out_stride // 2)
                    bitmask_full = bitmask.clone()
                    bitmask = bitmask[start::self.mask_out_stride, start::self.mask_out_stride]

                    assert bitmask.size(0) * self.mask_out_stride == im_h
                    assert bitmask.size(1) * self.mask_out_stride == im_w

                    per_im_bitmasks.append(bitmask)
                    per_im_bitmasks_full.append(bitmask_full)

                per_im_gt_inst.gt_bitmasks = torch.stack(per_im_bitmasks, dim=0)
                per_im_gt_inst.gt_bitmasks_full = torch.stack(per_im_bitmasks_full, dim=0)
            else: # RLE format bitmask
                bitmasks = per_im_gt_inst.get("gt_masks").tensor
                h, w = bitmasks.size()[1:]
                # pad to new size
                bitmasks_full = F.pad(bitmasks, (0, im_w - w, 0, im_h - h), "constant", 0)
                bitmasks = bitmasks_full[:, start::self.mask_out_stride, start::self.mask_out_stride]
                per_im_gt_inst.gt_bitmasks = bitmasks
                per_im_gt_inst.gt_bitmasks_full = bitmasks_full

    def postprocess(self, results, output_height, output_width, padded_im_h, padded_im_w, mask_threshold=0.5):
        """
        Resize the output instances.
        The input images are often resized when entering an object detector.
        As a result, we often need the outputs of the detector in a different
        resolution from its inputs.
        This function will resize the raw outputs of an R-CNN detector
        to produce outputs according to the desired output resolution.
        Args:
            results (Instances): the raw outputs from the detector.
                `results.image_size` contains the input image resolution the detector sees.
                This object might be modified in-place.
            output_height, output_width: the desired output resolution.
        Returns:
            Instances: the resized output from the model, based on the output resolution
        """
        scale_x, scale_y = (output_width / results.image_size[1], output_height / results.image_size[0])
        resized_im_h, resized_im_w = results.image_size
        results = Instances((output_height, output_width), **results.get_fields())

        if results.has("pred_boxes"):
            output_boxes = results.pred_boxes
        elif results.has("proposal_boxes"):
            output_boxes = results.proposal_boxes

        output_boxes.scale(scale_x, scale_y)
        output_boxes.clip(results.image_size)

        results = results[output_boxes.nonempty()]

        if results.has("pred_global_masks"):
            mask_h, mask_w = results.pred_global_masks.size()[-2:]
            factor_h = padded_im_h // mask_h
            factor_w = padded_im_w // mask_w
            assert factor_h == factor_w
            factor = factor_h
            pred_global_masks = aligned_bilinear(
                results.pred_global_masks, factor
            )
            pred_global_masks = pred_global_masks[:, :, :resized_im_h, :resized_im_w]
            pred_global_masks = F.interpolate(
                pred_global_masks,
                size=(output_height, output_width),
                mode="bilinear", align_corners=False
            )
            pred_global_masks = pred_global_masks[:, 0, :, :]
            results.pred_masks = (pred_global_masks > mask_threshold).float()

        return results
