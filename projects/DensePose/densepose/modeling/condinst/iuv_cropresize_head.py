from typing import Dict
import math

import torch
from torch import nn
import torch.nn.functional as F

from fvcore.nn import sigmoid_focal_loss_jit
from detectron2.layers import ShapeSpec

# from adet.layers import conv_with_kaiming_uniform
# from adet.utils.comm import aligned_bilinear
from detectron2.modeling.poolers import ROIPooler
from detectron2.layers import ConvTranspose2d
from densepose.layers import conv_with_kaiming_uniform, deform_conv
from densepose.utils.comm import compute_locations, compute_grid, aligned_bilinear
from ..roi_heads import DensePoseDeepLabHead
# from .. import (
#     build_densepose_data_filter,
#     build_densepose_head,
#     build_densepose_losses,
#     build_densepose_predictor,
#     densepose_inference,
# )
from lambda_networks import LambdaLayer
from .iuv_head import get_embedder, CoordGlobalIUVHead
import pdb

INF = 100000000

def build_iuv_cropresize_head(cfg, input_shape):
    # return GlobalIUVHead(cfg)
    return CoordGlobalIUVCropResizeHead(cfg, use_rel_coords=True, input_shape=input_shape)


class CoordGlobalIUVCropResizeHead(CoordGlobalIUVHead):
    def __init__(self, cfg, use_rel_coords=True, input_shape=None):
        super().__init__(cfg, use_rel_coords)

        # self.densepose_data_filter = build_densepose_data_filter(cfg)
        self.in_features  = cfg.MODEL.ROI_HEADS.IN_FEATURES
        dp_pooler_resolution       = cfg.MODEL.ROI_DENSEPOSE_HEAD.POOLER_RESOLUTION
        dp_pooler_sampling_ratio   = cfg.MODEL.ROI_DENSEPOSE_HEAD.POOLER_SAMPLING_RATIO
        dp_pooler_type             = cfg.MODEL.ROI_DENSEPOSE_HEAD.POOLER_TYPE
        self.use_decoder           = cfg.MODEL.ROI_DENSEPOSE_HEAD.DECODER_ON
        # fmt: on
        # pdb.set_trace()
        # if self.use_decoder:
        #     dp_pooler_scales = (1.0 / input_shape[self.in_features[0]].stride,)
        # else:
        #     dp_pooler_scales = tuple(1.0 / input_shape[k].stride for k in self.in_features)
        dp_pooler_scales = (1.0 / input_shape[self.in_features[0]].stride,)
        in_channels = [input_shape[f].channels for f in self.in_features][0]

        # if self.use_decoder:
        #     self.decoder = Decoder(cfg, input_shape, self.in_features)

        self.densepose_pooler = ROIPooler(
            output_size=dp_pooler_resolution,
            scales=dp_pooler_scales,
            sampling_ratio=dp_pooler_sampling_ratio,
            pooler_type=dp_pooler_type,
        )

    def forward(self, fpn_features, s_logits, iuv_feats, iuv_feat_stride, rel_coord, instances, fg_mask=None, gt_instances=None):
        assert not self.use_abs_coords

        N, _, H, W = iuv_feats.size()

        if self.use_rel_coords: 
            if self.use_pos_emb:
                rel_coord = self.position_embedder(rel_coord)
        else:
            rel_coord = torch.zeros([N,2,H,W], device=iuv_feats.device).to(dtype=iuv_feats.dtype)
        iuv_head_inputs = torch.cat([rel_coord, iuv_feats], dim=1) 


        # if self.use_abs_coords: 
        #     abs_coord = compute_grid(H, W, device=iuv_feats.device)[None,...].repeat(N,1,1,1)
        #     if self.use_pos_emb:
        #         abs_coord = self.position_embedder(abs_coord)
        # else:
        #     abs_coord = torch.zeros([N,2,H,W], device=iuv_feats.device).to(dtype=iuv_feats.dtype)
        # iuv_head_inputs = torch.cat([abs_coord, iuv_head_inputs], dim=1)


        ############
        # boxes_xyxy_list = [ins.get('gt_boxes').tensor for ins in gt_instances]
        # feat_list = []
        # cnt = 0
        # invalid_ij_list = []
        # for i in range(N):
        #     imgH, imgW = gt_instances[i].image_size
        #     boxes_xyxy = gt_instances[i].get('gt_boxes').tensor.clone()
        #     # boxes_xyxy[:,]

        #     boxes_xyxy[:,0] = boxes_xyxy[:,0]/imgW*W
        #     boxes_xyxy[:,1] = boxes_xyxy[:,1]/imgH*H
        #     boxes_xyxy[:,2] = boxes_xyxy[:,2]/imgW*W
        #     boxes_xyxy[:,3] = boxes_xyxy[:,3]/imgH*H

            
        #     for j in range(boxes_xyxy.shape[0]):
        #         x1,y1,x2,y2 = boxes_xyxy[j].int()
        #         x1 = max(0,x1)
        #         y1 = max(0,y1)
        #         x2 = min(W-1,x2)
        #         y2 = min(H-1,y2)
        #         if x2-x1<5 or y2-y1<5:
        #             invalid_ij_list.append((i,j))
        #         else:
        #             feat = F.interpolate(iuv_head_inputs[i:i+1,:,y1:y2,x1:x2], size=(56,56))
        #             feat_list.append(feat)
        #             cnt += 1
        # iuv_head_inputs = torch.cat(feat_list, dim=0)

        proposal_boxes = [x.get('gt_boxes') for x in gt_instances]
        iuv_head_inputs = self.densepose_pooler([iuv_head_inputs], proposal_boxes)


        iuv_logit = self.tower(iuv_head_inputs)

        assert iuv_feat_stride >= self.iuv_out_stride
        assert iuv_feat_stride % self.iuv_out_stride == 0
        iuv_logit = aligned_bilinear(iuv_logit, int(iuv_feat_stride / self.iuv_out_stride))


        ############

        r = int(iuv_feat_stride / self.iuv_out_stride)
        iuv_logit_all = torch.zeros([iuv_feats.shape[0],iuv_logit.shape[1],iuv_feats.shape[2]*r,iuv_feats.shape[3]*r], device=iuv_feats.device)
        mask_all = torch.zeros([iuv_feats.shape[0],1,iuv_feats.shape[2]*r,iuv_feats.shape[3]*r], device=iuv_feats.device)

        N, _, H, W = iuv_logit_all.size()

        cnt = 0
        for i in range(N):
            imgH, imgW = gt_instances[i].image_size
            boxes_xyxy = gt_instances[i].get('gt_boxes').tensor.clone()
            ins_masks = gt_instances[i].gt_bitmasks
            # boxes_xyxy[:,]

            boxes_xyxy[:,0] = boxes_xyxy[:,0]/imgW*W
            boxes_xyxy[:,1] = boxes_xyxy[:,1]/imgH*H
            boxes_xyxy[:,2] = boxes_xyxy[:,2]/imgW*W
            boxes_xyxy[:,3] = boxes_xyxy[:,3]/imgH*H
            # pdb.set_trace()

            for j in range(boxes_xyxy.shape[0]):
                if (i,j) not in invalid_ij_list:
                    x1,y1,x2,y2 = boxes_xyxy[j].int()
                    x1 = max(0,x1)
                    y1 = max(0,y1)
                    x2 = min(W-1,x2)
                    y2 = min(H-1,y2)
                    # print(x1,y1,x2,y2)
                    mask = F.interpolate(ins_masks[j:j+1][None,...].float(), size=(H,W), mode="nearest")
                    mask_all[i:i+1] += mask
                    tmp = torch.zeros_like(iuv_logit_all)[0:1]
                    tmp[:,:,y1:y2,x1:x2] = F.interpolate(iuv_logit[cnt:cnt+1], size=(y2-y1,x2-x1))
                    iuv_logit_all[i:i+1] += mask * tmp
                    cnt += 1
        # pdb.set_trace()
        # import imageio
        # imageio.imwrite("tmp/iuv_logit_all.png", iuv_logit_all[0][0].detach().cpu().numpy())
        # imageio.imwrite("tmp/mask_all.png", mask_all[0][0].detach().cpu().numpy())
        # imageio.imwrite("tmp/gt_bitmasks.png", gt_instances[0].gt_bitmasks.float().sum(dim=0).detach().cpu().numpy())
        
        iuv_logit_all = iuv_logit_all/(mask_all + 1e-5)





        return iuv_logit_all

