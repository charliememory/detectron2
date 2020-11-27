
import numpy as np
from typing import Dict, List, Optional
import math

import torch, pdb
from torch import nn
from torch.nn import functional as F
# from torch.utils.checkpoint import checkpoint_sequential
import torch.utils.checkpoint as checkpoint

from fvcore.nn import sigmoid_focal_loss_jit
import fvcore.nn.weight_init as weight_init
from detectron2.layers import Conv2d, ShapeSpec, get_norm

# from adet.layers import conv_with_kaiming_uniform
# from adet.utils.comm import aligned_bilinear
from densepose.layers import conv_with_kaiming_uniform
from densepose.utils.comm import aligned_bilinear, aligned_bilinear_layer
# from densepose.roi_heads.deeplab import ASPP

from lambda_networks import LambdaLayer

INF = 100000000

# Copied from
# https://github.com/pytorch/vision/blob/master/torchvision/models/segmentation/deeplabv3.py
# See https://arxiv.org/pdf/1706.05587.pdf for details
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(
                in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False
            ),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(),
        ]
        super(ASPPConv, self).__init__(*modules)


# class ASPPPooling(nn.Sequential):
#     def __init__(self, in_channels, out_channels):
#         super(ASPPPooling, self).__init__(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(in_channels, out_channels, 1, bias=False),
#             nn.GroupNorm(32, out_channels),
#             nn.ReLU(),
#         )

#     def forward(self, x):
#         size = x.shape[-2:]
#         x = super(ASPPPooling, self).forward(x)
#         return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels):
        super(ASPP, self).__init__()
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.GroupNorm(32, out_channels),
                nn.ReLU(inplace=True),
            )
        )

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        # modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            # nn.Dropout(0.5)
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


## Ref: densepose roi_head.py
class Decoder(nn.Module):
    """
    A semantic segmentation head described in detail in the Panoptic Feature Pyramid Networks paper
    (https://arxiv.org/abs/1901.02446). It takes FPN features as input and merges information from
    all levels of the FPN into single output.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec], in_features):
        super(Decoder, self).__init__()

        # fmt: off
        self.in_features      = in_features
        feature_strides       = {k: v.stride for k, v in input_shape.items()}
        feature_channels      = {k: v.channels for k, v in input_shape.items()}
        num_classes           = cfg.MODEL.ROI_DENSEPOSE_HEAD.DECODER_NUM_CLASSES
        conv_dims             = cfg.MODEL.ROI_DENSEPOSE_HEAD.DECODER_CONV_DIMS
        self.common_stride    = cfg.MODEL.ROI_DENSEPOSE_HEAD.DECODER_COMMON_STRIDE
        norm                  = cfg.MODEL.ROI_DENSEPOSE_HEAD.DECODER_NORM
        # fmt: on

        self.scale_heads = []
        for in_feature in self.in_features:
            head_ops = []
            head_length = max(
                1, int(np.log2(feature_strides[in_feature]) - np.log2(self.common_stride))
            )
            for k in range(head_length):
                conv = Conv2d(
                    feature_channels[in_feature] if k == 0 else conv_dims,
                    conv_dims,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=not norm,
                    norm=get_norm(norm, conv_dims),
                    activation=F.relu,
                )
                weight_init.c2_msra_fill(conv)
                head_ops.append(conv)
                if feature_strides[in_feature] != self.common_stride:
                    head_ops.append(
                        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
                    )
            self.scale_heads.append(nn.Sequential(*head_ops))
            self.add_module(in_feature, self.scale_heads[-1])
        self.predictor = Conv2d(conv_dims, num_classes, kernel_size=1, stride=1, padding=0)
        weight_init.c2_msra_fill(self.predictor)

    def forward(self, features: List[torch.Tensor]):
        for i, _ in enumerate(self.in_features):
            if i == 0:
                x = self.scale_heads[i](features[i])
            else:
                x = x + self.scale_heads[i](features[i])
        x = self.predictor(x)
        return x


def build_mask_branch(cfg, input_shape):
    return MaskBranch(cfg, input_shape)


class MaskBranch(nn.Module):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()
        self.in_features = cfg.MODEL.CONDINST.MASK_BRANCH.IN_FEATURES
        self.sem_loss_on = cfg.MODEL.CONDINST.MASK_BRANCH.SEMANTIC_LOSS_ON
        self.num_outputs = cfg.MODEL.CONDINST.MASK_BRANCH.OUT_CHANNELS
        norm = cfg.MODEL.CONDINST.MASK_BRANCH.NORM
        num_convs = cfg.MODEL.CONDINST.MASK_BRANCH.NUM_CONVS
        agg_channels = cfg.MODEL.CONDINST.MASK_BRANCH.AGG_CHANNELS
        channels = cfg.MODEL.CONDINST.MASK_BRANCH.CHANNELS
        self.out_stride = input_shape[self.in_features[0]].stride
        self.num_lambda_layer = cfg.MODEL.CONDINST.MASK_BRANCH.NUM_LAMBDA_LAYER
        self.use_aspp = cfg.MODEL.CONDINST.MASK_BRANCH.USE_ASPP
        lambda_layer_r = cfg.MODEL.CONDINST.MASK_BRANCH.LAMBDA_LAYER_R
        self.checkpoint_grad_num = cfg.MODEL.CONDINST.CHECKPOINT_GRAD_NUM
        self.v2 = cfg.MODEL.CONDINST.v2
        self.use_res_input   = cfg.MODEL.CONDINST.MASK_BRANCH.RESIDUAL_INPUT
        self.use_res_after_relu   = cfg.MODEL.CONDINST.MASK_BRANCH.RESIDUAL_SKIP_AFTER_RELU

        self.use_agg_feat    = cfg.MODEL.CONDINST.IUVHead.USE_AGG_FEATURES
        if self.use_agg_feat:
            self.in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES

        self.use_weight_std = cfg.MODEL.CONDINST.IUVHead.WEIGHT_STANDARDIZATION
        self.use_eca = cfg.MODEL.CONDINST.IUVHead.Efficient_Channel_Attention

        self.add_skeleton_feat = cfg.MODEL.CONDINST.IUVHead.SKELETON_FEATURES

        feature_channels = {k: v.channels for k, v in input_shape.items()}

        conv_block_no_act = conv_with_kaiming_uniform(norm, activation=False, use_weight_std=self.use_weight_std)
        conv_block = conv_with_kaiming_uniform(norm, activation=True, use_weight_std=self.use_weight_std)


        self.use_decoder = False
        # self.use_decoder           = cfg.MODEL.ROI_DENSEPOSE_HEAD.DECODER_ON
        # if self.use_decoder:
        #     self.decoder = Decoder(cfg, input_shape, self.in_features)
        #     assert agg_channels==cfg.MODEL.ROI_DENSEPOSE_HEAD.DECODER_CONV_DIMS
        # else:
        self.refine = nn.ModuleList()
        for idx, in_feature in enumerate(self.in_features):

            # if num_lambda_layer>=len(self.in_features)-idx:
            #     layer = LambdaLayer(
            #         dim = feature_channels[in_feature],
            #         dim_out = agg_channels,
            #         r = lambda_layer_r,         # the receptive field for relative positional encoding (23 x 23)
            #         dim_k = 16,
            #         heads = 4,
            #         dim_u = 4
            #     )
            #     self.refine.append(layer)
            # else:
            if self.v2 and idx>0 and in_feature not in ["p6","p7"]:
                if self.add_skeleton_feat:
                    self.refine.append(nn.Sequential(*[
                        conv_block_no_act(
                            feature_channels[in_feature],
                            agg_channels, 3, 1
                        ),
                        nn.Upsample(scale_factor=2**idx)
                    ]))
                else:
                    self.refine.append(nn.Sequential(*[
                        conv_block(
                            feature_channels[in_feature],
                            agg_channels, 3, 1
                        ),
                        nn.Upsample(scale_factor=2**idx)
                    ]))

                    # aligned_bilinear_layer(
                    #     factor=2**idx
                    # ),
            else:
                self.refine.append(
                    conv_block(
                        feature_channels[in_feature],
                        agg_channels, 3, 1
                    )
                )

        if self.add_skeleton_feat:
            self.conv_skeleton = conv_block(
                agg_channels+55,
                agg_channels, 3, 1
            )



        if self.use_eca:
            self.eca = eca_layer(agg_channels, k_size=3)


        if self.use_aspp:
            self.ASPP = ASPP(agg_channels, [12, 24, 112], agg_channels)  # 6, 12, 56
            self.add_module("ASPP", self.ASPP)

        if self.num_lambda_layer>0:
            self.lambda_layer = LambdaLayer(
                dim = agg_channels,
                dim_out = agg_channels,
                r = lambda_layer_r,         # the receptive field for relative positional encoding (23 x 23)
                dim_k = 16,
                heads = 4,
                dim_u = 4
            )
            # agg_channels = channels
        # if "p1" == self.in_features[0]:
        #     self.down_conv = conv_block(
        #         channels, channels, 3, 2, 1
        #     )
        if "p2" == self.in_features[0]:
            if self.v2:
                # if self.add_skeleton_feat:
                #     tower = [conv_block(
                #             agg_channels+55, channels, 3, 2, 1
                #         )]
                # else:
                tower = [conv_block(
                        agg_channels, channels, 3, 2, 1
                    )]
            else:
                self.down_conv = conv_block(
                    agg_channels, channels, 3, 2, 1
                )
                tower = [conv_block(
                        channels, channels, 3, 1
                    )]
        else:
            tower = [conv_block(
                    agg_channels, channels, 3, 1
                )]
        for i in range(1,num_convs):
            tower.append(conv_block(
                channels, channels, 3, 1
            ))
        tower.append(nn.Conv2d(
            channels, max(self.num_outputs, 1), 1
        ))
        if self.use_res_input or self.use_res_after_relu:
            for idx, layer in enumerate(tower):
                self.add_module('tower_layer{}'.format(idx), layer)
            self.tower = tower
        else:
            self.add_module('tower', nn.Sequential(*tower))

        # self.amp_enable =  cfg.SOLVER.AMP.ENABLED
        # if self.amp_enable:
        #     self = self.half()
        # ## debug
        # # if self.amp_enable:
        # # [p[1].data.dtype for p in self.named_parameters()]
        # for p in self.named_parameters():
        #     if p[1].data.dtype!=torch.float16:
        #         print(p[1].data.dtype)
        #         pdb.set_trace()

        # pdb.set_trace()
        # if self.sem_loss_on:
        #     num_classes = cfg.MODEL.FCOS.NUM_CLASSES
        #     self.focal_loss_alpha = cfg.MODEL.FCOS.LOSS_ALPHA
        #     self.focal_loss_gamma = cfg.MODEL.FCOS.LOSS_GAMMA

        #     in_channels = feature_channels[self.in_features[0]]
        #     self.seg_head = nn.Sequential(
        #         conv_block(in_channels, channels, kernel_size=3, stride=1),
        #         conv_block(channels, channels, kernel_size=3, stride=1)
        #     )

        #     self.logits = nn.Conv2d(channels, num_classes, kernel_size=1, stride=1)

        #     prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
        #     bias_value = -math.log((1 - prior_prob) / prior_prob)
        #     torch.nn.init.constant_(self.logits.bias, bias_value)

    ## Ref: https://github.com/prigoyal/pytorch_memonger/blob/master/tutorial/Checkpointing_for_PyTorch_models.ipynb
    def custom(self, module):
        def custom_forward(*inputs):
            inputs = module(inputs[0])
            return inputs
        return custom_forward

    def forward(self, features, skeleton_feats=None, gt_instances=None):

        # if self.use_decoder:
        #     features = [features[f] for f in self.in_features]
        #     # pdb.set_trace()
        #     x = self.decoder(features)
        # else:
        for i, f in enumerate(self.in_features):
            if i == 0:
                x = self.refine[i](features[f])
            else:
                if self.v2:
                    if self.checkpoint_grad_num>0:
                        modules = [module for k, module in self.refine[i]._modules.items()]
                        x_p = checkpoint.checkpoint_sequential(modules,1,features[f])
                    else:
                        x_p = self.refine[i](features[f])
                else:
                    x_p = self.refine[i](features[f])
                    target_h, target_w = x.size()[2:]
                    h, w = x_p.size()[2:]
                    assert target_h % h == 0
                    assert target_w % w == 0
                    factor_h, factor_w = target_h // h, target_w // w
                    assert factor_h == factor_w
                    x_p = aligned_bilinear(x_p, factor_h)
                x = x + x_p
        if self.add_skeleton_feat:
            # pdb.set_trace()
            x = self.conv_skeleton(torch.cat([x,skeleton_feats], dim=1))

        if self.use_aspp:
            x = self.ASPP(x)
        if self.num_lambda_layer>0:
            x = self.lambda_layer(x)
        if self.use_eca:
            x = self.eca(x)
        agg_feats = x
        # pdb.set_trace()
        # if "p1" == self.in_features[0]:
        #     mask_feats = self.tower(self.down_conv(x))

        if not self.v2:
            if "p2" == self.in_features[0]:
                x = self.down_conv(x)
        
        # if self.checkpoint_grad_num>0:
        #     # mask_feats = checkpoint.checkpoint(self.custom(self.tower), x)
        #     modules = [module for k, module in self.tower._modules.items()]
        #     mask_feats = checkpoint.checkpoint_sequential(modules,1,x)
        # else:
        #     mask_feats = self.tower(x)

        if self.use_res_after_relu:
            res = None
            for idx, layer in enumerate(self.tower):
                if idx==1:
                    res = x
                elif idx==3:
                    x = x + res
                x = layer(x)
            mask_feats = x
        elif self.use_res_input:
            res = None
            for idx, layer in enumerate(self.tower):
                if idx==0:
                    res = x
                elif idx==2:
                    x = x + res
                if idx==3:
                    res = x
                elif idx==4:
                    x = x + res
                x = layer(x)
            mask_feats = x
        else:
            mask_feats = self.tower(x)


        if self.num_outputs == 0:
            mask_feats = mask_feats[:, :self.num_outputs]

        losses = {}
        # auxiliary thing semantic loss
        # if self.training and self.sem_loss_on:
        #     logits_pred = self.logits(self.seg_head(
        #         features[self.in_features[0]]
        #     ))

        #     # compute semantic targets
        #     semantic_targets = []
        #     for per_im_gt in gt_instances:
        #         h, w = per_im_gt.gt_bitmasks_full.size()[-2:]
        #         areas = per_im_gt.gt_bitmasks_full.sum(dim=-1).sum(dim=-1)
        #         areas = areas[:, None, None].repeat(1, h, w)
        #         areas[per_im_gt.gt_bitmasks_full == 0] = INF
        #         areas = areas.permute(1, 2, 0).reshape(h * w, -1)
        #         min_areas, inds = areas.min(dim=1)
        #         per_im_sematic_targets = per_im_gt.gt_classes[inds] + 1
        #         per_im_sematic_targets[min_areas == INF] = 0
        #         per_im_sematic_targets = per_im_sematic_targets.reshape(h, w)
        #         semantic_targets.append(per_im_sematic_targets)

        #     semantic_targets = torch.stack(semantic_targets, dim=0)

        #     # resize target to reduce memory
        #     semantic_targets = semantic_targets[
        #                        :, None, self.out_stride // 2::self.out_stride,
        #                        self.out_stride // 2::self.out_stride
        #                        ]

        #     # prepare one-hot targets
        #     num_classes = logits_pred.size(1)
        #     class_range = torch.arange(
        #         num_classes, dtype=logits_pred.dtype,
        #         device=logits_pred.device
        #     )[:, None, None]
        #     class_range = class_range + 1
        #     one_hot = (semantic_targets == class_range).float()
        #     num_pos = (one_hot > 0).sum().float().clamp(min=1.0)

        #     loss_sem = sigmoid_focal_loss_jit(
        #         logits_pred, one_hot,
        #         alpha=self.focal_loss_alpha,
        #         gamma=self.focal_loss_gamma,
        #         reduction="sum",
        #     ) / num_pos
        #     losses['loss_sem'] = loss_sem

        return agg_feats, mask_feats, losses
