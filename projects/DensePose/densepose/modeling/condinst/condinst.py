# -*- coding: utf-8 -*-
import logging
from typing import Dict, List, Optional

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import pdb, copy

from detectron2.structures import ImageList, Boxes
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.structures.instances import Instances
from detectron2.structures.masks import PolygonMasks, polygons_to_bitmask
from detectron2.modeling.roi_heads import select_foreground_proposals
from detectron2.modeling.postprocessing import detector_postprocess

from .dynamic_mask_head import build_dynamic_mask_head
from .mask_branch import build_mask_branch
from .iuv_head import build_iuv_head
from .iuv_deeplab_head import build_iuv_deeplab_head
from .iuv_unet_head import build_iuv_unet_head
from .iuv_multiscale_head import build_iuv_multiscale_head
from .iuv_multilayermask_head import build_iuv_multilayermask_head
from .iuv_multilayercoord_head import build_iuv_multilayercoord_head
from .iuv_scaleattn_head import build_iuv_scaleattn_head
from .iuv_cropresize_head import build_iuv_cropresize_head
from .iuv_pooler_head import build_iuv_pooler_head
from .iuv_pooler2_head import build_iuv_pooler2_head
from .iuv_sparsepooler2_head import build_iuv_sparsepooler2_head

# from adet.utils.comm import aligned_bilinear
from densepose.utils.comm import aligned_bilinear

from .. import (
    build_densepose_data_filter,
    build_densepose_head,
    build_densepose_losses,
    build_densepose_predictor,
    densepose_inference,
)

import sys
sys.path.append('./RAFT/core')
from raft import RAFT
from raft_utils import flow_viz
# from .smooth_utils import TransferTexture, InputPadder, grid_sampler, make_meshgrid, sec_to_hm_str #, forward_interpolate
# from utils import img2tex_forwardwarp, torch_inpaint_oneChannel
# from utils import img_iuv_resize_with_center_pad, img_resize_with_center_crop
# from utils import disp2colormap_single, SIUV_logit_to_iuv_batch, SIUV_logit_to_IonehotUV_batch

__all__ = ["CondInst"]


logger = logging.getLogger(__name__)

class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

class flow_model_wrapper:
    def __init__(self, device):
    # def create_and_load_netFlow(self, device):
        # parser.add_argument('--model', help="restore checkpoint")
        # parser.add_argument('--seq_img_dir', help="sequence images for evaluation")
        # parser.add_argument('--backward_flow', action='store_true', help='calculate flow from i+1 to i')
        # parser.add_argument('--small', action='store_true', help='use small model')
        # parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
        # parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
        # args = parser.parse_args()
        sys.path.append('./RAFT/core')
        from raft import RAFT
        class Args:
            def __init__(self):
                # self.model = "./RAFT/models/raft-sintel.pth"
                # self.small = False
                # self.mixed_precision = False
                # self.dropout = 0
                # self.alternate_corr = False
                self.model = "./RAFT/models/raft-small.pth"
                self.small = True
                self.mixed_precision = False
                self.dropout = 0
                self.alternate_corr = False
        args = Args()

        # model = RAFT(args)
        self.model = torch.nn.DataParallel(RAFT(args))
        self.model.load_state_dict(torch.load(args.model, map_location=device))
        self.model.eval()
        print("Create and load model successfully")
        # return model

    def pred_flow(self, img0, img1, iters=20):
        try:
            assert img0.min()>=0 and img0.max()>=10 and img0.max()<=255, "input image range should be [0,255], but got [{},{}]".format(img0.min(),img0.max())
        except:
            print("input image range should be [0,255], but got [{},{}]".format(img0.min(),img0.max()))
            raise ValueError

        padder = InputPadder(img0.shape, mode='sintel')
        img0, img1 = padder.pad(img0, img1)

        flow_low, flow_pr = self.model(img0, img1, iters, test_mode=True)
        flow = padder.unpad(flow_pr)
        return flow

    def tensor_warp_via_flow(self, tensor, flow):
        b, _, h, w = tensor.shape
        coords = self.flow2coord(flow).permute([0,2,3,1]) # [0,h-1], [0,w-1]
        tensor = F.grid_sample(tensor, coords) #, mode='bilinear', align_corners=True)
        return tensor

    def flow2coord(self, flow):
        def meshgrid(height, width):
            x_t = torch.matmul(
                torch.ones(height, 1), torch.linspace(-1.0, 1.0, width).view(1, width))
            y_t = torch.matmul(
                torch.linspace(-1.0, 1.0, height).view(height, 1), torch.ones(1, width))

            grid_x = x_t.view(1, 1, height, width)
            grid_y = y_t.view(1, 1, height, width)
            return grid_x, grid_y
            # return torch.cat([grid_x,grid_y], dim=-1)

        b, _, h, w = flow.shape
        grid_x, grid_y = meshgrid(h, w)
        coord_x = flow[:,0:1]/w + grid_x.to(flow.device)
        coord_y = flow[:,1:2]/h + grid_y.to(flow.device)
        return torch.cat([coord_x,coord_y], dim=1)


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
        if cfg.MODEL.CONDINST.IUVHead.NAME=="IUVDeepLabHead":
            self.iuv_head = build_iuv_deeplab_head(cfg)
        elif cfg.MODEL.CONDINST.IUVHead.NAME=="IUVUnetHead":
            self.iuv_head = build_iuv_unet_head(cfg)
        elif cfg.MODEL.CONDINST.IUVHead.NAME=="IUVMultiscaleHead":
            self.iuv_head = build_iuv_multiscale_head(cfg)
        elif cfg.MODEL.CONDINST.IUVHead.NAME=="IUVMultilayermaskHead":
            self.iuv_head = build_iuv_multilayermask_head(cfg)
        elif cfg.MODEL.CONDINST.IUVHead.NAME=="IUVMultilayercoordHead":
            self.iuv_head = build_iuv_multilayercoord_head(cfg)
        elif cfg.MODEL.CONDINST.IUVHead.NAME=="IUVScaleAttnHead":
            self.iuv_head = build_iuv_scaleattn_head(cfg)
        elif cfg.MODEL.CONDINST.IUVHead.NAME=="IUVCropResizeHead":
            self.iuv_head = build_iuv_cropresize_head(cfg, self.backbone.output_shape())
        elif cfg.MODEL.CONDINST.IUVHead.NAME=="IUVPoolerHead":
            self.iuv_head = build_iuv_pooler_head(cfg, self.backbone.output_shape())
        elif cfg.MODEL.CONDINST.IUVHead.NAME=="IUVPooler2Head":
            self.iuv_head = build_iuv_pooler2_head(cfg, self.backbone.output_shape())
        elif cfg.MODEL.CONDINST.IUVHead.NAME=="IUVSparsePooler2Head":
            self.iuv_head = build_iuv_sparsepooler2_head(cfg, self.backbone.output_shape())
        elif cfg.MODEL.CONDINST.IUVHead.DISABLE==True:
            self.iuv_head = None
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
        self.rand_flip = cfg.MODEL.CONDINST.RAND_FLIP
        self.rand_scale = cfg.MODEL.CONDINST.RAND_SCALE
        self.rand_scale_gap = cfg.MODEL.CONDINST.RAND_SCALE_GAP
        self.infer_TTA_with_rand_flow = cfg.MODEL.CONDINST.INFER_TTA_WITH_RAND_FLOW

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
        self._init_raft(cfg)

        # from detectron2.modeling.proposal_generator import build_proposal_generator
        # from detectron2.modeling.roi_heads import build_roi_heads
        # self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())
        # self.roi_heads = build_roi_heads(cfg, self.backbone.output_shape())

        self.to(self.device)

        self.sparsity_list = []

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

    def _init_raft(self, cfg):
        ## load raft flow model
        self.infer_smooth_frame_num = cfg.MODEL.INFERENCE_SMOOTH_FRAME_NUM
        if self.infer_smooth_frame_num>0 or self.infer_TTA_with_rand_flow:
            self.flow_model = flow_model_wrapper(self.device)
            # self.create_and_load_netFlow()
            # self.flow_model.to(self.device)
            # self.flow_model.eval()
            self.img_queue = []
            self.dp_output_queue = []

    def _init_densepose_head(self, cfg):
        # fmt: off
        self.densepose_on          = cfg.MODEL.DENSEPOSE_ON
        if not self.densepose_on:
            return
        self.densepose_data_filter = build_densepose_data_filter(cfg)
        self.use_gt_ins = cfg.MODEL.CONDINST.IUVHead.GT_INSTANCES
        # if self.training and not self.use_gt_ins:
        #     assert cfg.MODEL.FCOS.YIELD_PROPOSAL==True
        self.checkpoint_grad_num = cfg.MODEL.CONDINST.CHECKPOINT_GRAD_NUM
        self.finetune_iuvhead_only = cfg.MODEL.CONDINST.FINETUNE_IUVHead_ONLY
        self.inference_global_siuv = cfg.MODEL.CONDINST.INFERENCE_GLOBAL_SIUV

        # self.add_skeleton_feat = cfg.MODEL.CONDINST.IUVHead.SKELETON_FEATURES
        # self.use_gt_skeleton = cfg.MODEL.CONDINST.IUVHead.GT_SKELETON

        self.use_aux_skeleton = cfg.MODEL.CONDINST.AUX_SUPERVISION_GLOBAL_SKELETON
        self.segm_trained_by_masks = cfg.MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS

        # ## original ROI based densepose head
        # from detectron2.modeling.poolers import ROIPooler
        # from ..roi_heads.roi_head import Decoder
        # input_shape = self.backbone.output_shape()
        # self.in_features  = cfg.MODEL.ROI_HEADS.IN_FEATURES
        # dp_pooler_resolution       = cfg.MODEL.ROI_DENSEPOSE_HEAD.POOLER_RESOLUTION
        # dp_pooler_sampling_ratio   = cfg.MODEL.ROI_DENSEPOSE_HEAD.POOLER_SAMPLING_RATIO
        # dp_pooler_type             = cfg.MODEL.ROI_DENSEPOSE_HEAD.POOLER_TYPE
        # self.use_decoder           = cfg.MODEL.ROI_DENSEPOSE_HEAD.DECODER_ON
        # # fmt: on
        # if self.use_decoder:
        #     dp_pooler_scales = (1.0 / input_shape[self.in_features[0]].stride,)
        # else:
        #     dp_pooler_scales = tuple(1.0 / input_shape[k].stride for k in self.in_features)
        # # pdb.set_trace()
        # in_channels = [input_shape[f].channels for f in self.in_features][0]

        # if self.use_decoder:
        #     self.decoder = Decoder(cfg, input_shape, self.in_features)

        # self.densepose_pooler = ROIPooler(
        #     output_size=dp_pooler_resolution,
        #     scales=dp_pooler_scales,
        #     sampling_ratio=dp_pooler_sampling_ratio,
        #     pooler_type=dp_pooler_type,
        # )
        # self.densepose_head = build_densepose_head(cfg, self.iuv_fea_dim)
        # # self.densepose_head = build_densepose_head(cfg, in_channels)
        # self.densepose_predictor = build_densepose_predictor(
        #     cfg, self.densepose_head.n_out_channels
        # )
        # self.densepose_losses = build_densepose_losses(cfg)

    # def create_and_load_netFlow(self):
    #     # parser.add_argument('--model', help="restore checkpoint")
    #     # parser.add_argument('--seq_img_dir', help="sequence images for evaluation")
    #     # parser.add_argument('--backward_flow', action='store_true', help='calculate flow from i+1 to i')
    #     # parser.add_argument('--small', action='store_true', help='use small model')
    #     # parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    #     # parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    #     # args = parser.parse_args()
    #     class Args:
    #         def __init__(self):
    #             # self.model = "./RAFT/models/raft-sintel.pth"
    #             # self.small = False
    #             # self.mixed_precision = False
    #             # self.dropout = 0
    #             # self.alternate_corr = False
    #             self.model = "./RAFT/models/raft-small.pth"
    #             self.small = True
    #             self.mixed_precision = False
    #             self.dropout = 0
    #             self.alternate_corr = False
    #     args = Args()

    #     # netFlow = RAFT(args)
    #     netFlow = torch.nn.DataParallel(RAFT(args))
    #     netFlow.load_state_dict(torch.load(args.model, map_location=self.device))
    #     print("Create and load netFlow successfully")
    #     return netFlow

    # "TODO"
    # def flow_based_iuvlogit_avg(self, inputs, outputs, frame_idxs):
    #     # weight_dic = {-2:0.2, -1:0.2, 0: 0.2, 1:0.2, 2:0.2}
    #     weight_dic = {-2:0.1, -1:0.2, 0: 0.4, 1:0.2, 2:0.1}
    #     # weight_dic = {-2:0.1, -1:0.3, 0: 1.2, 1:0.3, 2:0.1}
    #     # weight_dic = {-2:0., -1:0., 0: 1, 1:0., 2:0.}
    #     w_sum = sum([v for k,v in weight_dic.items()])
    #     weight_dic = {k:v/w_sum for k,v in weight_dic.items()} 
    #     # print(weight_dic)

    #     # iuvlogit_list = [inputs[("iuv", "pred", fid)] for fid in frame_idxs]
    #     # iuvlogit_avg = torch.stack(iuvlogit_list, dim=0).mean(dim=0)

    #     assert self.opt.use_warped_input and self.opt.use_iuv_logit

    #     ## warp input
    #     color_tgt = inputs[("color", 0)]
    #     boxes_tgt = inputs[("bbox", "pred", 0)]
    #     mask_fw_list, mask_bw_list = [], []
    #     iuvlogit_list, mask_list = [], []
    #     flow_list, iuv_warp_list, s_warp_list = [], [], []
    #     for fid in frame_idxs:
    #         color_ref = inputs[("color", fid)]
    #         iuv_ref = inputs[("iuv", "pred", fid)]
    #         s_ref = inputs[("ins_mask", "pred", fid)]
    #         # "TODO debug"
    #         # pdb.set_trace()
    #         ins_num = s_ref.shape[1]
    #         s_ref_list = torch.chunk(s_ref, ins_num, dim=1)
    #         s_ref_list = [s[:,0] for s in s_ref_list]
    #         siuv_ref = torch.cat(s_ref_list + [iuv_ref], dim=1)
    #         with torch.no_grad():
    #             ## forward flow
    #             if fid==0:
    #                 # continue
    #                 flow_fw = torch.zeros_like(flow_fw)
    #             else:
    #                 flow_fw = self.pred_flow(self.models["flow"], color_tgt*255, color_ref*255)
    #                 siuv_ref = self.tensor_warp_via_flow(siuv_ref, flow_fw)
    #             flow_list.append(flow_fw.detach().cpu())
    #             s_warp_fw, iuv_warp_fw = siuv_ref[:,:-77], siuv_ref[:,-77:]
    #             # inputs[("iuv", "pred", fid)] = iuv_warp_fw
    #             iuv_warp_list.append(iuv_warp_fw)
    #             s_warp_list.append(s_warp_fw)
    #     # pdb.set_trace()
    #     # iuvlogit_list = [inputs[("iuv", "pred", fid)]*weight_dic[fid] for fid in frame_idxs]
    #     # iuvlogit_list = [inputs[("iuv", "pred", fid)]*weight_dic[fid] for fid in frame_idxs]
    #     # pdb.set_trace()
    #     iuvlogit_list = [iuv_warp_list[i]*weight_dic[fid] for i,fid in enumerate(frame_idxs)]
    #     iuvlogit_avg = torch.stack(iuvlogit_list, dim=0).sum(dim=0)
    #     outputs[("flow", "list_list")].append(flow_list)
    #     return inputs, outputs, iuv_warp_list, iuvlogit_avg

    ## Ref: https://github.com/prigoyal/pytorch_memonger/blob/master/tutorial/Checkpointing_for_PyTorch_models.ipynb
    def custom(self, module):
        def custom_forward(*inputs):
            inputs = module(inputs[0])
            return inputs
        return custom_forward

    "TODO: convert raw input to detectron2.structures.instances.Instances to reuse densepose code"
    def forward(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]

        self.img_ori_size_list = [x.shape[1:] for x in images] # HxW
        self.is_flipped = False
        # assert not self.rand_flip, 'current densepose annotation is not suitable for flip augmentation'
        # if self.rand_flip and torch.rand(1)>0.5:
        #     images = [torch.flip(im,dims=[2]) for im in images] # CxHxW
        #     for batch_idx in range(len(batched_inputs)):
        #         instances = batched_inputs[batch_idx]['instances']
        #         # pdb.set_trace()  
        #         _, h, w = images[batch_idx].shape
        #         # for ins_idx in range(len(instances)):
        #         instances.gt_boxes.tensor = w - instances.gt_boxes.tensor
        #         # instances.gt_masks.tensor = 
        #     self.is_flipped = True
                
        self.scale_factor = 1.
        self.is_scaled = False
        if self.rand_scale and torch.rand(1)>0.5:
            # min_scale, max_scale = 0.8, 1.2
            min_scale, max_scale = 1.-self.rand_scale_gap, 1.+self.rand_scale_gap
            scale_factor = torch.rand(1)*(max_scale-min_scale) + min_scale
            # print([im.shape for im in images])
            "If contains too small instances, not downscale"
            for batch_idx in range(len(batched_inputs)):
                instances = batched_inputs[batch_idx]['instances']
                if self.rand_scale_gap>0.2:
                    thres = 1024
                else:
                    # thres = 64
                    thres = 512
                if instances.gt_masks.tensor.sum(dim=[1,2]).min()<=thres:
                    scale_factor = 1. + abs(1.-scale_factor)
                    # print(scale_factor)

            images = [F.interpolate(im[None,...], scale_factor=scale_factor)[0] for im in images]
            for batch_idx in range(len(batched_inputs)):
                # instances = batched_inputs[batch_idx]['instances']
                # for ins_idx in range(len(instances)):
                batched_inputs[batch_idx]['instances'].gt_boxes.tensor *= scale_factor
                # print(instances.gt_masks.tensor.sum(dim=[1,2]))

                # instances.gt_masks.tensor = F.interpolate(instances.gt_masks.tensor.float()[:,None,...], 
                #                                     scale_factor=scale_factor,mode='nearest')[:,0].bool()
            self.scale_factor = scale_factor
            self.is_scaled = True

        images = ImageList.from_tensors(images, self.backbone.size_divisibility)

        # import imageio
        # imageio.imwrite("tmp/img_0.png", images.tensor[0].permute(1,2,0).detach().cpu().numpy())
        # imageio.imwrite("tmp/img_1.png", images.tensor[1].permute(1,2,0).detach().cpu().numpy())
        # skeleton_feats = [x["skeleton_feat"].to(self.device) for x in batched_inputs]
        # skeleton_feats = ImageList.from_tensors(skeleton_feats, self.backbone.size_divisibility)
        # pdb.set_trace()
        skeleton_feats = None
        # if self.add_skeleton_feat:
        #     if self.use_gt_skeleton:
        #         skeleton_feats = self.process_skeleton_feats(batched_inputs, images.tensor.size(-2), images.tensor.size(-1))

        skeleton_feats_gt = None
        if self.use_aux_skeleton and self.training:
            skeleton_feats_gt = self.process_skeleton_feats(batched_inputs, images.tensor.size(-2), images.tensor.size(-1))

        if self.finetune_iuvhead_only:
            with torch.no_grad():
                features = self.backbone(images.tensor)

                if "instances" in batched_inputs[0]:
                    gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                    # if batched_inputs[0]['image'].shape[-1]!=images[0].shape[-1]:
                    #     pdb.set_trace()
                    self.add_bitmasks(gt_instances, images.tensor.size(-2), images.tensor.size(-1))
                else:
                    gt_instances = None

                agg_feats, mask_feats, sem_losses = self.mask_branch(features, skeleton_feats, gt_instances)
                # iuv_feats, s_ins_feats = mask_feats[:,:self.iuv_fea_dim], mask_feats[:,self.iuv_fea_dim:]
                if self.use_mask_feats_iuvhead:
                    iuv_feats, s_ins_feats = mask_feats, mask_feats
                else:
                    iuv_feats, s_ins_feats = agg_feats, mask_feats

                # iuv_logits = self.iuv_head(iuv_feats, self.mask_branch.out_stride)
                # pdb.set_trace()

                # if torch.isnan(iuv_logits.mean()):
                #     pdb.set_trace()
                 
                # pdb.set_trace()

                proposals, proposal_losses = self.proposal_generator(
                    images, features, gt_instances, self.controller
                )




            assert self.training
            densepose_loss_dict = self._forward_mask_heads_train(proposals, features, s_ins_feats, iuv_feats, gt_instances=gt_instances)

            # # instances = 
            # loss_densepose = self._forward_densepose_train(mask_feats, gt_instances)

            losses = {}
            losses.update(sem_losses)
            losses.update(proposal_losses)
            # losses.update({"loss_mask": loss_mask})
            losses.update(densepose_loss_dict)

            return losses
        else:

            if self.training:
                features = self.backbone(images.tensor)

                if "instances" in batched_inputs[0]:
                    gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                    self.add_bitmasks(gt_instances, images.tensor.size(-2), images.tensor.size(-1))
                else:
                    gt_instances = None

                agg_feats, mask_feats, sem_losses = self.mask_branch(features, skeleton_feats, gt_instances)
                if self.use_mask_feats_iuvhead:
                    iuv_feats, s_ins_feats = mask_feats, mask_feats
                else:
                    iuv_feats, s_ins_feats = agg_feats, mask_feats
                proposals, proposal_losses = self.proposal_generator(
                    images, features, gt_instances, self.controller
                )

                densepose_loss_dict = self._forward_mask_heads_train(proposals, features, s_ins_feats, iuv_feats, 
                                        gt_instances=gt_instances, images=images, skeleton_feats_gt=skeleton_feats_gt)

                # # instances = 
                # loss_densepose = self._forward_densepose_train(mask_feats, gt_instances)

                losses = {}
                losses.update(sem_losses)
                losses.update(proposal_losses)
                # losses.update({"loss_mask": loss_mask})
                losses.update(densepose_loss_dict)

                return losses
            else:
                assert len(batched_inputs)==1

                def _run_single_img(images, batched_inputs, skeleton_feats):
                    features = self.backbone(images.tensor)

                    if "instances" in batched_inputs[0]:
                        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                        self.add_bitmasks(gt_instances, images.tensor.size(-2), images.tensor.size(-1))
                        ## Print sparsity
                        # h, w = gt_instances[0].gt_bitmasks_full.shape[-2:]
                        # sparsity = torch.clamp(gt_instances[0].gt_bitmasks_full.float().sum(0),0,1).sum()/h/w
                        # self.sparsity_list.append(sparsity)
                        # ss = torch.tensor(self.sparsity_list)
                        # print("sparsity mean {}, min {}, max {}".format(ss.mean(), ss.min(), ss.max()))
                    else:
                        gt_instances = None

                    agg_feats, mask_feats, sem_losses = self.mask_branch(features, skeleton_feats, gt_instances)
                    if self.use_mask_feats_iuvhead:
                        iuv_feats, s_ins_feats = mask_feats, mask_feats
                    else:
                        iuv_feats, s_ins_feats = agg_feats, mask_feats
                    proposals, proposal_losses = self.proposal_generator(
                        images, features, gt_instances, self.controller
                    )

                    # "TODO add densepose inference"
                    assert len(batched_inputs)==1
                    imgsize = (batched_inputs[0]["height"],batched_inputs[0]["width"])
                    densepose_instances = self._forward_mask_heads_test(proposals, features, s_ins_feats, iuv_feats, gt_instances=gt_instances, imgsize=imgsize)

                    return densepose_instances

                def run_single_img(images, batched_inputs, skeleton_feats):
                    # if self.infer_TTA_with_rand_flow:
                    #     assert len(batched_inputs)==1
                    #     assert not self.use_gt_ins, "TTA img do not have gt"
                    #     gap = 12
                    #     offset_list = [[gap,gap], [gap,-gap], [-gap,gap], [-gap,-gap]]
                    #     num = len(offset_list) + 1
                    #     densepose_instances = _run_single_img(images, batched_inputs, skeleton_feats)
                    #     densepose_instances[0].pred_densepose.fine_segm /= num
                    #     densepose_instances[0].pred_densepose.u /= num
                    #     densepose_instances[0].pred_densepose.v /= num
                    #     densepose_instances[0].pred_densepose.coarse_segm = num
                    #     for (offset_x, offset_y) in offset_list:
                    #         flow_fw = torch.zeros_like(images.tensor)[:,:2]
                    #         flow_fw[:,0:1] += offset_x
                    #         flow_fw[:,1:2] += offset_y
                    #         # pdb.set_trace()
                    #         images.tensor = self.flow_model.tensor_warp_via_flow(images.tensor, flow_fw)
                    #         dp = _run_single_img(images, batched_inputs, skeleton_feats)
                            
                    # else:
                    densepose_instances = _run_single_img(images, batched_inputs, skeleton_feats)
                    return densepose_instances

                "TODO: parallerl and cache the warping to speed up"
                if self.infer_smooth_frame_num>0 or self.infer_TTA_with_rand_flow:
                    assert len(batched_inputs)==1
                    assert not self.use_gt_ins, "image_adj do not have gt"
                    # pdb.set_trace()
                    center_idx = self.infer_smooth_frame_num

                    if self.infer_TTA_with_rand_flow:
                        assert len(batched_inputs)==1
                        assert not self.use_gt_ins, "TTA img do not have gt"
                        gap = 4
                        offset_list = [[gap,gap], [gap,-gap], [0,0], [-gap,gap], [-gap,-gap]]
                        num = len(offset_list) 
                        self.img_queue = [batched_inputs[0]["image"]]*num
                        self.img_queue = [x.to(self.device) for x in self.img_queue]
                        self.dp_output_queue = []
                        for img, (offset_x, offset_y) in zip(self.img_queue, offset_list):
                            img = img[None,...]
                            flow_fw = torch.zeros_like(img)[:,:2]
                            # pdb.set_trace()
                            flow_fw[:,0:1] += offset_x
                            flow_fw[:,1:2] += offset_y
                            img_warp = self.flow_model.tensor_warp_via_flow(img, flow_fw)[0]

                            # pdb.set_trace()
                            # import imageio
                            # imageio.imwrite('tmp/img.jpg', img[0].permute([1,2,0]).cpu().numpy())
                            # imageio.imwrite('tmp/img_warp.jpg', img_warp.permute([1,2,0]).cpu().numpy()) 
                            densepose_instances = run_single_img(ImageList.from_tensors([self.normalizer(img_warp)], self.backbone.size_divisibility), 
                                                            batched_inputs, skeleton_feats)
                            self.dp_output_queue.append(densepose_instances)
                        self.img_queue = [F.interpolate(im[None,...], densepose_instances[0].pred_densepose.u.shape[-2:]) for im in self.img_queue]


                        # densepose_instances = _run_single_img(images, batched_inputs, skeleton_feats)
                        # densepose_instances[0].pred_densepose.fine_segm /= num
                        # densepose_instances[0].pred_densepose.u /= num
                        # densepose_instances[0].pred_densepose.v /= num
                        # densepose_instances[0].pred_densepose.coarse_segm = num
                        # for (offset_x, offset_y) in offset_list:
                        #     flow_fw = torch.zeros_like(images.tensor)[:,:2]
                        #     flow_fw[:,0:1] += offset_x
                        #     flow_fw[:,1:2] += offset_y
                        #     # pdb.set_trace()
                        #     images.tensor = self.flow_model.tensor_warp_via_flow(images.tensor, flow_fw)
                        #     dp = _run_single_img(images, batched_inputs, skeleton_feats)
                    else:
                        images_adj = batched_inputs[0]["image_adj_list"] #.to(self.device)
                        if self.img_queue == []:
                            self.img_queue = images_adj[:center_idx] + [batched_inputs[0]["image"]] + images_adj[center_idx:]
                            
                            self.img_queue = [x.to(self.device) for x in self.img_queue]
                            # self.img_queue = [self.normalizer(x.to(self.device)) for x in self.img_queue]
                            # self.img_queue = ImageList.from_tensors(self.img_queue, self.backbone.size_divisibility)
                            for img in self.img_queue:
                                densepose_instances = run_single_img(ImageList.from_tensors([self.normalizer(img)], self.backbone.size_divisibility), 
                                                                batched_inputs, skeleton_feats)
                                self.dp_output_queue.append(densepose_instances)
                            self.img_queue = [F.interpolate(im[None,...], densepose_instances[0].pred_densepose.u.shape[-2:]) for im in self.img_queue]
                        else:
                            self.img_queue.pop(0)
                            self.img_queue.append(images_adj[-1].to(self.device))


                            densepose_instances = run_single_img(ImageList.from_tensors([self.normalizer(self.img_queue[-1])], self.backbone.size_divisibility), 
                                                                batched_inputs, skeleton_feats)
                            # pdb.set_trace()
                            self.dp_output_queue.pop(0)
                            self.dp_output_queue.append(densepose_instances)
                            self.img_queue[-1] = F.interpolate(self.img_queue[-1][None,...], densepose_instances[0].pred_densepose.u.shape[-2:])


                    dp_output_warp = copy.deepcopy(self.dp_output_queue[center_idx])
                    coarse_segm = dp_output_warp[0].pred_densepose.coarse_segm
                    coarse_segm_list = torch.chunk(coarse_segm, coarse_segm.shape[0])
                    coarse_segm_list_list = [[a] for a in coarse_segm_list]
                    # coarse_segm_cnt_list = [1] * len(coarse_segm_list)

                    # weight_list = [0.2,0.2,0.2,0.2,0.2]
                    if self.infer_TTA_with_rand_flow:
                        weight_list = [0.2,0.2,0.4,0.2,0.2]
                    else:
                        weight_list = [0.1,0.2,0.4,0.2,0.1]
                    weight_list = [w/sum(weight_list) for w in weight_list]
                    dp_output_warp[0].pred_densepose.fine_segm *= weight_list[center_idx]
                    dp_output_warp[0].pred_densepose.u *= weight_list[center_idx]
                    dp_output_warp[0].pred_densepose.v *= weight_list[center_idx]
                    dp_output_warp[0].pred_densepose.coarse_segm *= weight_list[center_idx]
                    for i in range(len(self.img_queue)):
                        if i == center_idx:
                            continue
                        else:
                            # pdb.set_trace()
                            # img_tgt = torch.tensor(img_queue[center_idx]).permute([2,0,1])[None,...].float()
                            # img_ref = torch.tensor(img_queue[i]).permute([2,0,1])[None,...].float()
                            img_tgt = self.img_queue[center_idx]
                            img_ref = self.img_queue[i]
                            if self.infer_TTA_with_rand_flow:
                                img = self.img_queue[i]
                                offset_x, offset_y = offset_list[i]
                                flow_fw = torch.zeros_like(img)[:,:2]
                                # pdb.set_trace()
                                flow_fw[:,0:1] -= offset_x
                                flow_fw[:,1:2] -= offset_y
                            else:
                                flow_fw = self.flow_model.pred_flow(img_tgt, img_ref)
                            dp = copy.deepcopy(self.dp_output_queue[i])
                            if not hasattr(dp[0], "pred_densepose"):
                                continue
                            ## Warp IUV
                            dp_output_warp[0].pred_densepose.fine_segm += weight_list[i] * self.flow_model.tensor_warp_via_flow(dp[0].pred_densepose.fine_segm, flow_fw)
                            dp_output_warp[0].pred_densepose.u += weight_list[i] * self.flow_model.tensor_warp_via_flow(dp[0].pred_densepose.u, flow_fw)
                            dp_output_warp[0].pred_densepose.v += weight_list[i] * self.flow_model.tensor_warp_via_flow(dp[0].pred_densepose.v, flow_fw)
                            s_tmp = weight_list[i] * dp_output_warp[0].pred_densepose.coarse_segm
                            s_list_tmp = torch.chunk(s_tmp, s_tmp.shape[0])
                            binary_th = 0.1
                            iou_th = 0.7
                            for s_t in s_list_tmp:
                                s_t_b = s_t[0,1]>binary_th
                                for idx in range(len(coarse_segm_list_list)):
                                    s = coarse_segm_list_list[idx][0]
                                    s_b = s[0,1]>binary_th
                                    iou = (s_t_b*s_b).float().sum() / ((s_t_b+s_b).float().sum() + 1e-7)
                                    # print(iou)
                                    # import imageio
                                    # pdb.set_trace()
                                    # imageio.imwrite('tmp/s_{}.png'.format(iou), torch.cat([s_t,s], dim=-1)[0].permute(1,2,0).cuda().numpy())
                                    if iou>iou_th:
                                        coarse_segm_list_list[idx].append(s_t)
                                        # break

                    # dp_output_warp[0].pred_densepose.fine_segm /= len(self.img_queue)
                    # dp_output_warp[0].pred_densepose.u /= len(self.img_queue)
                    # dp_output_warp[0].pred_densepose.v /= len(self.img_queue)
                    # coarse_segm = [torch.stack(s_list,0).mean(0) for s_list in coarse_segm_list_list]
                    coarse_segm = [torch.stack(s_list,0).sum(0) for s_list in coarse_segm_list_list]

                    dp_output_warp[0].pred_densepose.coarse_segm = torch.cat(coarse_segm, dim=0)
                    densepose_instances = dp_output_warp


                else:
                    densepose_instances = run_single_img(images, batched_inputs, skeleton_feats)
                    # features = self.backbone(images.tensor)

                    # if "instances" in batched_inputs[0]:
                    #     gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                    #     self.add_bitmasks(gt_instances, images.tensor.size(-2), images.tensor.size(-1))
                    # else:
                    #     gt_instances = None

                    # agg_feats, mask_feats, sem_losses = self.mask_branch(features, skeleton_feats, gt_instances)
                    # if self.use_mask_feats_iuvhead:
                    #     iuv_feats, s_ins_feats = mask_feats, mask_feats
                    # else:
                    #     iuv_feats, s_ins_feats = agg_feats, mask_feats
                    # proposals, proposal_losses = self.proposal_generator(
                    #     images, features, gt_instances, self.controller
                    # )

                    # # "TODO add densepose inference"
                    # assert len(batched_inputs)==1
                    # imgsize = (batched_inputs[0]["height"],batched_inputs[0]["width"])
                    # densepose_instances = self._forward_mask_heads_test(proposals, features, s_ins_feats, iuv_feats, gt_instances=gt_instances, imgsize=imgsize)
                
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
                # S = densepose_instances[0].pred_densepose.coarse_segm.detach().cpu()
                # ins_mask = F.interpolate(torch.sum(S,dim=0,keepdim=True), size=(H,W))[0,1:2]
                # # S = F.interpolate(S, size=(H,W))
                # # S = (S[:,0:1]<S[:,1:2]).float()
                # # ins_mask = torch.sum(S,dim=0,keepdim=True)[0]\
                # im = ((im+ins_mask)/2).permute([1,2,0]).numpy()
                # imageio.imwrite('tmp/im_ins.png', im)
                # pdb.set_trace()

                # ## original ROI based densepose head
                # if self.iuv_head is None:
                #     instances = densepose_instances[0]['instances']
                #     pred_boxes = [x.pred_boxes for x in instances]

                #     features = [features[f] for f in self.in_features]
                #     if self.use_decoder:
                #         features = [self.decoder(features)]
                #     # features = [iuv_feats]

                #     features_dp = self.densepose_pooler(features, pred_boxes)
                #     if len(features_dp) > 0:
                #         densepose_head_outputs = self.densepose_head(features_dp)
                #         densepose_predictor_outputs = self.densepose_predictor(densepose_head_outputs)
                #     else:
                #         densepose_predictor_outputs = None

                #     densepose_inference(densepose_predictor_outputs, instances)

                #     densepose_instances, _ = self.roi_heads(images, features, proposals, None)

                "TODO: add postprocess to eval-test."
                "So, during test, first postprocess, then apply pooler"

                if self.inference_global_siuv:
                    padded_im_h, padded_im_w = images.tensor.size()[-2:]
                    processed_results = []
                    for im_id, (input_per_image, image_size) in enumerate(zip(batched_inputs, images.image_sizes)):
                        height = input_per_image.get("height", image_size[0])
                        width = input_per_image.get("width", image_size[1])

                        results_per_image = densepose_instances[im_id]
                        # if hasattr(results_per_image ,"pred_densepose"):
                        results_per_image = self.postprocess_global_siuv(
                            results_per_image, height, width,
                            padded_im_h, padded_im_w
                        )

                        processed_results.append({
                            "instances": results_per_image
                        })


                else:

                    processed_results = []
                    for results_per_image, input_per_image, image_size in zip(
                        densepose_instances, batched_inputs, images.image_sizes
                    ):
                        height = input_per_image.get("height", image_size[0])
                        width = input_per_image.get("width", image_size[1])
                        results_per_image._image_size = (height, width)

                        # "TODO"
                        # kpts =  torch.zeros([len(results_per_image),17,3])
                        # results_per_image.set("keypoints", kpts)

                        processed_results.append({"instances": results_per_image})

                torch.cuda.empty_cache()
                return processed_results




    def _forward_mask_heads_train(self, proposals, fpn_features, mask_feats, iuv_feats, gt_instances: List[Instances], images=None, skeleton_feats_gt=None):
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

        if not self.use_gt_ins:
            # keep_idxs = proposals["keep_idxs"]
            # prepare the inputs for mask heads
            pps = proposals["proposals_nms"]
            for im_id, per_im in enumerate(pps):
                per_im.im_inds = per_im.locations.new_ones(len(per_im), dtype=torch.long) * im_id
            # pdb.set_trace()
            for ii in range(len(pps)):
                pps[ii]._image_size = (0,0)
            pred_instances_nms = Instances.cat(pps)
            pred_instances_nms.mask_head_params = pred_instances_nms.top_feat
        else:
            # keep_idxs = None
            pred_instances_nms = None

        # loss_mask = self.mask_head(
        #     mask_feats, iuv_logits, self.mask_branch.out_stride,
        #     pred_instances, gt_instances
        # )
        loss_mask = self.mask_head(self.iuv_head, 
            fpn_features, mask_feats, iuv_feats, 
            self.mask_branch.out_stride, pred_instances, gt_instances=gt_instances, 
            mask_out_bg_feats=self.mask_out_bg_feats, pred_instances_nms=pred_instances_nms, images=images, skeleton_feats_gt=skeleton_feats_gt
        )

        return loss_mask

    def _forward_mask_heads_test(self, proposals, fpn_features, mask_feats, iuv_feats, gt_instances, imgsize):
        # prepare the inputs for mask heads
        for im_id, per_im in enumerate(proposals):
            per_im.im_inds = per_im.locations.new_ones(len(per_im), dtype=torch.long) * im_id
        pred_instances = Instances.cat(proposals)
        pred_instances.mask_head_params = pred_instances.top_feat
        # pdb.set_trace()
        # iuv_logits = self.iuv_head(iuv_feats, self.mask_branch.out_stride, pred_instances)
        # densepose_instances, densepose_outputs = self.mask_head(
        #     mask_feats, iuv_logits, self.mask_branch.out_stride, pred_instances
        # )

        densepose_instances, densepose_outputs = self.mask_head(self.iuv_head, 
            fpn_features, mask_feats, iuv_feats,
            self.mask_branch.out_stride, pred_instances, gt_instances=gt_instances,
            mask_out_bg_feats=self.mask_out_bg_feats
        )

        # im_inds = densepose_instances.get('im_inds')
        # if self.use_gt_ins:

        boxes = densepose_instances.pred_boxes.tensor
        boxes = boxes/densepose_instances.image_size[0]*imgsize[0]
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
        densepose_instances = [densepose_instances]
        densepose_inference(densepose_outputs, densepose_instances)
        return densepose_instances

        # return [{'instances': densepose_instances}]

        # if do_postprocess:
        #     return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        # else:
        #     return results
        # return [{'instances': densepose_instances}], densepose_outputs
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
    def _aug_bitmasks(self, bitmasks, is_flipped, is_scaled, scale_factor=1.):
        # bitmasks format: BxHxW
        # import os, imageio
        # imageio.imwrite('tmp/bitmask.png', bitmasks[0].cpu().numpy())
        if is_flipped:
            bitmasks = torch.flip(bitmasks, dims=[2])
        if is_scaled:
            bitmasks = F.interpolate(bitmasks.float()[:,None,...], scale_factor=scale_factor,mode='nearest')[:,0].bool()
        return bitmasks

    def add_bitmasks(self, instances, im_h, im_w):
        for per_im_gt_inst in instances:
            start = int(self.mask_out_stride // 2)

            if not self.segm_trained_by_masks:
                N_ins = len(per_im_gt_inst)
                # H,W = gt_instances[i].image_size
                boxes_xyxy = per_im_gt_inst.gt_boxes.tensor
                bitmasks_full = torch.zeros([N_ins,im_h,im_w], dtype=torch.float32, device=self.device)
                for j in range(len(per_im_gt_inst.gt_densepose)):
                    if per_im_gt_inst.gt_densepose[j] is not None:
                        x1,y1,x2,y2 = boxes_xyxy[j].int()
                        fg_mask = (per_im_gt_inst.gt_densepose[j].segm[None,None,...]>0).float()
                        fg_mask = self._aug_bitmasks(fg_mask, self.is_flipped, is_scaled=False)
                        bitmasks_full[j,y1:y2,x1:x2] = F.interpolate(fg_mask, (y2-y1,x2-x1), mode="nearest")[0,0]
                    # else:
                    #     print(j)
                    #     pdb.set_trace()
                bitmasks = bitmasks_full[:, start::self.mask_out_stride, start::self.mask_out_stride]
                # pdb.set_trace()
                # import imageio
                # imageio.imwrite("output/tmp/bitmasks_0.png",bitmasks[0].cpu().numpy())

                per_im_gt_inst.set('gt_bitmasks', bitmasks)
                per_im_gt_inst.set('gt_bitmasks_full', bitmasks_full)

                # pdb.set_trace()
            else:

                if not per_im_gt_inst.has("gt_masks"):
                    continue
                if isinstance(per_im_gt_inst.get("gt_masks"), PolygonMasks):
                    polygons = per_im_gt_inst.get("gt_masks").polygons
                    per_im_bitmasks = []
                    per_im_bitmasks_full = []
                    for per_polygons, img_ori_size in zip(polygons, self.img_ori_size_list):
                        h, w = img_ori_size
                        bitmask = polygons_to_bitmask(per_polygons, h, w)
                        bitmask = torch.from_numpy(bitmask).to(self.device).float()
                        bitmask = self._aug_bitmasks(bitmask)
                        bitmask = F.pad(bitmask, (0, im_w - w, 0, im_h - h), "constant", 0)

                        bitmask_full = bitmask.clone()
                        start = int(self.mask_out_stride // 2)
                        bitmask = bitmask[start::self.mask_out_stride, start::self.mask_out_stride]

                        assert bitmask.size(0) * self.mask_out_stride == im_h
                        assert bitmask.size(1) * self.mask_out_stride == im_w

                        per_im_bitmasks.append(bitmask)
                        per_im_bitmasks_full.append(bitmask_full)

                    per_im_gt_inst.gt_bitmasks = torch.stack(per_im_bitmasks, dim=0)
                    per_im_gt_inst.gt_bitmasks_full = torch.stack(per_im_bitmasks_full, dim=0)
                else: # RLE format bitmask
                    bitmasks = per_im_gt_inst.get("gt_masks").tensor
                    bitmasks = self._aug_bitmasks(bitmasks, self.is_flipped, self.is_scaled, self.scale_factor)
                    h, w = bitmasks.size()[1:]
                    # pad to new size
                    bitmasks_full = F.pad(bitmasks, (0, im_w - w, 0, im_h - h), "constant", 0)
                    # print(bitmasks.shape, bitmasks_full.shape)
                    # pdb.set_trace()
                    bitmasks = bitmasks_full[:, start::self.mask_out_stride, start::self.mask_out_stride]
                    per_im_gt_inst.gt_bitmasks = bitmasks
                    per_im_gt_inst.gt_bitmasks_full = bitmasks_full


    def process_skeleton_feats(self, batched_inputs, im_h, im_w):
        skeleton_feats = [x["skeleton_feat"] for x in batched_inputs]
        # skeleton_feats = ImageList.from_tensors(skeleton_feats, self.backbone.size_divisibility)
        im_h = im_h // self.mask_out_stride
        im_w = im_w // self.mask_out_stride
        ske_feat_list = []
        start = int(self.mask_out_stride // 2)
        for idx in range(len(skeleton_feats)):
            h, w = skeleton_feats[idx].shape[-2:]
            ske_feat = F.pad(skeleton_feats[idx], (0, im_w - w, 0, im_h - h), "constant", 0)
            # ske_feat_full = F.pad(skeleton_feats[idx], (0, im_w - w, 0, im_h - h), "constant", 0)
            # ske_feat = ske_feat_full[:, start::self.mask_out_stride, start::self.mask_out_stride]
            ske_feat_list.append(ske_feat)
        # pdb.set_trace()
        # import imageio
        # imageio.imwrite("tmp/ske_feat.png", ske_feat.sum(0).detach().cpu().numpy())

        return torch.stack(ske_feat_list, dim=0).to(self.device)



    # @staticmethod
    # def _postprocess(instances, batched_inputs, image_sizes):
    #     """
    #     Rescale the output instances to the target size.
    #     """
    #     # note: private function; subject to changes
    #     processed_results = []
    #     for results_per_image, input_per_image, image_size in zip(
    #         instances, batched_inputs, image_sizes
    #     ):
    #         height = input_per_image.get("height", image_size[0])
    #         width = input_per_image.get("width", image_size[1])
    #         results_per_image._image_size = (height, width)
    #         processed_results.append({"instances": results_per_image})
    #         # pdb.set_trace()
    #         # r = detector_postprocess(results_per_image, height, width)
    #         # processed_results.append({"instances": r})
    #     return processed_results

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

    def postprocess_global_siuv(self, results, output_height, output_width, padded_im_h, padded_im_w, mask_threshold=0.5, remove_empty=True):
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
        # scale_x, scale_y = (output_width / results.image_size[1], output_height / results.image_size[0])
        resized_im_h, resized_im_w = results.image_size
        results = Instances((output_height, output_width), **results.get_fields())
        # results._image_size = (output_height, output_width)

        # if results.has("pred_boxes"):
        #     output_boxes = results.pred_boxes
        # elif results.has("proposal_boxes"):
        #     output_boxes = results.proposal_boxes

        # output_boxes.scale(scale_x, scale_y)
        # output_boxes.clip(results.image_size)

        ## Change bbox to whole image size
        # N_ins = results.pred_boxes.tensor.shape[0]
        # boxes = torch.tensor([0,0,output_width,output_height], device=results.pred_boxes.tensor.device)
        # results.pred_boxes.tensor = torch.stack([boxes]*N_ins, dim=0)

        if not hasattr(results ,"pred_densepose"):
            return results

        if remove_empty:
            # valid_idxs = output_boxes.nonempty()
            valid_idxs = results.pred_densepose.coarse_segm[:,1:2].sum(dim=[1,2,3]) > 0
            # pdb.set_trace()
            pred_densepose = results.pred_densepose
            pred_densepose.coarse_segm = pred_densepose.coarse_segm[valid_idxs]
            results.remove("pred_densepose")
            results = results[valid_idxs]
            results.set("pred_densepose", pred_densepose)

        ## resize global siuv
        for name in ["coarse_segm", "fine_segm", "u", "v"]:
        # if results.has("pred_global_masks"):
            mask = getattr(results.pred_densepose, name)
            mask_h, mask_w = mask.size()[-2:]
            factor_h = padded_im_h // mask_h
            factor_w = padded_im_w // mask_w
            assert factor_h == factor_w
            factor = factor_h
            mask = aligned_bilinear(
                mask, factor
            )
            mask = mask[:, :, :resized_im_h, :resized_im_w]
            mask = F.interpolate(
                mask,
                size=(output_height, output_width),
                mode="bilinear", align_corners=False
            )
            # pred_global_masks = pred_global_masks[:, 0, :, :]
            # results.pred_masks = (pred_global_masks > mask_threshold).float()
            setattr(results.pred_densepose, name, mask)

        return results
