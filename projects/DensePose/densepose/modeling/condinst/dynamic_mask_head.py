from typing import Any, List
import torch
from torch.nn import functional as F
from torch import nn

from detectron2.structures import Instances, Boxes
# from adet.utils.comm import compute_locations, aligned_bilinear
from densepose.utils.comm import compute_locations, aligned_bilinear

from ...structures import DensePoseChartPredictorOutput
from .. import (
    build_densepose_data_filter,
    build_densepose_head,
    build_densepose_losses,
    build_densepose_predictor,
    densepose_inference,
)
import math, pdb

def constrain_bbox(pred_instances: Instances):
    for idx in range(len(pred_instances)):
        ins = pred_instances[idx] 
        imgH, imgW = ins.image_size
        bbox_xyxy = ins.pred_boxes.tensor
        for i in range(bbox_xyxy.shape[0]):
            x1, y1, x2, y2 = bbox_xyxy[i]
            bbox_xyxy[i][0] = max(x1,0)
            bbox_xyxy[i][1] = max(y1,0)
            bbox_xyxy[i][2] = min(x2,imgW-1)
            bbox_xyxy[i][3] = min(y2,imgH-1)
        pred_instances[idx].set('pred_boxes', Boxes(bbox_xyxy))
    return pred_instances

def convert_condInst_to_densepose_inference(densepose_outputs: DensePoseChartPredictorOutput, 
    pred_instances: Instances, size) -> List[Instances]:
    pred_instances = constrain_bbox(pred_instances)
    S_list, I_list, U_list, V_list = [], [], [], []
    for idx in range(len(pred_instances)):
        ins = pred_instances[idx] 
        imgH, imgW = ins.image_size
        im_idx = ins.im_inds
        assert im_idx==0, "batch inference is not supported yet"
        bbox_xyxy = ins.pred_boxes.tensor
        socre = ins.scores

        S = densepose_outputs.coarse_segm[idx:idx+1]  #,:,y1:y2,x1:x2]
        logitH, logitW = S.shape[-2:]
        _, y1, _, y2 = (bbox_xyxy[0]*logitH/imgH).floor().int()
        x1, _, x2, _ = (bbox_xyxy[0]*logitW/imgW).floor().int()
        # pdb.set_trace()
        S_list.append(densepose_outputs.coarse_segm[idx:idx+1,:,y1:y2,x1:x2])
        I_list.append(densepose_outputs.fine_segm[im_idx:im_idx+1,:,y1:y2,x1:x2])
        U_list.append(densepose_outputs.u[im_idx:im_idx+1,:,y1:y2,x1:x2])
        V_list.append(densepose_outputs.v[im_idx:im_idx+1,:,y1:y2,x1:x2])
    # pdb.set_trace()
    # try:
    S_list = [F.interpolate(t,size=size,mode='bilinear') for t in S_list]
    I_list = [F.interpolate(t,size=size,mode='bilinear') for t in I_list]
    U_list = [F.interpolate(t,size=size,mode='bilinear') for t in U_list]
    V_list = [F.interpolate(t,size=size,mode='bilinear') for t in V_list]
    # if len(S_list)==0:
    #     pdb.set_trace()

    densepose_outputs = DensePoseChartPredictorOutput(
                                                        coarse_segm=torch.cat(S_list,dim=0),
                                                        fine_segm=torch.cat(I_list,dim=0),
                                                        u=torch.cat(U_list,dim=0),
                                                        v=torch.cat(V_list,dim=0),
                                                     )
    # pdb.set_trace()
    # import imageio
    # I = torch.argmax(I_list[0], dim=1)/24.0
    # imageio.imwrite("tmp/I0.png", I[0].detach().cpu().numpy())

    pred_instances.set('pred_densepose', densepose_outputs)
    return pred_instances


# def convert_condInst_to_densepose_inference_global(densepose_outputs: DensePoseChartPredictorOutput, 
#     pred_instances: Instances, size) -> List[Instances]:
#     S_list, I_list, U_list, V_list = [], [], [], []
#     S = torch.zeros_like(densepose_outputs.coarse_segm)
#     I = torch.zeros_like(densepose_outputs.coarse_segm)
#     S = torch.zeros_like(densepose_outputs.coarse_segm)
#     S = torch.zeros_like(densepose_outputs.coarse_segm)
#     for idx in range(len(pred_instances)):
#         ins = pred_instances[idx] 
#         imgH, imgW = ins.image_size
#         im_idx = ins.im_inds
#         assert im_idx==0, "batch inference is not supported yet"
#         bbox_xyxy = ins.pred_boxes.tensor
#         socre = ins.scores

#         S = densepose_outputs.coarse_segm[idx:idx+1]  #,:,y1:y2,x1:x2]
#         logitH, logitW = S.shape[-2:]
#         # pdb.set_trace()
#         x1, y1, x2, y2 = (bbox_xyxy[0]*logitH/imgH).floor().int()
#         # print(x1, y1, x2, y2)
#         # pdb.set_trace()
#         # try:
#         #     t = densepose_outputs.coarse_segm[idx:idx+1,:,y1:y2,x1:x2]
#         #     S_list.append(F.interpolate(t,size=size,mode='bilinear'))
#         # except:
#         #     pdb.set_trace()
#         S_list.append(densepose_outputs.coarse_segm[idx:idx+1,:,y1:y2,x1:x2])
#         I_list.append(densepose_outputs.fine_segm[im_idx:im_idx+1,:,y1:y2,x1:x2])
#         U_list.append(densepose_outputs.u[im_idx:im_idx+1,:,y1:y2,x1:x2])
#         V_list.append(densepose_outputs.v[im_idx:im_idx+1,:,y1:y2,x1:x2])
#     # S_list = [F.interpolate(t,size=size,mode='bilinear') for t in S_list]
#     # I_list = [F.interpolate(t,size=size,mode='bilinear') for t in I_list]
#     # U_list = [F.interpolate(t,size=size,mode='bilinear') for t in U_list]
#     # V_list = [F.interpolate(t,size=size,mode='bilinear') for t in V_list]

#     densepose_outputs = DensePoseChartPredictorOutput(
#                                                         coarse_segm=torch.cat(S_list,dim=0),
#                                                         fine_segm=torch.cat(I_list,dim=0),
#                                                         u=torch.cat(U_list,dim=0),
#                                                         v=torch.cat(V_list,dim=0),
#                                                      )
#     pred_instances.set('pred_densepose', densepose_outputs)


#         boxes = densepose_instances.pred_boxes.tensor
#         boxes = boxes/densepose_instances.image_size[0]*imgsize[0]
#         densepose_instances.set('pred_boxes', Boxes(boxes))

#     return pred_instances

def dice_coefficient(x, target):
    eps = 1e-5
    n_inst = x.size(0)
    x = x.reshape(n_inst, -1)
    target = target.reshape(n_inst, -1)
    intersection = (x * target).sum(dim=1)
    union = (x ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1) + eps
    loss = 1. - (2 * intersection / union)
    return loss


def parse_dynamic_params(params, channels, weight_nums, bias_nums, n_segm_chan):
    assert params.dim() == 2
    assert len(weight_nums) == len(bias_nums)
    assert params.size(1) == sum(weight_nums) + sum(bias_nums)

    num_insts = params.size(0)
    num_layers = len(weight_nums)

    params_splits = list(torch.split_with_sizes(
        params, weight_nums + bias_nums, dim=1
    ))

    weight_splits = params_splits[:num_layers]
    bias_splits = params_splits[num_layers:]

    for l in range(num_layers):
        if l < num_layers - 1:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
        else:
            # out_channels x in_channels x 1 x 1
            # pdb.set_trace()
            weight_splits[l] = weight_splits[l].reshape(num_insts * n_segm_chan, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts * n_segm_chan)

    return weight_splits, bias_splits


def build_dynamic_mask_head(cfg):
    return DynamicMaskHead(cfg)


class DynamicMaskHead(nn.Module):
    def __init__(self, cfg):
        super(DynamicMaskHead, self).__init__()
        self.num_layers = cfg.MODEL.CONDINST.MASK_HEAD.NUM_LAYERS
        self.channels = cfg.MODEL.CONDINST.MASK_HEAD.CHANNELS
        # self.in_channels = cfg.MODEL.CONDINST.MASK_BRANCH.OUT_CHANNELS
        self.in_channels = cfg.MODEL.CONDINST.MASK_HEAD.CHANNELS
        # MASK_HEAD.CHANNELS
        self.mask_out_stride = cfg.MODEL.CONDINST.MASK_OUT_STRIDE
        self.disable_rel_coords = cfg.MODEL.CONDINST.MASK_HEAD.DISABLE_REL_COORDS

        soi = cfg.MODEL.FCOS.SIZES_OF_INTEREST
        self.register_buffer("sizes_of_interest", torch.tensor(soi + [soi[-1] * 2]))


        # self.n_segm_chan = cfg.MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS
        self.n_segm_chan = 1
        # self.n_segm_chan = 2 + 1 ## S logit (of SIUV) has 2-channels, instance mask has 1-channel

        weight_nums, bias_nums = [], []
        for l in range(self.num_layers):
            if l == 0:
                if not self.disable_rel_coords:
                    weight_nums.append((self.in_channels + 2) * self.channels)
                else:
                    weight_nums.append(self.in_channels * self.channels)
                bias_nums.append(self.channels)
            elif l == self.num_layers - 1:
                # weight_nums.append(self.channels * 1)
                # bias_nums.append(1)
                weight_nums.append(self.channels * self.n_segm_chan)
                bias_nums.append(self.n_segm_chan)
            else:
                weight_nums.append(self.channels * self.channels)
                bias_nums.append(self.channels)

        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.num_gen_params = sum(weight_nums) + sum(bias_nums)



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
        ## DensePose related
        self.densepose_data_filter = build_densepose_data_filter(cfg)
        self.densepose_losses = build_densepose_losses(cfg)
        self.heatmap_size = cfg.MODEL.ROI_DENSEPOSE_HEAD.HEATMAP_SIZE
        # self.n_segm_chan  = cfg.MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS
        # self.n_segm_chan = 1
        self.segm_trained_by_masks = cfg.MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS
        self.w_segm       = cfg.MODEL.ROI_DENSEPOSE_HEAD.INDEX_WEIGHTS
        self.use_gt_ins = cfg.MODEL.CONDINST.IUVHead.GT_INSTANCES
        self.norm_coord_boxHW = cfg.MODEL.CONDINST.IUVHead.NORM_COORD_BOXHW
        self.dilate_ks = cfg.MODEL.CONDINST.IUVHead.DILATE_FGMASK_KENERAL_SIZE
        # self.use_ins_gn = cfg.MODEL.CONDINST.IUVHead.INSTANCE_AWARE_GN
        self.no_mask_overlap = cfg.MODEL.CONDINST.IUVHead.REMOVE_MASK_OVERLAP
        self.use_weight_std = cfg.MODEL.CONDINST.IUVHead.WEIGHT_STANDARDIZATION
        self.finetune_iuvhead_only = cfg.MODEL.CONDINST.FINETUNE_IUVHead_ONLY

        # self.amp_enable =  cfg.SOLVER.AMP.ENABLED
        # if self.amp_enable:
        #     self = self.half()
        ## debug
        # # if self.amp_enable:
        # # [p[1].data.dtype for p in self.named_parameters()]
        # for p in self.named_parameters():
        #     if p[1].data.dtype!=torch.float16:
        #         print(p[1].data.dtype)
        #         pdb.set_trace()

    def mask_heads_forward(self, features, weights, biases, num_insts):
        '''
        :param features
        :param weights: [w0, w1, ...]
        :param bias: [b0, b1, ...]
        :return:
        '''
        assert features.dim() == 4
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):

            ## Weight Standardization
            if self.use_weight_std:
                weight = w
                weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                          keepdim=True).mean(dim=3, keepdim=True)
                weight = weight - weight_mean
                std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
                weight = weight / std.expand_as(weight)
                w = weight

            x = F.conv2d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_insts
            )
            if i < n_layers - 1:
                x = F.relu(x)
        return x

    def mask_heads_forward_with_coords(
            self, mask_feats, mask_feat_stride, instances, norm_coord_boxHW=False
    ):
        locations = compute_locations(
            mask_feats.size(2), mask_feats.size(3),
            stride=mask_feat_stride, device=mask_feats.device
        )
        n_inst = len(instances)

        im_inds = instances.im_inds
        mask_head_params = instances.mask_head_params

        N, _, H, W = mask_feats.size()

        if not self.disable_rel_coords:
            instance_locations = instances.locations
            # pdb.set_trace()
            relative_coords = instance_locations.reshape(-1, 1, 2) - locations.reshape(1, -1, 2)
            relative_coords = relative_coords.permute(0, 2, 1).float()
            soi = self.sizes_of_interest.float()[instances.fpn_levels]
            # pdb.set_trace()
            relative_coords = relative_coords / soi.reshape(-1, 1, 1)
            relative_coords = relative_coords.to(dtype=mask_feats.dtype)

            mask_head_inputs = torch.cat([
                relative_coords, mask_feats[im_inds].reshape(n_inst, self.in_channels, H * W)
            ], dim=1)
        else:
            relative_coords = None
            mask_head_inputs = mask_feats[im_inds].reshape(n_inst, self.in_channels, H * W)

        mask_head_inputs = mask_head_inputs.reshape(1, -1, H, W)

        weights, biases = parse_dynamic_params(
            mask_head_params, self.channels,
            self.weight_nums, self.bias_nums,
            self.n_segm_chan
        )

        mask_logits = self.mask_heads_forward(mask_head_inputs, weights, biases, n_inst)

        # mask_logits = mask_logits.reshape(-1, 1, H, W)
        mask_logits = mask_logits.reshape(-1, self.n_segm_chan, H, W)

        # assert mask_feat_stride >= self.mask_out_stride
        # assert mask_feat_stride % self.mask_out_stride == 0
        # mask_logits = aligned_bilinear(mask_logits, int(mask_feat_stride / self.mask_out_stride))

        if norm_coord_boxHW:
            # pdb.set_trace()
            cnt = 0
            for idx in range(N):
                imgH, imgW = instances[idx].image_size
                # pdb.set_trace()
                boxes = instances[idx].pred_boxes.tensor
                for i in range(boxes.shape[0]):
                    boxH, boxW = boxes[i,3]-boxes[i,1], boxes[i,2]-boxes[i,0]
                    relative_coords[cnt+i,0:1] = relative_coords[cnt+i,0:1]*imgW/boxW
                    relative_coords[cnt+i,1:2] = relative_coords[cnt+i,1:2]*imgH/boxH
                cnt += boxes.shape[0]

        assert mask_feat_stride >= self.mask_out_stride
        assert mask_feat_stride % self.mask_out_stride == 0
        b,_,h,w = mask_logits.shape
        ratio = int(mask_feat_stride / self.mask_out_stride)
        mask_logits = aligned_bilinear(mask_logits, ratio)
        # pdb.set_trace()
        relative_coords = aligned_bilinear(relative_coords.reshape(b, 2, h,w), ratio).reshape(b, 2, -1) / ratio
        # relative_coords = aligned_bilinear(relative_coords.reshape(b, 2, h,w), ratio).reshape(b, 2, -1) * ratio


        return mask_logits, relative_coords
        # m = mask_logits[:,-1:,:,:]
        # return mask_logits[:,:-1,:,:], m.sigmoid()

    def _torch_dilate(self, binary_img, kernel_size=3, mode='nearest'):
        if kernel_size==0:
            return binary_img
        if not hasattr(self, 'dilate_kernel'):
            # self.dilate_kernel = torch.Tensor(torch.ones([kernel_size,kernel_size]), device=binary_img.device)[None,None,...]
            self.dilate_kernel = torch.ones([1,1,kernel_size,kernel_size], device=binary_img.device)
        # pdb.set_trace()
        pad = nn.ReflectionPad2d(int(kernel_size//2))
        out = torch.clamp(torch.nn.functional.conv2d(pad(binary_img), self.dilate_kernel, padding=0), 0, 1)
        out = F.interpolate(out, size=binary_img.shape[2:], mode=mode)
        return out

    def _torch_erode(self, binary_img, kernel_size=3, mode='nearest'):
        if kernel_size==0:
            return binary_img
        if not hasattr(self, 'erode_kernel'):
            # self.dilate_kernel = torch.Tensor(torch.ones([kernel_size,kernel_size]), device=binary_img.device)[None,None,...]
            self.erode_kernel = torch.ones([1,1,kernel_size,kernel_size], device=binary_img.device)
        # pdb.set_trace()
        pad = nn.ReflectionPad2d(int(kernel_size//2))
        out = torch.clamp(torch.nn.functional.conv2d(pad(1-binary_img), self.erode_kernel, padding=0), 0, 1)
        out = F.interpolate(out, size=binary_img.shape[2:], mode=mode)
        return 1-out

    def _create_rel_coord_gt(self, gt_instances, H, W, stride, device, norm_coord_boxHW=True, dilate_ks=0):
        
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

        gt_bitmasks = torch.cat([per_im.gt_bitmasks for per_im in gt_instances])
        gt_bitmasks = F.interpolate(gt_bitmasks[:,None,...].float(), size=(H,W), mode='nearest')
        gt_bitmasks = self._torch_erode(gt_bitmasks, kernel_size=dilate_ks)
        gt_bitmasks = self._torch_dilate(gt_bitmasks, kernel_size=dilate_ks*2)
        ins_mask_list = []

        rel_coord_gt = torch.zeros([N,2,H,W], device=device).float()
        coord_all = relative_coords.reshape(-1, 2, H, W) * gt_bitmasks
        cnt = 0


        for idx in range(N):
            num = gt_instances[idx].gt_bitmasks.shape[0]
            coord = torch.sum(coord_all[cnt:cnt+num], dim=0, keepdim=True) 
            rel_coord_gt[idx:idx+1] = coord #.reshape(1, 2, H, W)
            # if 0==gt_bitmasks[cnt:cnt+num,0].shape[0]:
            #     pdb.set_trace()
            ins_mask_list.append(gt_bitmasks[cnt:cnt+num,0])
            cnt += num
            # pdb.set_trace()
        return rel_coord_gt, ins_mask_list

    def remove_mask_overlap(self, ins_mask_list):
        for b in range(len(ins_mask_list)):
            fg_mask = ins_mask_list[b]
            num = fg_mask.shape[0]
            free_space = 1 - fg_mask[0:1]
            for i in range(1,num):
                fg_mask[i:i+1] = fg_mask[i:i+1] * free_space
                free_space = free_space - fg_mask[i:i+1]
            ins_mask_list[b] = fg_mask
        #     if torch.sum(fg_mask,dim=0).max()>1:
        #         pdb.set_trace()
        # if torch.sum(ins_mask_list[0],dim=0).max()>1:
        #     pdb.set_trace()
        # if torch.sum(ins_mask_list[1],dim=0).max()>1:
        #     pdb.set_trace()
        return ins_mask_list

    def __call__(self, iuv_head_func, fpn_features, mask_feats, iuv_feats, mask_feat_stride, pred_instances, 
            gt_instances=None, mask_out_bg_feats="none", pred_instances_nms=None):
        torch.cuda.empty_cache()
        if self.training:
            gt_inds = pred_instances.gt_inds
            gt_bitmasks = torch.cat([per_im.gt_bitmasks for per_im in gt_instances])
            gt_bitmasks = gt_bitmasks[gt_inds].unsqueeze(dim=1).to(dtype=mask_feats.dtype)
            # pdb.set_trace()
            # if pred_instances.labels.sum()>0:
            #     print("non-person instances exist")
            #     pdb.set_trace()

            losses = {}
            if len(pred_instances) == 0:
                loss_mask = mask_feats.sum() * 0 + pred_instances.mask_head_params.sum() * 0
                losses["loss_mask"] = loss_mask.float()
                losses["loss_densepose_I"] = mask_feats.sum() * 0
                losses["loss_densepose_U"] = mask_feats.sum() * 0
                losses["loss_densepose_V"] = mask_feats.sum() * 0
                losses["loss_densepose_S"] = mask_feats.sum() * 0
            else:
                if self.finetune_iuvhead_only:
                    with torch.no_grad():
                        s_logits, relative_coords = self.mask_heads_forward_with_coords(
                            mask_feats, mask_feat_stride, pred_instances
                        )
                else:
                    s_logits, relative_coords = self.mask_heads_forward_with_coords(
                        mask_feats, mask_feat_stride, pred_instances
                    )
                if self.n_segm_chan==1:
                    s_logits = s_logits.sigmoid()
                elif self.n_segm_chan==3:
                    s_logits = s_logits[:,:1].sigmoid()
                else:
                    raise NotImplementedError

                if iuv_head_func is None:
                    # assert mask_feat_stride >= self.mask_out_stride
                    # assert mask_feat_stride % self.mask_out_stride == 0
                    # s_logits = aligned_bilinear(s_logits, int(mask_feat_stride / self.mask_out_stride))
                    loss_mask = dice_coefficient(s_logits, gt_bitmasks)
                    losses["loss_densepose_S"] = loss_mask.mean() * self.w_segm
                else:
                    ## create_rel_coord_gt if needed
                    # H, W = s_logits.shape[-2:]
                    N, _, H, W = fpn_features[list(fpn_features.keys())[0]].shape
                    N_ins = s_logits.shape[0]
                    if self.use_gt_ins:
                        rel_coord, ins_mask_list = self._create_rel_coord_gt(gt_instances, H, W, self.mask_out_stride, 
                            s_logits.device, self.norm_coord_boxHW, self.dilate_ks)
                        # fg_mask = (rel_coord[:,0:1]!=0).float()
                        # if self.no_mask_overlap:
                        #     ins_mask_list = self.remove_mask_overlap(ins_mask_list)
                        # fg_mask = [ins_mask.sum(dim=0,keepdim=True) for ins_mask in ins_mask_list]
                        # fg_mask = torch.stack(fg_mask, dim=0)

                        # fg_mask = F.interpolate(fg_mask, (H,W))
                        # ins_mask_list = [per_im.gt_bitmasks for per_im in gt_instances]
                    else:


                        if self.finetune_iuvhead_only:
                            with torch.no_grad():
                                s_logits_keep, relative_coords_keep = self.mask_heads_forward_with_coords(
                                    mask_feats, mask_feat_stride, pred_instances_nms
                                )
                        else:
                            s_logits_keep, relative_coords_keep = self.mask_heads_forward_with_coords(
                                mask_feats, mask_feat_stride, pred_instances_nms
                            )

                        "TODO: upsample or recalculate rel_coord"
                        assert not self.norm_coord_boxHW
                        im_inds = pred_instances_nms.im_inds
                        rel_coord_list = []
                        pred_bitmasks = (s_logits_keep[:,-1:].detach()>0.05).float()
                        pred_bitmasks = F.interpolate(pred_bitmasks, (H,W))
                        pred_bitmasks = self._torch_erode(pred_bitmasks, kernel_size=self.dilate_ks)
                        pred_bitmasks = self._torch_dilate(pred_bitmasks, kernel_size=self.dilate_ks*2)
                        ins_mask_list = []
                        for idx in range(N):
                            if idx in im_inds:
                                # pdb.set_trace()
                                cc = relative_coords_keep[im_inds==idx,].reshape(-1, 2, H, W)
                                mm = pred_bitmasks[im_inds==idx,]
                                coord = torch.sum(cc*mm, dim=0, keepdim=True) 
                                rel_coord_list.append(coord) #.reshape(1, 2, H, W)
                                ins_mask_list.append(mm[:,0])
                        rel_coord = torch.cat(rel_coord_list, dim=0)
                    # pdb.set_trace()
                    if self.no_mask_overlap:
                        ins_mask_list = self.remove_mask_overlap(ins_mask_list)
                    fg_mask = [ins_mask.sum(dim=0,keepdim=True) for ins_mask in ins_mask_list]
                    fg_mask = torch.clamp(torch.stack(fg_mask, dim=0), min=0, max=1)
                    # pdb.set_trace()



                    # import imageio
                    # for ii in range(ins_mask_list[0].shape[0]):
                    #     imageio.imwrite("tmp/ins_mask_{}.png".format(ii), ins_mask_list[0][ii].detach().cpu().numpy())
                    # imageio.imwrite("tmp/fg_mask.png", fg_mask[0,0].detach().cpu().numpy())
                    # fg_mask2 = [ins_mask.sum(dim=0,keepdim=True) for ins_mask in ins_mask_list]
                    # fg_mask2 = torch.stack(fg_mask2, dim=0).float()
                    # imageio.imwrite("tmp/fg_mask2.png", (fg_mask2[0,0]/fg_mask2[0,0].max()).detach().cpu().numpy())
                    # pdb.set_trace()

                    ## remove overlap
                    # print(torch.sum(ins_mask_list[0],dim=0).max())
                    # for i in range(len(ins_mask_list)):
                    #     if ins_mask_list[i].sum()!=fg_mask[i].sum():
                    #         pdb.set_trace()

                    # print(ins_mask_list[0].sum())
                    # print(torch.sum(ins_mask_list[1]))

                    # gt_ins_mask_list = [per_im.gt_bitmasks for per_im in gt_instances]
                    # rel_coord_gt = self._create_rel_coord_gt(gt_instances, H, W, self.mask_out_stride, s_logits.device, self.norm_coord_boxHW)
                    ########## Debug
                    # pdb.set_trace()

                    # rel_coord = self._create_rel_coord_gt(gt_instances, H, W, self.mask_out_stride, s_logits.device)
                    # rel_coord_gt = rel_coord.clone()

                    # im_inds = pred_instances.im_inds
                    # rel_coord_list = []
                    # for idx in range(N_ins):
                    #     if idx in im_inds:
                    #         cc = relative_coords[im_inds==idx,].reshape(-1, 2, H, W)
                    #         ss = s_logits[im_inds==idx,-1:]
                    #         coord = torch.mean(cc*ss, dim=0, keepdim=True) 
                    #         rel_coord_list.append(coord) #.reshape(1, 2, H, W)
                    # rel_coord = torch.cat(rel_coord_list, dim=0)

                    # imageio.imwrite('tmp/rel_coord_gt.png',rel_coord_gt[0,0].detach().cpu().numpy())
                    # imageio.imwrite('tmp/rel_coord.png',rel_coord[0,0].detach().cpu().numpy())
                    # r = rel_coord_gt[0,0]
                    # g = rel_coord[0,0]
                    # b = torch.zeros_like(g)
                    # imageio.imwrite('tmp/rel_coord_gt_est.png',torch.stack([r,g,b],dim=-1).detach().cpu().numpy())
                    # pdb.set_trace()
                    ##########


                    if mask_out_bg_feats == "none":
                        fg_mask = torch.ones([N,1,H,W], device=iuv_feats.device)
                    # else:
                    #     # if self.use_ins_gn:
                    #     #     pass
                    #     # else:
                    #     # N = iuv_feats.shape[0]
                    #     # if mask_out_bg_feats=="hard":
                    #     fg_mask = (rel_coord[:,0:1]!=0).float()
                    #     # if self.use_gt_ins:
                    #     #     fg_mask = (rel_coord[:,0:1]!=0).float()
                    #     # else:
                    #     #     fg_mask = s_logits.detach()
                    #     #     fg_mask_list = []
                    #     #     for i in range(N):
                    #     #         fg_mask_list.append(torch.max(fg_mask[pred_instances.im_inds==i], dim=0, keepdim=True)[0])
                    #     #     # pdb.set_trace()
                    #     #     fg_mask = torch.cat(fg_mask_list, dim=0).detach()
                    #     #     fg_mask = (fg_mask>0.05).float()
                    #     fg_mask = F.interpolate(fg_mask, (H,W))
                    # pdb.set_trace()
                    # if self.dilate_ks>0:
                    #     fg_mask = self._torch_dilate(fg_mask, kernel_size=self.dilate_ks)

                            # pdb.set_trace()
                            # import imageio
                            # imageio.imwrite("tmp/fg_mask_soft.png", fg_mask_list[0][0,0].detach().cpu().numpy())
                            # imageio.imwrite("tmp/fg_mask_hard_0.1.png", (fg_mask_list[0][0,0]>0.1).float().detach().cpu().numpy())
                            # imageio.imwrite("tmp/fg_mask_hard_0.05.png", (fg_mask_list[0][0,0]>0.05).float().detach().cpu().numpy())
                            # imageio.imwrite("tmp/fg_mask_hard_0.05_dilate.png", fg_mask[0,0].detach().cpu().numpy())

                    # import imageio
                    # rel_coord_gt, ins_mask_gt_list = self._create_rel_coord_gt(gt_instances, H, W, self.mask_out_stride, s_logits.device, self.norm_coord_boxHW, self.dilate_ks)
                    # for ii in range(ins_mask_list[0].shape[0]):
                    #     imageio.imwrite("tmp/ins_mask_{}.png".format(ii), ins_mask_list[0][ii].detach().cpu().numpy())
                    # for ii in range(ins_mask_gt_list[0].shape[0]):
                    #     imageio.imwrite("tmp/ins_mask_gt_{}.png".format(ii), ins_mask_gt_list[0][ii].detach().cpu().numpy())
                    # imageio.imwrite("tmp/rel_coord.png", rel_coord[0,0].detach().cpu().numpy())
                    # imageio.imwrite("tmp/rel_coord_gt.png", rel_coord_gt[0,0].detach().cpu().numpy())
                    # imageio.imwrite("tmp/fg_mask.png", fg_mask[0,0].detach().cpu().numpy())
                    # fg_mask_gt = [ins_mask.sum(dim=0,keepdim=True) for ins_mask in ins_mask_gt_list]
                    # fg_mask_gt = torch.clamp(torch.stack(fg_mask_gt, dim=0), min=0, max=1)
                    # imageio.imwrite("tmp/fg_mask_gt.png", fg_mask_gt[0,0].detach().cpu().numpy())
                    # pdb.set_trace()
                    # imageio.imwrite("tmp/ins_mask.png", torch.sum(ins_mask_list[0],dim=0).detach().cpu().numpy())
                    # imageio.imwrite("tmp/ins_mask1.png", torch.sum(ins_mask_list[1],dim=0).detach().cpu().numpy())

                    # fg_mask_gt = (rel_coord_gt[:,0:1]!=0).float()
                    # fg_mask_gt = F.interpolate(fg_mask_gt, (H,W))
                    # imageio.imwrite("tmp/fg_mask_gt.png", fg_mask_gt[0,0].detach().cpu().numpy())

                    # if mask_out_bg_feats != "none":
                    #     iuv_logits = iuv_head_func(fpn_features, s_logits.detach(), iuv_feats*fg_mask, mask_feat_stride, rel_coord, pred_instances, fg_mask, gt_instances)
                    # else:
                    iuv_logits = iuv_head_func(fpn_features, s_logits.detach(), iuv_feats, mask_feat_stride, rel_coord, pred_instances, fg_mask, gt_instances, ins_mask_list)
                    
                    # iuv_logits = iuv_feats[:1,:75,:112,:112].expand_as(iuv_logits)

                    if isinstance(iuv_logits, list):
                        densepose_outputs = []
                        for iuv in iuv_logits:
                            densepose_outputs.append(DensePoseChartPredictorOutput(
                                                                                coarse_segm=s_logits,
                                                                                fine_segm=iuv[:,:25],
                                                                                u=iuv[:,25:50],
                                                                                v=iuv[:,50:75],
                                                                                aux_supervision=iuv[:,75:],
                                                                                stride=self.mask_out_stride,
                                                                             ))
                    else:
                        densepose_outputs = DensePoseChartPredictorOutput(
                                                                            coarse_segm=s_logits,
                                                                            fine_segm=iuv_logits[:,:25],
                                                                            u=iuv_logits[:,25:50],
                                                                            v=iuv_logits[:,50:75],
                                                                            aux_supervision=iuv_logits[:,75:],
                                                                            stride=self.mask_out_stride,
                                                                         )
                    for i in range(len(gt_instances)):
                        gt_instances[i].set('proposal_boxes', gt_instances[i].get('gt_boxes').clone())

                    # pdb.set_trace()
                    
                    _, gt_instances = self.densepose_data_filter(None, gt_instances)
                    # if len(gt_instances)!=len(proposals):
                    #     pdb.set_trace()
                    densepose_loss_dict = self.densepose_losses(
                        gt_instances, densepose_outputs, gt_bitmasks
                    )
                    losses.update(densepose_loss_dict)
            torch.cuda.empty_cache()
            return losses

        else:
            if len(pred_instances)>0 and iuv_head_func is not None:
                pred_instances = constrain_bbox(pred_instances)
                s_logits, relative_coords = self.mask_heads_forward_with_coords(
                    mask_feats, mask_feat_stride, pred_instances, norm_coord_boxHW=self.norm_coord_boxHW
                )

                # if self.n_segm_chan==1:
                #     s_logits = s_logits.sigmoid()
                # elif self.n_segm_chan==3:
                #     s_logits = s_logits[:,:1].sigmoid()
                # else:
                #     raise NotImplementedError

                if self.n_segm_chan==1:
                    "To mimic 2 channels segmentation during inference"
                    s_logits = s_logits.sigmoid()
                    s_logits = torch.cat([1-s_logits,s_logits],dim=1)
                elif self.n_segm_chan==3:
                    s_logits = s_logits[:,:1].sigmoid()
                    s_logits = torch.cat([1-s_logits,s_logits],dim=1)
                else:
                    raise NotImplementedError

                ## create_rel_coord_gt if needed
                # N, _, H, W = s_logits.shape[-2:]
                N, _, H, W = fpn_features[list(fpn_features.keys())[0]].shape

                N_ins = s_logits.shape[0]
                # if self.use_gt_ins:
                #     rel_coord = self._create_rel_coord_gt(gt_instances, H, W, self.mask_out_stride, s_logits.device)
                # else:

                "TODO: upsample or recalculate rel_coord"
                assert not self.norm_coord_boxHW
                im_inds = pred_instances.im_inds
                rel_coord_list = []
                fg_mask_list = []

                pred_bitmasks = (s_logits[:,-1:].detach()>0.05).float()
                pred_bitmasks = F.interpolate(pred_bitmasks, (H,W))
                pred_bitmasks = self._torch_erode(pred_bitmasks, kernel_size=self.dilate_ks)
                pred_bitmasks = self._torch_dilate(pred_bitmasks, kernel_size=self.dilate_ks*2)
                ins_mask_list = []
                for idx in range(N):
                    if idx in im_inds:
                        
                        cc = relative_coords[im_inds==idx,].reshape(-1, 2, H, W)
                        mm = pred_bitmasks[im_inds==idx,]
                        # cc = relative_coords[im_inds==idx,].reshape(-1, 2, s_logits.shape[-2]//2, s_logits.shape[-1]//2)
                        # ss = F.interpolate(s_logits, scale_factor=0.5)[im_inds==idx,-1:]
                        # pdb.set_trace()
                        coord = torch.sum(cc*mm, dim=0, keepdim=True) 
                        rel_coord_list.append(coord) #.reshape(1, 2, H, W)
                        fg_mask_list.append((torch.sum(mm, dim=0, keepdim=True)>0).float())
                        ins_mask_list.append(mm[:,0])
                rel_coord = torch.cat(rel_coord_list, dim=0)

                if self.no_mask_overlap:
                    ins_mask_list = self.remove_mask_overlap(ins_mask_list)
                fg_mask = [ins_mask.sum(dim=0,keepdim=True) for ins_mask in ins_mask_list]
                fg_mask = torch.clamp(torch.stack(fg_mask, dim=0), min=0, max=1)
                # pdb.set_trace()
                # rel_coord = torch.zeros([N,2,H,W], device=mask_feats.device).float()
                # im_inds = pred_instances.im_inds
                # for idx in range(N):
                #     if idx in im_inds:
                #         cc = relative_coords[im_inds==idx,].reshape(-1, 2, H, W)
                #         ss = s_logits[im_inds==idx,-1:]
                #         coord = torch.mean(cc*ss, dim=0, keepdim=True) 
                #         rel_coord[idx:idx+1] = coord #.reshape(1, 2, H, W)

                # import imageio
                # for ii in range(ins_mask_list[0].shape[0]):
                #     imageio.imwrite("tmp/ins_mask_{}.png".format(ii), ins_mask_list[0][ii].detach().cpu().numpy())
                # imageio.imwrite("tmp/fg_mask.png", fg_mask[0,0].detach().cpu().numpy())
                # fg_mask2 = [ins_mask.sum(dim=0,keepdim=True) for ins_mask in ins_mask_list]
                # fg_mask2 = torch.stack(fg_mask2, dim=0).float()
                # imageio.imwrite("tmp/fg_mask2.png", (fg_mask2[0,0]/fg_mask2[0,0].max()).detach().cpu().numpy())
                # pdb.set_trace()
                

                if mask_out_bg_feats == "none":
                    fg_mask = torch.ones([N,1,H,W], device=iuv_feats.device)
                # else:
                #     # N = iuv_feats.shape[0]
                #     fg_mask = s_logits[:,-1:].detach()
                #     fg_mask_list = []
                #     for i in range(N):
                #         fg_mask_list.append(torch.max(fg_mask[pred_instances.im_inds==i], dim=0, keepdim=True)[0])
                #     fg_mask = torch.cat(fg_mask_list, dim=0).detach()
                #     if mask_out_bg_feats=="hard":
                #         fg_mask = (fg_mask>0.1).float()
                #     fg_mask = F.interpolate(fg_mask, (H,W))
                # if self.dilate_ks>0:
                #     fg_mask = self._torch_dilate(fg_mask, kernel_size=self.dilate_ks)
                # fg_mask = self._torch_dilate(fg_mask, kernel_size=3)
                #     iuv_logits = iuv_head_func(s_logits.detach(), iuv_feats*fg_mask, mask_feat_stride, rel_coord, pred_instances)
                # else:
                iuv_logits = iuv_head_func(fpn_features, s_logits.detach(), iuv_feats, mask_feat_stride, rel_coord, pred_instances, fg_mask, ins_mask_list=ins_mask_list)
                
                # iuv_logits = iuv_head_func(s_logits, iuv_feats, mask_feat_stride, pred_instances)

                # assert mask_feat_stride >= self.mask_out_stride
                # assert mask_feat_stride % self.mask_out_stride == 0
                # s_logits = aligned_bilinear(s_logits, int(mask_feat_stride / self.mask_out_stride))

                ## multiscale aggregation during infereance (average)
                if isinstance(iuv_logits, list):
                    H = max([iuv.shape[-2] for iuv in iuv_logits])
                    W = max([iuv.shape[-1] for iuv in iuv_logits])
                    iuv_logits = [F.interpolate(iuv, size=(H,W)) for iuv in iuv_logits]
                    iuv_logits = torch.stack(iuv_logits, dim=0).mean(dim=0)

                densepose_outputs = DensePoseChartPredictorOutput(
                                                                    coarse_segm=iuv_logits[:,:2],
                                                                    fine_segm=iuv_logits[:,2:27],
                                                                    u=iuv_logits[:,27:52],
                                                                    v=iuv_logits[:,52:77],
                                                                 )
                pred_instances.set('pred_densepose', densepose_outputs)
                # pred_instances = convert_condInst_to_densepose_inference(densepose_outputs, 
                #                     pred_instances, size=(256,256))
            else:
                densepose_outputs = None
            torch.cuda.empty_cache()
            return pred_instances, densepose_outputs




    # def __call__(self, mask_feats, iuv_logits, mask_feat_stride, pred_instances, gt_instances=None):
    #     if self.training:
    #         gt_inds = pred_instances.gt_inds
    #         gt_bitmasks = torch.cat([per_im.gt_bitmasks for per_im in gt_instances])
    #         gt_bitmasks = gt_bitmasks[gt_inds].unsqueeze(dim=1).to(dtype=mask_feats.dtype)

    #         # gt_densepose_segm_list = []
    #         # for per_im in gt_instances:
    #         #     for t in per_im.gt_densepose:
    #         #         if t is None:
    #         #             pdb.set_trace()
    #         #         gt_densepose_segm_list.append(t.segm)
    #         # gt_densepose_segms = torch.stack(gt_densepose_segm_list, dim=0)
    #         # gt_densepose_segms = gt_densepose_segms[gt_inds].unsqueeze(dim=1).to(dtype=mask_feats.dtype)

    #         losses = {}
    #         if len(pred_instances) == 0:
    #             loss_mask = mask_feats.sum() * 0 + pred_instances.mask_head_params.sum() * 0
    #             losses["loss_mask"] = loss_mask.float()
    #             losses["loss_densepose_I"] = mask_feats.sum() * 0
    #             losses["loss_densepose_U"] = mask_feats.sum() * 0
    #             losses["loss_densepose_V"] = mask_feats.sum() * 0
    #             losses["loss_densepose_S"] = mask_feats.sum() * 0
    #         else:
    #             s_logits = self.mask_heads_forward_with_coords(
    #                 mask_feats, mask_feat_stride, pred_instances
    #             )
    #             # print(mask_scores.shape, gt_bitmasks.shape)
    #             # pdb.set_trace()
    #             # mask_losses = dice_coefficient(s_logits.sigmoid(), gt_bitmasks)
    #             # loss_mask = mask_losses.mean()
    #             # pdb.set_trace()

    #     # if not self.segm_trained_by_masks:
    #     #     if self.n_segm_chan == 2:
    #     #         s_gt = s_gt > 0
    #     #     s_loss = F.cross_entropy(s_est, s_gt.long()) * self.w_segm
    #     #     losses["loss_densepose_S"] = s_loss

    #             ## DensePose
    #             # densepose_outputs, _, confidences, _ = self.densepose_predictor(
    #             #     densepose_head_outputs
    #             # )
    #             # iuv_logits
    #             # densepose_outputs = [s_logits, iuv_logits[:,:25], iuv_logits[:,25:50], iuv_logits[:,50:75]]
    #             # # print('s_logits.shape:', s_logits.shape, 'iuv_logits.shape:', iuv_logits.shape)
    #             # # pdb.set_trace()
    #             # confidences = (None,None,None,None,None,None)


    #             # pdb.set_trace()
    #             if self.n_segm_chan==1:
    #                 s_logits = s_logits.sigmoid()
    #             elif self.n_segm_chan==3:
    #                 s_logits = s_logits[:,:1].sigmoid()
    #             else:
    #                 raise NotImplementedError
    #             densepose_outputs = DensePoseChartPredictorOutput(
    #                                                                 coarse_segm=s_logits,
    #                                                                 fine_segm=iuv_logits[:,:25],
    #                                                                 u=iuv_logits[:,25:50],
    #                                                                 v=iuv_logits[:,50:75],
    #                                                              )
    #             # proposal_boxes: Boxes
    #             # gt_boxes
    #             # gt_densepose
    #             # proposals_with_gt = []
    #             for i in range(len(gt_instances)):
    #                 gt_instances[i].set('proposal_boxes', gt_instances[i].get('gt_boxes').clone())

    #             # densepose_loss_dict = self.densepose_losses(
    #             #     gt_instances, densepose_outputs, confidences, bbox_free=True
    #             # )
    #             densepose_loss_dict = self.densepose_losses(
    #                 gt_instances, densepose_outputs, gt_bitmasks
    #             )
    #             losses.update(densepose_loss_dict)
    #             # losses["loss_mask"] = loss_mask.float()

    #         return losses

    #         # return loss_mask.float()
    #     else:
    #         if len(pred_instances) > 0:
    #             s_logits = self.mask_heads_forward_with_coords(
    #                 mask_feats, mask_feat_stride, pred_instances
    #             )

    #             if self.n_segm_chan==1:
    #                 "To mimic 2 channels segmentation during inference"
    #                 s_logits = s_logits.sigmoid()
    #                 # import imageio
    #                 # pdb.set_trace()
    #                 # ss = torch.cat(torch.split(s_logits, 1, dim=0), dim=-1)
    #                 # imageio.imwrite("tmp/s_logits_sigmoid_1chSeg.png", ss[0,0].detach().cpu().numpy())
    #                 s_logits = torch.cat([1-s_logits,s_logits],dim=1)
    #             elif self.n_segm_chan==3:
    #                 s_logits = s_logits[:,:1].sigmoid()
    #                 s_logits = torch.cat([1-s_logits,s_logits],dim=1)
    #             else:
    #                 raise NotImplementedError
    #             densepose_outputs = DensePoseChartPredictorOutput(
    #                                                                 coarse_segm=s_logits,
    #                                                                 fine_segm=iuv_logits[:,:25],
    #                                                                 u=iuv_logits[:,25:50],
    #                                                                 v=iuv_logits[:,50:75],
    #                                                              )
    #         else:
    #             densepose_outputs = None
    #         # pdb.set_trace()
    #         # densepose_inference(densepose_outputs, pred_instances)
    #         # pred_instances.set('pred_densepose', densepose_outputs)
    #         pred_instances = convert_condInst_to_densepose_inference(densepose_outputs, 
    #                             pred_instances, size=(256,256))
    #         # pred_instances = convert_condInst_to_densepose_inference_global(densepose_outputs, 
    #         #                     pred_instances, size=(self.heatmap_size,self.heatmap_size))
    #         return pred_instances, densepose_outputs









