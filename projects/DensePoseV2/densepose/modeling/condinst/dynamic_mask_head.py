from typing import Any, List
import torch
from torch.nn import functional as F
from torch import nn

from detectron2.structures import Instances
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
import pdb


def convert_condInst_to_densepose_inference(densepose_outputs: DensePoseChartPredictorOutput, 
    pred_instances: Instances, size) -> List[Instances]:
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
        # pdb.set_trace()
        x1, y1, x2, y2 = (bbox_xyxy[0]*logitH/imgH).round().int()
        # print(x1, y1, x2, y2)
        # pdb.set_trace()
        # try:
        #     t = densepose_outputs.coarse_segm[idx:idx+1,:,y1:y2,x1:x2]
        #     S_list.append(F.interpolate(t,size=size,mode='bilinear'))
        # except:
        #     pdb.set_trace()
        S_list.append(densepose_outputs.coarse_segm[idx:idx+1,:,y1:y2,x1:x2])
        I_list.append(densepose_outputs.fine_segm[im_idx:im_idx+1,:,y1:y2,x1:x2])
        U_list.append(densepose_outputs.u[im_idx:im_idx+1,:,y1:y2,x1:x2])
        V_list.append(densepose_outputs.v[im_idx:im_idx+1,:,y1:y2,x1:x2])
    S_list = [F.interpolate(t,size=size,mode='bilinear') for t in S_list]
    I_list = [F.interpolate(t,size=size,mode='bilinear') for t in I_list]
    U_list = [F.interpolate(t,size=size,mode='bilinear') for t in U_list]
    V_list = [F.interpolate(t,size=size,mode='bilinear') for t in V_list]

    densepose_outputs = DensePoseChartPredictorOutput(
                                                        coarse_segm=torch.cat(S_list,dim=0),
                                                        fine_segm=torch.cat(I_list,dim=0),
                                                        u=torch.cat(U_list,dim=0),
                                                        v=torch.cat(V_list,dim=0),
                                                     )
    pred_instances.set('pred_densepose', densepose_outputs)
    return pred_instances


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


        self.n_segm_chan = cfg.MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS
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
        self.densepose_losses = build_densepose_losses(cfg)
        self.heatmap_size = cfg.MODEL.ROI_DENSEPOSE_HEAD.HEATMAP_SIZE
        self.n_segm_chan  = cfg.MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS
        self.segm_trained_by_masks = cfg.MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS
        self.w_segm       = cfg.MODEL.ROI_DENSEPOSE_HEAD.INDEX_WEIGHTS

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
            x = F.conv2d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_insts
            )
            if i < n_layers - 1:
                x = F.relu(x)
        return x

    def mask_heads_forward_with_coords(
            self, mask_feats, mask_feat_stride, instances
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
            relative_coords = instance_locations.reshape(-1, 1, 2) - locations.reshape(1, -1, 2)
            relative_coords = relative_coords.permute(0, 2, 1).float()
            soi = self.sizes_of_interest.float()[instances.fpn_levels]
            relative_coords = relative_coords / soi.reshape(-1, 1, 1)
            relative_coords = relative_coords.to(dtype=mask_feats.dtype)

            mask_head_inputs = torch.cat([
                relative_coords, mask_feats[im_inds].reshape(n_inst, self.in_channels, H * W)
            ], dim=1)
        else:
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

        assert mask_feat_stride >= self.mask_out_stride
        assert mask_feat_stride % self.mask_out_stride == 0
        mask_logits = aligned_bilinear(mask_logits, int(mask_feat_stride / self.mask_out_stride))

        # return mask_logits.sigmoid()
        # return mask_logits[:,:-1,:,:], mask_logits[:,-1:,:,:].sigmoid()
        # if self.n_segm_chan==1:
        #     mask_logits = mask_logits.sigmoid()
        #     return mask_logits
        # elif self.n_segm_chan==2:
        #     return mask_logits
        # else:
        #     raise NotImplementedError
        return mask_logits
        # m = mask_logits[:,-1:,:,:]
        # return mask_logits[:,:-1,:,:], m.sigmoid()

    def __call__(self, mask_feats, iuv_logits, mask_feat_stride, pred_instances, gt_instances=None):
        if self.training:
            gt_inds = pred_instances.gt_inds
            gt_bitmasks = torch.cat([per_im.gt_bitmasks for per_im in gt_instances])
            gt_bitmasks = gt_bitmasks[gt_inds].unsqueeze(dim=1).to(dtype=mask_feats.dtype)

            # gt_densepose_segm_list = []
            # for per_im in gt_instances:
            #     for t in per_im.gt_densepose:
            #         if t is None:
            #             pdb.set_trace()
            #         gt_densepose_segm_list.append(t.segm)
            # gt_densepose_segms = torch.stack(gt_densepose_segm_list, dim=0)
            # gt_densepose_segms = gt_densepose_segms[gt_inds].unsqueeze(dim=1).to(dtype=mask_feats.dtype)

            losses = {}
            if len(pred_instances) == 0:
                loss_mask = mask_feats.sum() * 0 + pred_instances.mask_head_params.sum() * 0
                losses["loss_mask"] = loss_mask.float()
                losses["loss_densepose_I"] = mask_feats.sum() * 0
                losses["loss_densepose_U"] = mask_feats.sum() * 0
                losses["loss_densepose_V"] = mask_feats.sum() * 0
                losses["loss_densepose_S"] = mask_feats.sum() * 0
            else:
                s_logits = self.mask_heads_forward_with_coords(
                    mask_feats, mask_feat_stride, pred_instances
                )
                # print(mask_scores.shape, gt_bitmasks.shape)
                # pdb.set_trace()
                # mask_losses = dice_coefficient(s_logits.sigmoid(), gt_bitmasks)
                # loss_mask = mask_losses.mean()
                # pdb.set_trace()

        # if not self.segm_trained_by_masks:
        #     if self.n_segm_chan == 2:
        #         s_gt = s_gt > 0
        #     s_loss = F.cross_entropy(s_est, s_gt.long()) * self.w_segm
        #     losses["loss_densepose_S"] = s_loss

                ## DensePose
                # densepose_outputs, _, confidences, _ = self.densepose_predictor(
                #     densepose_head_outputs
                # )
                # iuv_logits
                # densepose_outputs = [s_logits, iuv_logits[:,:25], iuv_logits[:,25:50], iuv_logits[:,50:75]]
                # # print('s_logits.shape:', s_logits.shape, 'iuv_logits.shape:', iuv_logits.shape)
                # # pdb.set_trace()
                # confidences = (None,None,None,None,None,None)


                # pdb.set_trace()
                if self.n_segm_chan==1:
                    s_logits = s_logits.sigmoid()
                densepose_outputs = DensePoseChartPredictorOutput(
                                                                    coarse_segm=s_logits,
                                                                    fine_segm=iuv_logits[:,:25],
                                                                    u=iuv_logits[:,25:50],
                                                                    v=iuv_logits[:,50:75],
                                                                 )
                # proposal_boxes: Boxes
                # gt_boxes
                # gt_densepose
                # proposals_with_gt = []
                for i in range(len(gt_instances)):
                    gt_instances[i].set('proposal_boxes', gt_instances[i].get('gt_boxes').clone())

                # densepose_loss_dict = self.densepose_losses(
                #     gt_instances, densepose_outputs, confidences, bbox_free=True
                # )
                densepose_loss_dict = self.densepose_losses(
                    gt_instances, densepose_outputs, gt_bitmasks
                )
                losses.update(densepose_loss_dict)
                # losses["loss_mask"] = loss_mask.float()

            return losses

            # return loss_mask.float()
        else:
            if len(pred_instances) > 0:
                s_logits = self.mask_heads_forward_with_coords(
                    mask_feats, mask_feat_stride, pred_instances
                )

                if self.n_segm_chan==1:
                    "To mimic 2 channels segmentation during inference"
                    s_logits = s_logits.sigmoid()
                    s_logits = torch.cat([1-s_logits,s_logits],dim=1)
                densepose_outputs = DensePoseChartPredictorOutput(
                                                                    coarse_segm=s_logits,
                                                                    fine_segm=iuv_logits[:,:25],
                                                                    u=iuv_logits[:,25:50],
                                                                    v=iuv_logits[:,50:75],
                                                                 )
            else:
                densepose_outputs = None
            # pdb.set_trace()
            # densepose_inference(densepose_outputs, pred_instances)
            pred_instances = convert_condInst_to_densepose_inference(densepose_outputs, 
                                pred_instances, size=(self.heatmap_size,self.heatmap_size))
            return pred_instances









