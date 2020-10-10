# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import numpy as np
from typing import Dict, List, Optional
import fvcore.nn.weight_init as weight_init
import torch, pdb, os, pickle
import torch.nn as nn
from torch.nn import functional as F

from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads import select_foreground_proposals
from detectron2.structures import ImageList, Instances, Boxes

from .. import (
    build_densepose_data_filter,
    build_densepose_head,
    build_densepose_losses,
    build_densepose_predictor,
    densepose_inference,
)

import io, copy
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

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


@ROI_HEADS_REGISTRY.register()
class DensePoseROIHeads(StandardROIHeads):
    """
    A Standard ROIHeads which contains an addition of DensePose head.
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        self._init_densepose_head(cfg, input_shape)

    def _init_densepose_head(self, cfg, input_shape):
        # fmt: off
        self.densepose_on          = cfg.MODEL.DENSEPOSE_ON
        if not self.densepose_on:
            return
        self.densepose_data_filter = build_densepose_data_filter(cfg)
        dp_pooler_resolution       = cfg.MODEL.ROI_DENSEPOSE_HEAD.POOLER_RESOLUTION
        dp_pooler_sampling_ratio   = cfg.MODEL.ROI_DENSEPOSE_HEAD.POOLER_SAMPLING_RATIO
        dp_pooler_type             = cfg.MODEL.ROI_DENSEPOSE_HEAD.POOLER_TYPE
        self.use_decoder           = cfg.MODEL.ROI_DENSEPOSE_HEAD.DECODER_ON
        # fmt: on
        if self.use_decoder:
            dp_pooler_scales = (1.0 / input_shape[self.in_features[0]].stride,)
        else:
            dp_pooler_scales = tuple(1.0 / input_shape[k].stride for k in self.in_features)
        in_channels = [input_shape[f].channels for f in self.in_features][0]

        if self.use_decoder:
            self.decoder = Decoder(cfg, input_shape, self.in_features)

        self.densepose_pooler = ROIPooler(
            output_size=dp_pooler_resolution,
            scales=dp_pooler_scales,
            sampling_ratio=dp_pooler_sampling_ratio,
            pooler_type=dp_pooler_type,
        )
        self.densepose_head = build_densepose_head(cfg, in_channels)
        self.densepose_predictor = build_densepose_predictor(
            cfg, self.densepose_head.n_out_channels
        )
        self.densepose_losses = build_densepose_losses(cfg)

    def _forward_densepose(self, features: Dict[str, torch.Tensor], instances: List[Instances]):
        """
        Forward logic of the densepose prediction branch.
        Args:
            features (dict[str, Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            instances (list[Instances]): length `N` list of `Instances`. The i-th
                `Instances` contains instances for the i-th input image,
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.
        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "densepose" and return it.
        """
        if not self.densepose_on:
            return {} if self.training else instances

        features = [features[f] for f in self.in_features]
        if self.training:
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            features, proposals = self.densepose_data_filter(features, proposals)
            if len(proposals) > 0:
                proposal_boxes = [x.proposal_boxes for x in proposals]

                if self.use_decoder:
                    features = [self.decoder(features)]

                features_dp = self.densepose_pooler(features, proposal_boxes)
                densepose_head_outputs = self.densepose_head(features_dp)
                densepose_outputs, _, confidences, _ = self.densepose_predictor(
                    densepose_head_outputs
                )
                densepose_loss_dict = self.densepose_losses(
                    proposals, densepose_outputs, confidences
                )
                return densepose_loss_dict
        else:
            pred_boxes = [x.pred_boxes for x in instances]

            if self.use_decoder:
                features = [self.decoder(features)]

            features_dp = self.densepose_pooler(features, pred_boxes)
            if len(features_dp) > 0:
                densepose_head_outputs = self.densepose_head(features_dp)
                densepose_outputs, _, confidences, _ = self.densepose_predictor(
                    densepose_head_outputs
                )
            else:
                # If no detection occurred instances
                # set densepose_outputs to empty tensors
                empty_tensor = torch.zeros(size=(0, 0, 0, 0), device=features_dp.device)
                densepose_outputs = tuple([empty_tensor] * 4)
                confidences = tuple([empty_tensor] * 6)

            densepose_inference(densepose_outputs, confidences, instances)
            return instances

    def _forward_densepose_smooth_save(self, features: Dict[str, torch.Tensor], instances: List[Instances]):
        """
        Forward logic of the densepose prediction branch.

        Args:
            features (dict[str, Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            instances (list[Instances]): length `N` list of `Instances`. The i-th
                `Instances` contains instances for the i-th input image,
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "densepose" and return it.
        """
        if not self.densepose_on:
            return {} if self.training else instances

        ## MLQ added
        assert not self.training
        self._register_hooks()
        self.cnt = 1
        self.smooth_k = cfg.SMOOTH_K
        self.prev_instances = None
        # self.data_dir = "/esat/dragon/liqianma/datasets/Pose/youtube/youtube_single"
        # self.data_dir = "/esat/dragon/liqianma/datasets/Pose/youtube/liqian01"
        self.data_dir = cfg.DATA_DIR
        print("--> data_dir: ", self.data_dir)
        self.in_dir = os.path.join(self.data_dir, "DP_fea")
        if self.smooth_k>0 and os.path.exists(self.in_dir) and len(os.listdir(self.in_dir))>0:
            self.out_dir = os.path.join(self.data_dir, "DP_fea_smooth{}".format(self.smooth_k))
        else:
            self.out_dir = os.path.join(self.data_dir, "DP_fea")
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        pred_boxes = [x.pred_boxes for x in instances]
        scores = [x.scores for x in instances]
        # pdb.set_trace()
        if self.smooth_k>0:
            pred_boxes, idx = self._smooth_bbox(self.in_dir, self.cnt, self.smooth_k, single_person=True)
            
            for i in range(len(instances)):
                if len(instances[i])==0:
                    instances = copy.copy(self.prev_instances)
                    pred_boxes = [instances[i].pred_boxes]
                elif len(instances[i])>1:
                    try:
                        instances[i] = instances[i][idx.item()]
                    except:
                        print(idx)
                        instances[i] = instances[i][idx]
                    instances[i].pred_boxes = pred_boxes[i]
                else:
                    instances[i].pred_boxes = pred_boxes[i]
                # except:
                #     pdb.set_trace()
        self.prev_instances = copy.copy(instances)

        if self.use_decoder:
            features = [self.decoder(features)]

        "TODO: (1) smooth the pred_boxes with t+-1, save all bbox and load for (track) smooth;" 
        "TODO: (2) save densepose_outputs, confidences"
        "TODO: (3) track bbox for multi-person via densepose similarity"
        features_dp = self.densepose_pooler(features, pred_boxes)
        if len(features_dp) > 0:
            densepose_head_outputs = self.densepose_head(features_dp)
            densepose_outputs, _, confidences, _ = self.densepose_predictor(
                densepose_head_outputs
            )
        else:
            # If no detection occurred instances
            # set densepose_outputs to empty tensors
            empty_tensor = torch.zeros(size=(0, 0, 0, 0), device=features_dp.device)
            densepose_outputs = tuple([empty_tensor] * 4)
            confidences = tuple([empty_tensor] * 6)

        # pdb.set_trace()
        # out_dict = {"pred_boxes":pred_boxes, "densepose_outputs":densepose_outputs,
        #             "confidences":confidences, "scores":scores}
        # pdb.set_trace()
        out_dict = {"pred_boxes":self.to_cpu(pred_boxes), 
                    "densepose_outputs":self.to_cpu(densepose_outputs),
                    "confidences":self.to_cpu(confidences), 
                    "scores":self.to_cpu(scores),
                    "height":instances[0].image_size[0],
                    "width":instances[0].image_size[1],
                    "instances":instances}
        # pdb.set_trace()
        path = os.path.join(self.out_dir, "frame_{:06d}.pkl".format(self.cnt))
        pickle.dump(out_dict, open(path,"wb"))
        self.cnt += 1

        densepose_inference(densepose_outputs, confidences, instances)
        return instances

    def to_cpu(self, obj_list):
        obj_list = list(obj_list)
        for i in range(len(obj_list)):
            try:
                obj_list[i] = obj_list[i].to("cpu")
            except:
                pass
        return obj_list

    def to_cuda(self, obj_list):
        obj_list = list(obj_list)
        for i in range(len(obj_list)):
            try:
                obj_list[i] = obj_list[i].to("cuda")
            except:
                pass
        return obj_list

    ## MLQ added
    def _smooth_bbox(self, data_dir, fid, k=3, single_person=True, score_threshold=0.8):
        bbox_tensor_list = []
        fid_max = len(os.listdir(data_dir))
        # if fid_max==0:
        #     return None

        for fid_delta in range(-k,k+1):
            fid_i = min(fid_max, max(1,fid+fid_delta))
            path = os.path.join(data_dir, "frame_{:06d}.pkl".format(fid_i))
            # if torch.cuda.is_available():
            #     data = pickle.load(open(path,"rb"))
            # else:
            #     data = CPU_Unpickler(open(path,"rb")).load()
            if torch.cuda.is_available():
                data = pickle.load(open(os.path.join(data_dir, "frame_{:06d}.pkl".format(fid_i)),"rb"))
                for k in data.keys():
                    try:
                        data[k] = self.to_cuda(data[k])
                    except:
                        pass
            else:
                data = pickle.load(open(path,"rb"))
                # with torch.loading_context(map_location='cpu'):
                #     data = pickle.load(open(path,"rb")) # In my case this call is buried deeper in torch-agnostic code

            boxes = data["pred_boxes"][0]
            scores = data["scores"][0]
            if single_person:
                idx = torch.argmax(scores) if scores.shape[0]>1 else 0
                boxes_tensor = boxes.tensor[idx:idx+1]
            else:
                "TODO: multi-object association"
                pass
            if boxes_tensor.shape[0]>0:
                bbox_tensor_list.append(boxes_tensor)
        # pdb.set_trace()
        if bbox_tensor_list==[]:
            return None, None
        else:
            xywh = torch.stack(bbox_tensor_list, dim=-1) ## xywh in Nx4x(2k+1) shape
            xywh_smooth = torch.zeros_like(xywh[:,:,0])
            xywh_smooth[:,0] = torch.min(xywh[:,0], dim=-1)[0]
            xywh_smooth[:,1] = torch.min(xywh[:,1], dim=-1)[0]
            xywh_smooth[:,2] = torch.max(xywh[:,2], dim=-1)[0]
            xywh_smooth[:,3] = torch.max(xywh[:,3], dim=-1)[0]
            return [Boxes(xywh_smooth)], idx


    ## MLQ added
    def _register_hooks(self):
        ## Ref:https://gist.github.com/Tushar-N/680633ec18f5cb4b47933da7d10902af
        # Registering hooks for all the Conv2d layers
        # Note: Hooks are called EVERY TIME the module performs a forward pass. For modules that are
        # called repeatedly at different stages of the forward pass (like RELUs), this will save different
        # activations. Editing the forward pass code to save activations is the way to go for these cases.
        # a dictionary that keeps saving the activations as they come
        # self.activations = collections.defaultdict(list)
        from functools import partial
        import collections
        # def save_activation(name, mod, inp, out):
        #     self.activations[name].append(out)
        self.activations = collections.defaultdict()
        def save_activation(name, mod, inp, out):
            self.activations[name] = out

        for name, m in self.densepose_head.named_modules():
            if type(m)==nn.Conv2d:
                # partial to assign the layer name to each hook
                m.register_forward_hook(partial(save_activation, name))

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ):
        instances, losses = super().forward(images, features, proposals, targets)
        del targets, images

        if self.training:
            losses.update(self._forward_densepose(features, instances))
        return instances, losses

    def forward_with_given_boxes(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ):
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        This is useful for downstream tasks where a box is known, but need to obtain
        other attributes (outputs of other heads).
        Test-time augmentation also uses this.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (list[Instances]):
                the same `Instances` objects, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """

        instances = super().forward_with_given_boxes(features, instances)
        instances = self._forward_densepose(features, instances) ## original inference
        # instances = self._forward_densepose_smooth_save(features, instances) ## MLQ modified

        return instances
