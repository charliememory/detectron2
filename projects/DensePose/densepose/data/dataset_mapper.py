# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import copy, pdb
import logging
from typing import Any, Dict, Tuple, List
import torch
from fvcore.common.file_io import PathManager
import numpy as np

from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.layers import ROIAlign
from detectron2.structures import BoxMode

from .structures import DensePoseDataRelative, DensePoseList, DensePoseTransformData

from .skeleton_feat import genSkeletons


def build_augmentation(cfg, is_train):
    logger = logging.getLogger(__name__)
    result = utils.build_augmentation(cfg, is_train)
    if is_train:
        random_rotation = T.RandomRotation(
            cfg.INPUT.ROTATION_ANGLES, expand=False, sample_style="choice"
        )
        result.append(random_rotation)
        logger.info("DensePose-specific augmentation used in training: " + str(random_rotation))
    return result


class DatasetMapper:
    """
    A customized version of `detectron2.data.DatasetMapper`
    """

    def __init__(self, cfg, is_train=True):
        self.augmentation = build_augmentation(cfg, is_train)

        # fmt: off
        self.img_format     = cfg.INPUT.FORMAT
        self.mask_on        = (
            cfg.MODEL.MASK_ON or (
                cfg.MODEL.DENSEPOSE_ON
                and cfg.MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS)
        )
        self.keypoint_on    = cfg.MODEL.KEYPOINT_ON
        self.densepose_on   = cfg.MODEL.DENSEPOSE_ON
        assert not cfg.MODEL.LOAD_PROPOSALS, "not supported yet"
        # fmt: on
        if self.keypoint_on and is_train:
            # Flip only makes sense in training
            self.keypoint_hflip_indices = utils.create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)
        else:
            self.keypoint_hflip_indices = None

        if self.densepose_on:
            densepose_transform_srcs = [
                MetadataCatalog.get(ds).densepose_transform_src
                for ds in cfg.DATASETS.TRAIN + cfg.DATASETS.TEST
            ]
            # pdb.set_trace()
            # densepose_transform_srcs = []
            # for ds in cfg.DATASETS.TRAIN + cfg.DATASETS.TEST:
            #     try:
            #         ts = MetadataCatalog.get(ds).densepose_transform_src
            #     except:
            #         ts = 'https://dl.fbaipublicfiles.com/densepose/data/UV_symmetry_transforms.mat'
            #     densepose_transform_srcs.append(ts)
            assert len(densepose_transform_srcs) > 0
            # TODO: check that DensePose transformation data is the same for
            # all the datasets. Otherwise one would have to pass DB ID with
            # each entry to select proper transformation data. For now, since
            # all DensePose annotated data uses the same data semantics, we
            # omit this check.
            densepose_transform_data_fpath = PathManager.get_local_path(densepose_transform_srcs[0])
            self.densepose_transform_data = DensePoseTransformData.load(
                densepose_transform_data_fpath
            )

        self.is_train = is_train
        self.use_gt_ins = cfg.MODEL.CONDINST.IUVHead.GT_INSTANCES
        self.mask_out_stride = cfg.MODEL.CONDINST.MASK_OUT_STRIDE
        self.use_gt_skeleton = cfg.MODEL.CONDINST.IUVHead.GT_SKELETON
        if self.use_gt_skeleton:
            self.keypoint_on = True

        self.infer_smooth_frame_num = cfg.MODEL.INFERENCE_SMOOTH_FRAME_NUM

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        image, transforms = T.apply_transform_gens(self.augmentation, image)
        image_shape = image.shape[:2]  # h, w
        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

        # pdb.set_trace()
        if self.infer_smooth_frame_num>0:
            # assert "frame_" in dataset_dict["file_name"], "we use ## to separate name and frame num"
            if "frame_" in dataset_dict["file_name"]:
                ## filename format from ffmpeg
                name, fid = dataset_dict["file_name"].split("frame_")
                name = name + "frame_"
            else:
                ## filename format from posetrack
                split_list = dataset_dict["file_name"].split("/")
                name = "/".join(split_list[:-1])
                name = name + "/"
                fid = split_list[-1]

            # num_len = len(fid)
            # file_name = dataset_dict["file_name"]
            image_adj_list = []
            for i in range(self.infer_smooth_frame_num):
                fid_adj = str(max(0,int(fid)-i))
                fid_adj = "".join(["0"]*(len(fid)-len(fid_adj))) + fid_adj
                file_name = name + fid_adj
                try:
                    image_adj = utils.read_image(file_name, format=self.img_format)
                    image_adj, _ = T.apply_transform_gens(self.augmentation, image_adj)
                except:
                    print("image does not exist:", file_name)
                    image_adj = image
                image_adj_list.append(torch.as_tensor(image_adj.transpose(2, 0, 1).astype("float32")))
            dataset_dict["image_adj_list"] = image_adj_list


        # if not self.is_train:
        if not self.is_train and not self.use_gt_ins and not self.use_gt_skeleton:
            dataset_dict.pop("annotations", None)
            return dataset_dict

        for anno in dataset_dict["annotations"]:
            if not self.mask_on:
                anno.pop("segmentation", None)
            if not self.keypoint_on:
                anno.pop("keypoints", None)

        # USER: Implement additional transformations if you have other types of data
        # USER: Don't call transpose_densepose if you don't need
        annos = [
            self._transform_densepose(
                utils.transform_instance_annotations(
                    obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                ),
                transforms,
            )
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]

        if self.mask_on:
            self._add_densepose_masks_as_segmentation(annos, image_shape)

        instances = utils.annotations_to_instances(annos, image_shape, mask_format="bitmask")
        densepose_annotations = [obj.get("densepose") for obj in annos]
        if densepose_annotations and not all(v is None for v in densepose_annotations):
            instances.gt_densepose = DensePoseList(
                densepose_annotations, instances.gt_boxes, image_shape
            )

        if self.keypoint_on:
            dataset_dict["skeleton_feat"] = self._get_skeleton_feat(annos, image_shape)
            # if ske_feat is not None:
            # dataset_dict["skeleton_feats"] = ske_feats

        dataset_dict["instances"] = instances[instances.gt_boxes.nonempty()]
        return dataset_dict

    "TODO"
    # def _add_skeleton_feat(
    #     self, annotations: Dict[str, Any], image_shape_hw: Tuple[int, int]
    # ):
    def _get_skeleton_feat(self, annotation, image_shape):
        "TODO"
        ske_feat_all = None
        cnt = 0
        for idx, obj in enumerate(annotation):
            kpts = obj['keypoints']
            if kpts.sum()!=0:
                # assert obj['densepose'] is not None
                ske_feat = genSkeletons(kpts[None,...], image_shape[0], image_shape[1], 
                        stride=self.mask_out_stride, sigma=7, threshold=1, visdiff = True).transpose(2, 0, 1)
                # ske_feat = genSkeletons(kpts[None,...], image_shape[0], image_shape[1], 
                #         stride=1, sigma=5, threshold=3, visdiff = True).transpose(2, 0, 1)
                # obj["skeleton_feat"] = torch.from_numpy(ske_feat)
            # else:
                # obj["skeleton_feat"] = None
                if ske_feat_all is None:
                    ske_feat_all = ske_feat
                else:
                    ske_feat_all += ske_feat
                cnt += (ske_feat!=0).astype(np.float)

        if ske_feat_all is not None:
            ske_feat_all = ske_feat_all/(cnt+1e-5)
            return torch.from_numpy(ske_feat_all).float()
        else:
            return None

        # if ske_feat_all is not None:
        #     ske_feat_all = ske_feat_all/(cnt+1e-5)
        #     # import imageio
        #     # h, w = ske_feat_all.shape[-2:]
        #     # paf_x = ske_feat_all[-38:].reshape([19,2,h,w])[:,0]
        #     # paf_y = ske_feat_all[-38:].reshape([19,2,h,w])[:,1]
        #     # imageio.imwrite("tmp/ske_feat.png", ske_feat_all.sum(0))
        #     # imageio.imwrite("tmp/paf_x.png", paf_x.sum(0))
        #     # imageio.imwrite("tmp/paf_y.png", paf_y.sum(0))
        #     pdb.set_trace()

    def _transform_densepose(self, annotation, transforms):
        if not self.densepose_on:
            return annotation

        # Handle densepose annotations
        is_valid, reason_not_valid = DensePoseDataRelative.validate_annotation(annotation)
        if is_valid:
            densepose_data = DensePoseDataRelative(annotation, cleanup=True)
            densepose_data.apply_transform(transforms, self.densepose_transform_data)
            annotation["densepose"] = densepose_data
        else:
            # logger = logging.getLogger(__name__)
            # logger.debug("Could not load DensePose annotation: {}".format(reason_not_valid))
            DensePoseDataRelative.cleanup_annotation(annotation)
            # NOTE: annotations for certain instances may be unavailable.
            # 'None' is accepted by the DensePostList data structure.
            annotation["densepose"] = None
        return annotation

    def _add_densepose_masks_as_segmentation(
        self, annotations: List[Any], image_shape_hw: Tuple[int, int]
    ):
        for obj in annotations:
            if ("densepose" not in obj) or ("segmentation" in obj):
                continue
            # DP segmentation: torch.Tensor [S, S] of float32, S=256
            segm_dp = torch.zeros_like(obj["densepose"].segm)
            segm_dp[obj["densepose"].segm > 0] = 1
            segm_h, segm_w = segm_dp.shape
            bbox_segm_dp = torch.tensor((0, 0, segm_h - 1, segm_w - 1), dtype=torch.float32)
            # image bbox
            x0, y0, x1, y1 = (
                v.item() for v in BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS)
            )
            segm_aligned = (
                ROIAlign((y1 - y0, x1 - x0), 1.0, 0, aligned=True)
                .forward(segm_dp.view(1, 1, *segm_dp.shape), bbox_segm_dp)
                .squeeze()
            )
            image_mask = torch.zeros(*image_shape_hw, dtype=torch.float32)
            image_mask[y0:y1, x0:x1] = segm_aligned
            # segmentation for BitMask: np.array [H, W] of np.bool
            obj["segmentation"] = image_mask >= 0.5
