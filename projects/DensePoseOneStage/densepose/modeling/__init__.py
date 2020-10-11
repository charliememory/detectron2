# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from .confidence import DensePoseConfidenceModelConfig, DensePoseUVConfidenceType
from .filter import DensePoseDataFilter
from .inference import densepose_inference
from .utils import initialize_module_params
from .build import (
    build_densepose_data_filter,
    build_densepose_head,
    build_densepose_losses,
    build_densepose_predictor,
)


# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import pdb
# Just to register models from adet (not sure why there is an error about "already registered")
try:
	from .fcos import FCOS  
except Exception as e:
	print(e)
# from .blendmask import BlendMask
# try:
# 	from .backbone import build_fcos_resnet_fpn_backbone
# except Exception as e:
# 	print(e)
# from .one_stage_detector import OneStageDetector, OneStageRCNN
# from .roi_heads.text_head import TextHead
# from .batext import BAText
# from .MEInst import MEInst

# try:
# 	from .condinst import condinst
# except Exception as e:
# 	print(e)
# pdb.set_trace()

_EXCLUDE = {"torch", "ShapeSpec"}
__all__ = [k for k in globals().keys() if k not in _EXCLUDE and not k.startswith("_")]
