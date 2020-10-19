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

# Just to register models from adet 
from .fcos import FCOS  
from .backbone import build_fcos_resnet_fpn_backbone
from .condinst import condinst

_EXCLUDE = {"torch", "ShapeSpec"}
__all__ = [k for k in globals().keys() if k not in _EXCLUDE and not k.startswith("_")]
