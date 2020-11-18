# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from .v1convx import DensePoseV1ConvXHead
from .v1convxgn import DensePoseV1ConvXGNHead
from .v1convxgn_sparse import DensePoseV1ConvXGNSparseHead
from .v1convxgn_sparsegn import DensePoseV1ConvXGNSparseGNHead
from .v1convxgn_sparsegn_esp import DensePoseV1ConvXGNSparseGNESPHead
from .deeplab import DensePoseDeepLabHead
from .deeplab_sparse import DensePoseDeepLabSparseHead
from .deeplab2_sparse import DensePoseDeepLab2SparseHead
from .registry import ROI_DENSEPOSE_HEAD_REGISTRY
from .roi_head import Decoder, DensePoseROIHeads
from .roi_global_head import DecoderGlobal, DensePoseROIGlobalHeads
