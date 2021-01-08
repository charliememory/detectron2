# Copyright (c) Facebook, Inc. and its affiliates.

from .chart import DensePoseChartLoss
from .chart_with_confidences import DensePoseChartWithConfidenceLoss
# <<<<<<< HEAD
# from .chart_global_IUV_seperated_S import DensePoseChartGlobalIUVSeparatedSLoss
# from .chart_global_IUV_seperated_S_cropResize import DensePoseChartGlobalIUVSeparatedSCropResizeLoss
# from .chart_global_IUV_seperated_S_multiscale_cropResize import DensePoseChartGlobalIUVSeparatedSMultiscaleCropResizeLoss
from .chart_global_IUV_seperated_S_pooler import DensePoseChartGlobalIUVSeparatedSPoolerLoss
# =======
from .cse import DensePoseCseLoss
from .registry import DENSEPOSE_LOSS_REGISTRY


__all__ = [
    "DensePoseChartLoss",
    "DensePoseChartWithConfidenceLoss",
    "DensePoseCseLoss",
    "DENSEPOSE_LOSS_REGISTRY",
    "DensePoseChartGlobalIUVSeparatedSPoolerLoss"
]
# >>>>>>> upstream/master
