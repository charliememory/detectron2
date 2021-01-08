# Copyright (c) Facebook, Inc. and its affiliates.

from dataclasses import dataclass
from typing import Union, Optional
import torch


@dataclass
class DensePoseChartPredictorOutput:
    """
    Predictor output that contains segmentation and inner coordinates predictions for predefined
    body parts:
     * coarse segmentation, a tensor of shape [N, K, Hout, Wout]
     * fine segmentation, a tensor of shape [N, C, Hout, Wout]
     * U coordinates, a tensor of shape [N, C, Hout, Wout]
     * V coordinates, a tensor of shape [N, C, Hout, Wout]
    where
     - N is the number of instances
     - K is the number of coarse segmentation channels (
         2 = foreground / background,
         15 = one of 14 body parts / background)
     - C is the number of fine segmentation channels (
         24 fine body parts / background)
     - Hout and Wout are height and width of predictions
    """
    # def __init__(self, 
    #              coarse_segm: torch.Tensor, 
    #              fine_segm: torch.Tensor,
    #              u: torch.Tensor,
    #              v: torch.Tensor,
    #              aux_supervision: Optional[torch.Tensor]=None):

    coarse_segm: torch.Tensor
    fine_segm: torch.Tensor
    u: torch.Tensor
    v: torch.Tensor
    aux_supervision: Optional[torch.Tensor] = None
    stride: Optional[int] = None

    def __len__(self):
        """
        Number of instances (N) in the output
        """
        return self.coarse_segm.size(0)

    def __getitem__(
        self, item: Union[int, slice, torch.BoolTensor]
    ) -> "DensePoseChartPredictorOutput":
        """
        Get outputs for the selected instance(s)

        Args:
            item (int or slice or tensor): selected items
        """
        if self.aux_supervision is None:
            if isinstance(item, int):
                return DensePoseChartPredictorOutput(
                    coarse_segm=self.coarse_segm[item].unsqueeze(0),
                    fine_segm=self.fine_segm[item].unsqueeze(0),
                    u=self.u[item].unsqueeze(0),
                    v=self.v[item].unsqueeze(0),
                )
            else:
                return DensePoseChartPredictorOutput(
                    coarse_segm=self.coarse_segm[item],
                    fine_segm=self.fine_segm[item],
                    u=self.u[item],
                    v=self.v[item],
                )
        else:
# <<<<<<< HEAD
            if isinstance(item, int):
                return DensePoseChartPredictorOutput(
                    coarse_segm=self.coarse_segm[item].unsqueeze(0),
                    fine_segm=self.fine_segm[item].unsqueeze(0),
                    u=self.u[item].unsqueeze(0),
                    v=self.v[item].unsqueeze(0),
                    aux_supervision=self.aux_supervision[item].unsqueeze(0),
                    stride=self.stride[item].unsqueeze(0),
                )
            else:
                return DensePoseChartPredictorOutput(
                    coarse_segm=self.coarse_segm[item],
                    fine_segm=self.fine_segm[item],
                    u=self.u[item],
                    v=self.v[item],
                    aux_supervision=self.aux_supervision[item],
                    stride=self.stride[item],
                )
# =======
            # return DensePoseChartPredictorOutput(
            #     coarse_segm=self.coarse_segm[item],
            #     fine_segm=self.fine_segm[item],
            #     u=self.u[item],
            #     v=self.v[item],
            # )

    def to(self, device: torch.device):
        """
        Transfers all tensors to the given device
        """
        coarse_segm = self.coarse_segm.to(device)
        fine_segm = self.fine_segm.to(device)
        u = self.u.to(device)
        v = self.v.to(device)
        return DensePoseChartPredictorOutput(coarse_segm=coarse_segm, fine_segm=fine_segm, u=u, v=v)
# >>>>>>> upstream/master
