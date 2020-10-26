# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
from collections import OrderedDict

from detectron2.checkpoint import DetectionCheckpointer


def _rename_HRNet_weights(weights):
    # We detect and  rename HRNet weights for DensePose. 1956 and 1716 are values that are
    # common to all HRNet pretrained weights, and should be enough to accurately identify them
    if (
        len(weights["model"].keys()) == 1956
        and len([k for k in weights["model"].keys() if k.startswith("stage")]) == 1716
    ):
        hrnet_weights = OrderedDict()
        for k in weights["model"].keys():
            hrnet_weights["backbone.bottom_up." + str(k)] = weights["model"][k]
        return {"model": hrnet_weights}
    else:
        return weights


class DensePoseCheckpointer(DetectionCheckpointer):
    """
    Same as :class:`DetectionCheckpointer`, but is able to handle HRNet weights
    """

    def __init__(self, model, save_dir="", *, save_to_disk=None, **checkpointables):
        super().__init__(model, save_dir, save_to_disk=save_to_disk, **checkpointables)

    def _load_file(self, filename: str) -> object:
        """
        Adding hrnet support;
        Add AdelaiDet support, convert models, such as LPF backbone.
        """
        if filename.endswith(".pkl"):
            with PathManager.open(filename, "rb") as f:
                data = pickle.load(f, encoding="latin1")
            if "model" in data and "__author__" in data:
                # file is in Detectron2 model zoo format
                self.logger.info("Reading a file from '{}'".format(data["__author__"]))
                return data
            else:
                # assume file is from Caffe2 / Detectron1 model zoo
                if "blobs" in data:
                    # Detection models have "blobs", but ImageNet models don't
                    data = data["blobs"]
                data = {k: v for k, v in data.items() if not k.endswith("_momentum")}
                if "weight_order" in data:
                    del data["weight_order"]
                return {"model": data, "__author__": "Caffe2", "matching_heuristics": True}

        loaded = super()._load_file(filename)  # load native pth checkpoint
        if "model" not in loaded:
            loaded = {"model": loaded}

        basename = os.path.basename(filename).lower()
        if "lpf" in basename or "dla" in basename:
            loaded["matching_heuristics"] = True

        return _rename_HRNet_weights(loaded)


