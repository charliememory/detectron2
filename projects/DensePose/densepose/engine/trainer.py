# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
import os
from collections import OrderedDict
from typing import List, Optional
import torch
from torch import nn

import torch
import time, pdb
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode
# from detectron2.engine import DefaultTrainer
# <<<<<<< HEAD
# from detectron2.engine import DefaultTrainer
from .defaults import DefaultTrainer
# from detectron2.evaluation import COCOEvaluator, DatasetEvaluators
# =======
from detectron2.evaluation import (
    COCOEvaluator,
    DatasetEvaluator,
    DatasetEvaluators,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.solver.build import get_default_optimizer_params, maybe_add_gradient_clipping
from detectron2.utils import comm
# >>>>>>> upstream/master
from detectron2.utils.events import EventWriter, get_event_storage

from densepose import (
    DensePoseCOCOEvaluator,
    DensePoseDatasetMapperTTA,
    DensePoseGeneralizedRCNNWithTTA,
    load_from_cfg,
)
from densepose.data import (
    DatasetMapper,
    build_combined_loader,
    build_detection_test_loader,
    build_detection_train_loader,
    build_inference_based_loaders,
    has_inference_based_loaders,
)
from densepose.modeling.cse import Embedder


class SampleCountingLoader:
    def __init__(self, loader):
        self.loader = loader

    def __iter__(self):
        it = iter(self.loader)
        storage = get_event_storage()
        while True:
            try:
                batch = next(it)
                num_inst_per_dataset = {}
                for data in batch:
                    dataset_name = data["dataset"]
                    if dataset_name not in num_inst_per_dataset:
                        num_inst_per_dataset[dataset_name] = 0
                    num_inst = len(data["instances"])
                    num_inst_per_dataset[dataset_name] += num_inst
                for dataset_name in num_inst_per_dataset:
                    storage.put_scalar(f"batch/{dataset_name}", num_inst_per_dataset[dataset_name])
                yield batch
            except StopIteration:
                break


class SampleCountMetricPrinter(EventWriter):
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def write(self):
        storage = get_event_storage()
        batch_stats_strs = []
        for key, buf in storage.histories().items():
            if key.startswith("batch/"):
                batch_stats_strs.append(f"{key} {buf.avg(20)}")
        self.logger.info(", ".join(batch_stats_strs))


class Trainer(DefaultTrainer):
    @classmethod
    def extract_embedder_from_model(cls, model: nn.Module) -> Optional[Embedder]:
        if isinstance(model, nn.parallel.DistributedDataParallel):
            model = model.module
        if hasattr(model, "roi_heads") and hasattr(model.roi_heads, "embedder"):
            return model.roi_heads.embedder
        return None

    # TODO: the only reason to copy the base class code here is to pass the embedder from
    # the model to the evaluator; that should be refactored to avoid unnecessary copy-pasting
    @classmethod
    def test(cls, cfg: CfgNode, model: nn.Module, evaluators: List[DatasetEvaluator] = None):
        """
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.

        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    embedder = cls.extract_embedder_from_model(model)
                    evaluator = cls.build_evaluator(cfg, dataset_name, embedder=embedder)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results

    @classmethod
    def build_evaluator(
        cls,
        cfg: CfgNode,
        dataset_name: str,
        output_folder: Optional[str] = None,
        embedder: Embedder = None,
    ) -> DatasetEvaluators:
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluators = [COCOEvaluator(dataset_name, output_dir=output_folder)]
        if cfg.MODEL.DENSEPOSE_ON:
# <<<<<<< HEAD
#             evaluators.append(DensePoseCOCOEvaluator(dataset_name, cfg, True, output_folder))
# =======
            evaluators.append(
                DensePoseCOCOEvaluator(dataset_name, cfg, True, output_folder, embedder=embedder)
            )
# >>>>>>> upstream/master
        return DatasetEvaluators(evaluators)

    @classmethod
    def build_optimizer(cls, cfg: CfgNode, model: nn.Module):
        params = get_default_optimizer_params(
            model,
            base_lr=cfg.SOLVER.BASE_LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
            bias_lr_factor=cfg.SOLVER.BIAS_LR_FACTOR,
            weight_decay_bias=cfg.SOLVER.WEIGHT_DECAY_BIAS,
            overrides={
                "features": {
                    "lr": cfg.SOLVER.BASE_LR * cfg.MODEL.ROI_DENSEPOSE_HEAD.CSE.FEATURES_LR_FACTOR,
                },
                "embeddings": {
                    "lr": cfg.SOLVER.BASE_LR * cfg.MODEL.ROI_DENSEPOSE_HEAD.CSE.EMBEDDING_LR_FACTOR,
                },
            },
        )
        optimizer = torch.optim.SGD(
            params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM, nesterov=cfg.SOLVER.NESTEROV
        )
        return maybe_add_gradient_clipping(cfg, optimizer)

    @classmethod
    def build_test_loader(cls, cfg: CfgNode, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=DatasetMapper(cfg, False))

    @classmethod
    def build_train_loader(cls, cfg: CfgNode):
        data_loader = build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, True))
        if not has_inference_based_loaders(cfg):
            return data_loader
        model = cls.build_model(cfg)
        model.to(cfg.BOOTSTRAP_MODEL.DEVICE)
        DetectionCheckpointer(model).resume_or_load(cfg.BOOTSTRAP_MODEL.WEIGHTS, resume=False)
        inference_based_loaders, ratios = build_inference_based_loaders(cfg, model)
        loaders = [data_loader] + inference_based_loaders
        ratios = [1.0] + ratios
        combined_data_loader = build_combined_loader(cfg, loaders, ratios)
        sample_counting_loader = SampleCountingLoader(combined_data_loader)
        return sample_counting_loader

    def build_writers(self):
        writers = super().build_writers()
        writers.append(SampleCountMetricPrinter())
        return writers

    @classmethod
    def test_with_TTA(cls, cfg: CfgNode, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        transform_data = load_from_cfg(cfg)
        model = DensePoseGeneralizedRCNNWithTTA(
            cfg, model, transform_data, DensePoseDatasetMapperTTA(cfg)
        )
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

    # ## Ref: https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/10
    # def plot_grad_flow(self, named_parameters):
    #     '''Plots the gradients flowing through different layers in the net during training.
    #     Can be used for checking for possible gradient vanishing / exploding problems.
        
    #     Usage: Plug this function in Trainer class after loss.backwards() as 
    #     "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    #     if not hasattr(self,"train_step"):
    #         self.train_step = 0
    #     if self.train_step%20!=0:
    #         self.train_step += 1
    #         return
    #     ave_grads = []
    #     max_grads= []
    #     layers = []
    #     for n, p in named_parameters:
    #         if(p.requires_grad) and (p.grad is not None) and ("bias" not in n):
    #             layers.append(n)
    #             ave_grads.append(p.grad.abs().mean())
    #             max_grads.append(p.grad.abs().max())

    #     # figure = plt.gcf() # get current figure
    #     figure = plt.figure()
    #     figure.set_size_inches(18, 18)

    #     plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    #     plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    #     plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    #     plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    #     plt.xlim(left=0, right=len(ave_grads))
    #     plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    #     plt.xlabel("Layers")
    #     plt.ylabel("average gradient")
    #     plt.title("Gradient flow")
    #     plt.grid(True)
    #     plt.legend([Line2D([0], [0], color="c", lw=4),
    #                 Line2D([0], [0], color="b", lw=4),
    #                 Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    #     plt.savefig("tmp/roi_plot_grad_flow_{}.pdf".format(self.train_step))
    #     self.train_step += 1
    #     plt.close()
    
    # def run_step(self):
    #     """
    #     Implement the standard training logic described above.
    #     """
    #     assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
    #     start = time.perf_counter()
    #     """
    #     If you want to do something with the data, you can wrap the dataloader.
    #     """
    #     data = next(self._data_loader_iter)
    #     data_time = time.perf_counter() - start

    #     """
    #     If you want to do something with the losses, you can wrap the model.
    #     """
    #     loss_dict = self.model(data)
    #     losses = sum(loss_dict.values())

    #     """
    #     If you need to accumulate gradients or do something similar, you can
    #     wrap the optimizer with your custom `zero_grad()` method.
    #     """
    #     self.optimizer.zero_grad()
    #     losses.backward()

    #     # self.plot_grad_flow(self.model.named_parameters())
    #     # pdb.set_trace()

    #     # use a new stream so the ops don't wait for DDP
    #     with torch.cuda.stream(
    #         torch.cuda.Stream()
    #     ) if losses.device.type == "cuda" else _nullcontext():
    #         metrics_dict = loss_dict
    #         metrics_dict["data_time"] = data_time
    #         self._write_metrics(metrics_dict)
    #         self._detect_anomaly(losses, loss_dict)

    #     """
    #     If you need gradient clipping/scaling or other processing, you can
    #     wrap the optimizer with your custom `step()` method. But it is
    #     suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
    #     """
    #     self.optimizer.step()

