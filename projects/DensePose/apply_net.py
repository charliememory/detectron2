#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import glob
import logging
import os, pdb, tqdm, copy
import pickle
import sys
from typing import Any, ClassVar, Dict, List
import torch
import torch.nn.functional as F

from detectron2.config import CfgNode
from detectron2.data.detection_utils import read_image
from detectron2.structures.boxes import BoxMode
from detectron2.structures.instances import Instances
from detectron2.utils.logger import setup_logger
# from detectron2.engine.defaults import DefaultPredictor
from densepose.engine.defaults import DefaultPredictor

# from densepose import add_densepose_config, add_hrnet_config
from densepose.config import get_cfg, add_densepose_config, add_hrnet_config
from densepose.utils.logger import verbosity_to_level
from densepose.vis.base import CompoundVisualizer
from densepose.vis.bounding_box import ScoredBoundingBoxVisualizer
from densepose.vis.densepose_outputs_vertex import (
    DensePoseOutputsTextureVisualizer,
    DensePoseOutputsVertexVisualizer,
    get_texture_atlases,
)
from densepose.vis.densepose_results import (
    DensePoseResultsContourVisualizer,
    DensePoseResultsFineSegmentationVisualizer,
    DensePoseResultsUVisualizer,
    DensePoseResultsVVisualizer,
)
from densepose.vis.densepose_results_textures import (
    DensePoseResultsVisualizerWithTexture,
    get_texture_atlas,
)
# from densepose.vis.extractor import CompoundExtractor, create_extractor
from densepose.vis.extractor import CompoundExtractor, DensePoseResultExtractor, create_extractor

DOC = """Apply Net - a tool to print / visualize DensePose results
"""

LOGGER_NAME = "apply_net"
logger = logging.getLogger(LOGGER_NAME)

_ACTION_REGISTRY: Dict[str, "Action"] = {}

## MLQ added

class Action(object):
    @classmethod
    def add_arguments(cls: type, parser: argparse.ArgumentParser):
        parser.add_argument(
            "-v",
            "--verbosity",
            action="count",
            help="Verbose mode. Multiple -v options increase the verbosity.",
        )


def register_action(cls: type):
    """
    Decorator for action classes to automate action registration
    """
    global _ACTION_REGISTRY
    _ACTION_REGISTRY[cls.COMMAND] = cls
    return cls


class InferenceAction(Action):
    @classmethod
    def add_arguments(cls: type, parser: argparse.ArgumentParser):
        super(InferenceAction, cls).add_arguments(parser)
        parser.add_argument("cfg", metavar="<config>", help="Config file")
        parser.add_argument("model", metavar="<model>", help="Model file")
        parser.add_argument("input", metavar="<input>", help="Input data")
        # parser.add_argument("--smooth_k", type=int, default=0, metavar="<smooth_k>", help="smooth_k")
        parser.add_argument(
            "--opts",
            help="Modify config options using the command-line 'KEY VALUE' pairs",
            default=[],
            nargs=argparse.REMAINDER,
        )

    @classmethod
    def execute(cls: type, args: argparse.Namespace):
        logger.info(f"Loading config from {args.cfg}")
        opts = []
        cfg = cls.setup_config(args.cfg, args.model, args, opts)
        logger.info(f"Loading model from {args.model}")
        predictor = DefaultPredictor(cfg)
        logger.info(f"Loading data from {args.input}")
        file_list = cls._get_input_file_list(args.input)
        if len(file_list) == 0:
            logger.warning(f"No input images for {args.input}")
            return
# <<<<<<< HEAD
        # context = cls.create_context(args)
        context = cls.create_context(args, cfg)
        smooth_radius = cfg.MODEL.INFERENCE_SMOOTH_FRAME_NUM

        # # pdb.set_trace()
        # if smooth_radius>0:
        #     # weight_dic = {-2:0.2, -1:0.2, 0: 0.2, 1:0.2, 2:0.2}
        #     # weight_dic = [0.1,0.2,0.4,0.2,0.1]
        #     flow_model = flow_model_wrapper(predictor.model.device)
        #     # self.flow_model.eval()

        #     img_queue = []
        #     dp_output_queue = []

        #     # cnt = 0
        #     for fid in tqdm.tqdm(range(len(file_list))):
        #         file_name = file_list[fid]
        #         if img_queue==[]:
        #             for i in range(-smooth_radius, smooth_radius+1):
        #                 ii = min(len(file_list)-1, max(0,fid+i))
        #                 img = read_image(file_list[ii], format="BGR")
        #                 img_queue.append(img.copy())
        #                 with torch.no_grad():
        #                     outputs = predictor(img)["instances"]
        #                 # if cfg.MODEL.CONDINST.INFERENCE_GLOBAL_SIUV:
        #                 #     outputs = cls.seperate_global_siuv(outputs)
        #                 dp_output_queue.append(outputs)
        #         else:
        #             img_queue.pop(0)
        #             dp_output_queue.pop(0)
        #             ii = min(len(file_list)-1, fid+smooth_radius)
        #             img = read_image(file_list[ii], format="BGR")
        #             img_queue.append(img.copy())
        #             with torch.no_grad():
        #                 outputs = predictor(img)["instances"]
        #             dp_output_queue.append(outputs)

        #         center_idx = len(img_queue)//2
        #         dp_output_warp = copy.copy(dp_output_queue[center_idx])
        #         for i in range(len(img_queue)):
        #             if i == center_idx:
        #                 continue
        #             else:
        #                 pdb.set_trace()
        #                 img_tgt = torch.tensor(img_queue[center_idx]).permute([2,0,1])[None,...].float()
        #                 img_ref = torch.tensor(img_queue[i]).permute([2,0,1])[None,...].float()
        #                 flow_fw = flow_model.pred_flow(img_tgt, img_ref)
        #                 dp = copy.copy(dp_output_queue[i])
        #                 "TODO, warp siuv"
        #                 dp_output_warp.pred_densepose.fine_segm += flow_model.tensor_warp_via_flow(dp.pred_densepose.fine_segm, flow_fw)
        #                 dp_output_warp.pred_densepose.u += flow_model.tensor_warp_via_flow(dp.pred_densepose.u, flow_fw)
        #                 dp_output_warp.pred_densepose.v += flow_model.tensor_warp_via_flow(dp.pred_densepose.v, flow_fw)
        #         dp_output_warp.pred_densepose.fine_segm /= len(img_queue)
        #         dp_output_warp.pred_densepose.u /= len(img_queue)
        #         dp_output_warp.pred_densepose.v /= len(img_queue)
        #         pdb.set_trace()
        #         if cfg.MODEL.CONDINST.INFERENCE_GLOBAL_SIUV:
        #             dp_output_warp = cls.seperate_global_siuv(dp_output_warp)
        #         # # cnt += 1
        #         # # if cnt<50:
        #         # #     continue
        #         # # pdb.set_trace()
        #         # # if smooth_radius>0:
        #         # img = read_image(file_name, format="BGR")  # predictor expects BGR image.
        #         # image_adj_list = []
        #         # for i in range(-smooth_radius, smooth_radius+1):
        #         #     if i==0:
        #         #         continue
        #         #     ii = min(len(file_list)-1, max(0,fid+i))
        #         #     image_adj_list.append(read_image(file_list[ii], format="BGR"))
        #         # with torch.no_grad():
        #         #     outputs = predictor(img, image_adj_list)["instances"]

        #         # if cfg.MODEL.CONDINST.INFERENCE_GLOBAL_SIUV:
        #         #     outputs = cls.seperate_global_siuv(outputs)

        #         cls.execute_on_outputs(context, {"file_name": file_name, "image": img}, dp_output_warp)
                    
        #         torch.cuda.empty_cache()
        # else:
        for fid in tqdm.tqdm(range(len(file_list))):
            file_name = file_list[fid]

            if smooth_radius>0:
                img = read_image(file_name, format="BGR")  # predictor expects BGR image.
                image_adj_list = []
                for i in range(-smooth_radius, smooth_radius+1):
                    if i==0:
                        continue
                    ii = min(len(file_list)-1, max(0,fid+i))
                    image_adj_list.append(read_image(file_list[ii], format="BGR"))
                with torch.no_grad():
                    outputs = predictor(img, image_adj_list)["instances"]
            else:
                img = read_image(file_name, format="BGR")  # predictor expects BGR image.
                with torch.no_grad():
                    outputs = predictor(img)["instances"]

            if cfg.MODEL.CONDINST.INFERENCE_GLOBAL_SIUV:
                outputs = cls.seperate_global_siuv(outputs)

            cls.execute_on_outputs(context, {"file_name": file_name, "image": img}, outputs)
                
            torch.cuda.empty_cache()
# =======
#         context = cls.create_context(args, cfg)
#         for file_name in file_list:
#             img = read_image(file_name, format="BGR")  # predictor expects BGR image.
#             with torch.no_grad():
#                 outputs = predictor(img)["instances"]
#                 cls.execute_on_outputs(context, {"file_name": file_name, "image": img}, outputs)
# >>>>>>> upstream/master
        cls.postexecute(context)

    ## MLQ added
    @classmethod
    def seperate_global_siuv(cls: type, outputs: Instances):
        "TODO"
        N_ins = outputs.pred_boxes.tensor.shape[0]
        if N_ins==0:
            return outputs
        # pdb.set_trace()
        assert N_ins==outputs.pred_densepose.coarse_segm.shape[0]
        device = outputs.pred_boxes.tensor.device
        H, W = outputs.image_size
        # print(N_ins)
        outputs.pred_boxes.tensor = torch.tensor([[0,0,W,H]], device=device).repeat(N_ins, 1)
        outputs.pred_densepose.fine_segm = outputs.pred_densepose.fine_segm.repeat(N_ins, 1, 1, 1)
        outputs.pred_densepose.u = outputs.pred_densepose.u.repeat(N_ins, 1, 1, 1)
        outputs.pred_densepose.v = outputs.pred_densepose.v.repeat(N_ins, 1, 1, 1)
        return outputs

    # ## MLQ added
    # def create_and_load_netFlow(self, device):
    #     # parser.add_argument('--model', help="restore checkpoint")
    #     # parser.add_argument('--seq_img_dir', help="sequence images for evaluation")
    #     # parser.add_argument('--backward_flow', action='store_true', help='calculate flow from i+1 to i')
    #     # parser.add_argument('--small', action='store_true', help='use small model')
    #     # parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    #     # parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    #     # args = parser.parse_args()
    #     class Args:
    #         def __init__(self):
    #             # self.model = "./RAFT/models/raft-sintel.pth"
    #             # self.small = False
    #             # self.mixed_precision = False
    #             # self.dropout = 0
    #             # self.alternate_corr = False
    #             self.model = "./RAFT/models/raft-small.pth"
    #             self.small = True
    #             self.mixed_precision = False
    #             self.dropout = 0
    #             self.alternate_corr = False
    #     args = Args()

    #     # netFlow = RAFT(args)
    #     netFlow = torch.nn.DataParallel(RAFT(args))
    #     netFlow.load_state_dict(torch.load(args.model, map_location=device))
    #     print("Create and load netFlow successfully")
    #     return netFlow

    # def pred_flow(self, model, img0, img1, iters=20):
    #     try:
    #         assert img0.min()>=0 and img0.max()>=10 and img0.max()<=255, "input image range should be [0,255], but got [{},{}]".format(img0.min(),img0.max())
    #     except:
    #         print("input image range should be [0,255], but got [{},{}]".format(img0.min(),img0.max()))
    #         raise ValueError

    #     padder = InputPadder(img0.shape, mode='sintel')
    #     img0, img1 = padder.pad(img0, img1)

    #     flow_low, flow_pr = model(img0, img1, iters, test_mode=True)
    #     flow = padder.unpad(flow_pr)
    #     return flow

    # def tensor_warp_via_flow(self, tensor, flow):
    #     b, _, h, w = tensor.shape
    #     coords = self.flow2coord(flow).permute([0,2,3,1]) # [0,h-1], [0,w-1]
    #     tensor = F.grid_sample(tensor, coords) #, mode='bilinear', align_corners=True)
    #     return tensor

    # def flow2coord(self, flow):
    #     def meshgrid(height, width):
    #         x_t = torch.matmul(
    #             torch.ones(height, 1), torch.linspace(-1.0, 1.0, width).view(1, width))
    #         y_t = torch.matmul(
    #             torch.linspace(-1.0, 1.0, height).view(height, 1), torch.ones(1, width))

    #         grid_x = x_t.view(1, 1, height, width)
    #         grid_y = y_t.view(1, 1, height, width)
    #         return grid_x, grid_y
    #         # return torch.cat([grid_x,grid_y], dim=-1)

    #     b, _, h, w = flow.shape
    #     grid_x, grid_y = meshgrid(h, w)
    #     coord_x = flow[:,0:1]/w + grid_x.to(flow.device)
    #     coord_y = flow[:,1:2]/h + grid_y.to(flow.device)
    #     return torch.cat([coord_x,coord_y], dim=1)

    # @classmethod
    # def execute(cls: type, args: argparse.Namespace, multi_frames=True):
    #     logger.info(f"Loading config from {args.cfg}")
    #     opts = []
    #     cfg = cls.setup_config(args.cfg, args.model, args, opts)
    #     logger.info(f"Loading model from {args.model}")
    #     predictor = DefaultPredictor(cfg) if not multi_frames else DefaultPredictorMulti(cfg)
    #     # pdb.set_trace()
    #     logger.info(f"Loading data from {args.input}")
    #     file_list = cls._get_input_file_list(args.input)
    #     if len(file_list) == 0:
    #         logger.warning(f"No input images for {args.input}")
    #         return
    #     context = cls.create_context(args)
    #     # for file_name in tqdm.tqdm(file_list):
    #     for idx in tqdm.tqdm(range(len(file_list))):
    #         if multi_frames:
    #             time_scope = 1
    #             img = []
    #             for i in range(-time_scope,time_scope+1,1):
    #                 idx_valid = min(max(idx+i,0),len(file_list))
    #                 file_name = file_list[idx_valid]
    #                 img.append(file_name)
    #         else:
    #             img = read_image(file_name, format="BGR")  # predictor expects BGR image.
    #         with torch.no_grad():
    #             outputs = predictor(img)["instances"]
    #             cls.execute_on_outputs(context, {"file_name": file_name, "image": img}, outputs)
    #     cls.postexecute(context)

    @classmethod
    def setup_config(
        cls: type, config_fpath: str, model_fpath: str, args: argparse.Namespace, opts: List[str]
    ):
        cfg = get_cfg()
        add_densepose_config(cfg)
        add_hrnet_config(cfg)
        cfg.merge_from_file(config_fpath)
        cfg.merge_from_list(args.opts)
        if opts:
            cfg.merge_from_list(opts)
        cfg.MODEL.WEIGHTS = model_fpath
        ## MLQ added
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        cfg.DATA_DIR = os.path.dirname(args.input)
        # cfg.SMOOTH_K = args.smooth_k
        cfg.freeze()
        return cfg

    @classmethod
    def _get_input_file_list(cls: type, input_spec: str):
        if os.path.isdir(input_spec):
            subfile = os.path.join(input_spec, os.listdir(input_spec)[0])
            if os.path.isdir(subfile):
                subsubfile = os.path.join(subfile, os.listdir(subfile)[0])
                if os.path.isdir(subsubfile):
                    file_list = glob.glob(os.path.join(input_spec, '*/*/*.jpg'))
                # pdb.set_trace()
            else:
                file_list = [
                    os.path.join(input_spec, fname)
                    for fname in os.listdir(input_spec)
                    if os.path.isfile(os.path.join(input_spec, fname))
                ]
        elif os.path.isfile(input_spec):
            file_list = [input_spec]
        else:
            file_list = glob.glob(input_spec)

        file_list = sorted(file_list)
        return file_list


@register_action
class DumpAction(InferenceAction):
    """
    Dump action that outputs results to a pickle file
    """

    COMMAND: ClassVar[str] = "dump"

    @classmethod
    def add_parser(cls: type, subparsers: argparse._SubParsersAction):
        parser = subparsers.add_parser(cls.COMMAND, help="Dump model outputs to a file.")
        cls.add_arguments(parser)
        parser.set_defaults(func=cls.execute)

    @classmethod
    def add_arguments(cls: type, parser: argparse.ArgumentParser):
        super(DumpAction, cls).add_arguments(parser)
        parser.add_argument(
            "--output",
            metavar="<dump_file>",
            default="results.pkl",
            help="File name to save dump to",
        )

    # @classmethod
    # def execute_on_outputs(
    #     cls: type, context: Dict[str, Any], entry: Dict[str, Any], outputs: Instances
    # ):
    #     image_fpath = entry["file_name"]
    #     logger.info(f"Processing {image_fpath}")
    #     result = {"file_name": image_fpath}
    #     if outputs.has("scores"):
    #         result["scores"] = outputs.get("scores").cpu()
    #     if outputs.has("pred_boxes"):
    #         result["pred_boxes_XYXY"] = outputs.get("pred_boxes").tensor.cpu()
    #         if outputs.has("pred_densepose"):
    #             # pdb.set_trace()
    #             boxes_XYWH = BoxMode.convert(result["pred_boxes_XYXY"], BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    #             result["pred_densepose"] = outputs.get("pred_densepose").to_result(boxes_XYWH)
    #     context["results"].append(result)

    @classmethod
    def execute_on_outputs(
        cls: type, context: Dict[str, Any], entry: Dict[str, Any], outputs: Instances
    ):
        image_fpath = entry["file_name"]
        logger.info(f"Processing {image_fpath}")
        result = {"file_name": image_fpath}
        if outputs.has("scores"):
            result["scores"] = outputs.get("scores").cpu()
        if outputs.has("pred_boxes"):
            result["pred_boxes_XYXY"] = outputs.get("pred_boxes").tensor.cpu()
            if outputs.has("pred_densepose"):
                result["pred_densepose"], _ = DensePoseResultExtractor()(outputs)
                for i in range(len(result["pred_densepose"])):
                    result["pred_densepose"][i].labels = result["pred_densepose"][i].labels.cpu()
                    result["pred_densepose"][i].uv = result["pred_densepose"][i].uv.cpu()
        context["results"].append(result)

    @classmethod
    def create_context(cls: type, args: argparse.Namespace):
        context = {"results": [], "out_fname": args.output}
        return context

    @classmethod
    def postexecute(cls: type, context: Dict[str, Any]):
        out_fname = context["out_fname"]
        out_dir = os.path.dirname(out_fname)
        if len(out_dir) > 0 and not os.path.exists(out_dir):
            os.makedirs(out_dir)
        with open(out_fname, "wb") as hFile:
            pickle.dump(context["results"], hFile)
            logger.info(f"Output saved to {out_fname}")


@register_action
class ShowAction(InferenceAction):
    """
    Show action that visualizes selected entries on an image
    """

    COMMAND: ClassVar[str] = "show"
    VISUALIZERS: ClassVar[Dict[str, object]] = {
        "dp_contour": DensePoseResultsContourVisualizer,
        "dp_segm": DensePoseResultsFineSegmentationVisualizer,
        "dp_u": DensePoseResultsUVisualizer,
        "dp_v": DensePoseResultsVVisualizer,
        "dp_iuv_texture": DensePoseResultsVisualizerWithTexture,
        "dp_cse_texture": DensePoseOutputsTextureVisualizer,
        "dp_vertex": DensePoseOutputsVertexVisualizer,
        "bbox": ScoredBoundingBoxVisualizer,
    }

    @classmethod
    def add_parser(cls: type, subparsers: argparse._SubParsersAction):
        parser = subparsers.add_parser(cls.COMMAND, help="Visualize selected entries")
        cls.add_arguments(parser)
        parser.set_defaults(func=cls.execute)

    @classmethod
    def add_arguments(cls: type, parser: argparse.ArgumentParser):
        super(ShowAction, cls).add_arguments(parser)
        parser.add_argument(
            "visualizations",
            metavar="<visualizations>",
            help="Comma separated list of visualizations, possible values: "
            "[{}]".format(",".join(sorted(cls.VISUALIZERS.keys()))),
        )
        parser.add_argument(
            "--min_score",
            metavar="<score>",
            default=0.8,
            type=float,
            help="Minimum detection score to visualize",
        )
        parser.add_argument(
            "--nms_thresh", metavar="<threshold>", default=None, type=float, help="NMS threshold"
        )
        parser.add_argument(
            "--texture_atlas",
            metavar="<texture_atlas>",
            default=None,
            help="Texture atlas file (for IUV texture transfer)",
        )
        parser.add_argument(
            "--texture_atlases_map",
            metavar="<texture_atlases_map>",
            default=None,
            help="JSON string of a dict containing texture atlas files for each mesh",
        )
        parser.add_argument(
            "--output",
            metavar="<image_file>",
            default="outputres.png",
            help="File name to save output to",
        )
        parser.add_argument(
            "--vis_rgb_img",
            action='store_true',
            help="Wheather visualize rgb image instead of gray (default)",
        )
        parser.add_argument(
            "--vis_black_img",
            action='store_true',
            help="Wheather visualize a black image instead of input image",
        )

    @classmethod
    def setup_config(
        cls: type, config_fpath: str, model_fpath: str, args: argparse.Namespace, opts: List[str]
    ):
        opts.append("MODEL.ROI_HEADS.SCORE_THRESH_TEST")
        opts.append(str(args.min_score))
        if args.nms_thresh is not None:
            opts.append("MODEL.ROI_HEADS.NMS_THRESH_TEST")
            opts.append(str(args.nms_thresh))
        cfg = super(ShowAction, cls).setup_config(config_fpath, model_fpath, args, opts)
        return cfg

    @classmethod
    def execute_on_outputs(
        cls: type, context: Dict[str, Any], entry: Dict[str, Any], outputs: Instances
    ):
        import cv2, pdb
        import numpy as np

        visualizer = context["visualizer"]
        extractor = context["extractor"]
        image_fpath = entry["file_name"]
        logger.info(f"Processing {image_fpath}")
        if context["vis_rgb_img"]:
            image = entry["image"][:]
            # pdb.set_trace()
        else:
            image = cv2.cvtColor(entry["image"], cv2.COLOR_BGR2GRAY)
            image = np.tile(image[:, :, np.newaxis], [1, 1, 3])
        if context["vis_black_img"]:
            image = np.zeros_like(image) 
        # pdb.set_trace()
        data = extractor(outputs)
        # pdb.set_trace()
        image_vis = visualizer.visualize(image, data)
        entry_idx = context["entry_idx"] + 1
        # out_fname = cls._get_out_fname(entry_idx, context["out_fname"])
        out_fname = os.path.join(context["out_fname"], image_fpath.split('/')[-1])
        out_dir = os.path.dirname(out_fname)
        if len(out_dir) > 0 and not os.path.exists(out_dir):
            os.makedirs(out_dir)
        cv2.imwrite(out_fname, image_vis)
        logger.info(f"Output saved to {out_fname}")
        context["entry_idx"] += 1

    @classmethod
    def postexecute(cls: type, context: Dict[str, Any]):
        pass

    @classmethod
    def _get_out_fname(cls: type, entry_idx: int, fname_base: str):
        base, ext = os.path.splitext(fname_base)
        # return base + ".{0:04d}".format(entry_idx) + ext
        return base + "{0:06d}".format(entry_idx) + ext

    @classmethod
    def create_context(cls: type, args: argparse.Namespace, cfg: CfgNode) -> Dict[str, Any]:
        vis_specs = args.visualizations.split(",")
        visualizers = []
        extractors = []
        for vis_spec in vis_specs:
            texture_atlas = get_texture_atlas(args.texture_atlas)
            texture_atlases_dict = get_texture_atlases(args.texture_atlases_map)
            vis = cls.VISUALIZERS[vis_spec](
                cfg=cfg,
                texture_atlas=texture_atlas,
                texture_atlases_dict=texture_atlases_dict,
            )
            visualizers.append(vis)
            extractor = create_extractor(vis)
            extractors.append(extractor)
        visualizer = CompoundVisualizer(visualizers)
        extractor = CompoundExtractor(extractors)
        context = {
            "extractor": extractor,
            "visualizer": visualizer,
            "out_fname": args.output,
            "vis_rgb_img": args.vis_rgb_img,
            "vis_black_img": args.vis_black_img,
            "entry_idx": 0,
        }
        return context


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=DOC,
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=120),
    )
    parser.set_defaults(func=lambda _: parser.print_help(sys.stdout))
    subparsers = parser.add_subparsers(title="Actions")
    for _, action in _ACTION_REGISTRY.items():
        action.add_parser(subparsers)
    return parser


def main():
    parser = create_argument_parser()
    args = parser.parse_args()
    verbosity = args.verbosity if hasattr(args, "verbosity") else None
    global logger
    logger = setup_logger(name=LOGGER_NAME)
    logger.setLevel(verbosity_to_level(verbosity))
    args.func(args)


if __name__ == "__main__":
    main()
