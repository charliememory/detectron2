source ~/.bashrc_liqianma

# cfg_name='densepose_CondInst_R_50_s1x'
# CUDA_LAUNCH_BLOCKING=1 python train_net.py --config-file configs/${cfg_name}.yaml \
#   --resume \
#     SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 \
#     OUTPUT_DIR ./output/${cfg_name}_1chSeg_cropResizeNew_deepIUVHead_lambda2 \
#     MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS 1 \
#     MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS True \
#     SOLVER.CHECKPOINT_PERIOD 5000 \
#     DATALOADER.NUM_WORKERS 2 \
#   MODEL.ROI_DENSEPOSE_HEAD.LOSS_NAME "DensePoseChartGlobalIUVSeparatedSCropResizeLoss" \
#   MODEL.CONDINST.IUVHead.NUM_CONVS 7 \
#     MODEL.CONDINST.IUVHead.NUM_LAMBDA_LAYER 2 \
#     # SOLVER.CLIP_GRADIENTS.ENABLED True \
#   # MODEL.ROI_DENSEPOSE_HEAD.NAME "DensePoseDeepLabHead" \
#   # MODEL.ROI_DENSEPOSE_HEAD.POINT_REGRESSION_WEIGHTS 1.0 \
#   # MODEL.CONDINST.IUVHead.REL_COORDS True \
#   # SOLVER.WARMUP_ITERS 2000
#     # SOLVER.CLIP_GRADIENTS.ENABLED True SOLVER.CLIP_GRADIENTS.CLIP_TYPE norm SOLVER.CLIP_GRADIENTS.CLIP_VALUE 100.0\


# cfg_name='densepose_CondInst_R_50_BiFPN_s1x'
# CUDA_LAUNCH_BLOCKING=1 python train_net.py --config-file configs/${cfg_name}.yaml \
#     --resume \
#     SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 \
#     OUTPUT_DIR ./output/${cfg_name}_1chSeg_cropResizeNew_maskSoftOutBg_deepIUVHead_lambda1_AbsRelCoordPE10 \
#     MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS 1 \
#     MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS True \
#     SOLVER.CHECKPOINT_PERIOD 5000 \
#     DATALOADER.NUM_WORKERS 2 \
#     MODEL.ROI_DENSEPOSE_HEAD.LOSS_NAME "DensePoseChartGlobalIUVSeparatedSCropResizeLoss" \
#     MODEL.CONDINST.IUVHead.NUM_CONVS 7 \
#     MODEL.CONDINST.IUVHead.ABS_COORDS True \
#     MODEL.CONDINST.IUVHead.REL_COORDS True \
#     MODEL.CONDINST.IUVHead.POSE_EMBEDDING_NUM_FREQS 10 \
#     MODEL.CONDINST.IUVHead.NUM_LAMBDA_LAYER 1 \
#     MODEL.CONDINST.IUVHead.MASK_OUT_BG_FEATURES "soft" \


# cfg_name='densepose_CondInst_R_101_s1x'
# cfg_name='densepose_CondInst_R_50_BiFPN_s1x'
# CUDA_LAUNCH_BLOCKING=1 python train_net.py --config-file configs/${cfg_name}.yaml \
#     --resume \
#     SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 \
#     OUTPUT_DIR ./output/${cfg_name}_1chSeg_cropResizeNew_deepWide2IUVHead_lambda51_AbsRelCoordPE10 \
#     MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS 1 \
#     MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS True \
#     SOLVER.CHECKPOINT_PERIOD 5000 \
#     DATALOADER.NUM_WORKERS 2 \
#     MODEL.ROI_DENSEPOSE_HEAD.LOSS_NAME "DensePoseChartGlobalIUVSeparatedSCropResizeLoss" \
#     MODEL.CONDINST.IUVHead.NUM_CONVS 7 \
#     MODEL.CONDINST.MASK_BRANCH.AGG_CHANNELS 256\
#     MODEL.CONDINST.IUVHead.CHANNELS 256 \
#     MODEL.CONDINST.IUVHead.ABS_COORDS True \
#     MODEL.CONDINST.IUVHead.REL_COORDS True \
#     MODEL.CONDINST.IUVHead.POSE_EMBEDDING_NUM_FREQS 10 \
#     MODEL.CONDINST.IUVHead.NUM_LAMBDA_LAYER 1 \
#     MODEL.CONDINST.IUVHead.LAMBDA_LAYER_R 51 \
#     # MODEL.ROI_DENSEPOSE_HEAD.NAME "DensePoseMultilayermaskHead" \

# cfg_name='densepose_CondInst_R_50_BiFPN_s1x'
# CUDA_LAUNCH_BLOCKING=1 python train_net.py --config-file configs/${cfg_name}.yaml \
#     --resume \
#     SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 \
#     OUTPUT_DIR ./output/${cfg_name}_1chSeg_cropResizeNew_deepWide2IUVHead_lambda1_AbsRelCoordPE10_GTins \
#     MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS 1 \
#     MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS True \
#     SOLVER.CHECKPOINT_PERIOD 5000 \
#     DATALOADER.NUM_WORKERS 2 \
#     MODEL.ROI_DENSEPOSE_HEAD.LOSS_NAME "DensePoseChartGlobalIUVSeparatedSCropResizeLoss" \
#     MODEL.CONDINST.IUVHead.NUM_CONVS 7 \
#     MODEL.CONDINST.MASK_BRANCH.AGG_CHANNELS 256\
#     MODEL.CONDINST.IUVHead.CHANNELS 256 \
#     MODEL.CONDINST.IUVHead.ABS_COORDS True \
#     MODEL.CONDINST.IUVHead.REL_COORDS True \
#     MODEL.CONDINST.IUVHead.GT_INSTANCES True \
#     MODEL.CONDINST.IUVHead.POSE_EMBEDDING_NUM_FREQS 10 \
#     MODEL.CONDINST.IUVHead.NUM_LAMBDA_LAYER 1 \
#     # MODEL.CONDINST.IUVHead.DOWN_UP_SAMPLING True \
#     # MODEL.CONDINST.IUVHead.LAMBDA_LAYER_R 51 \
#     # MODEL.ROI_DENSEPOSE_HEAD.NAME "DensePoseMultilayermaskHead" \


if [ -d "/usr/local/cuda-10.2/bin" ] 
then
    echo "/usr/local/cuda-10.2/bin exists." 
else
    echo "/usr/local/cuda-10.2/bin does not exists. Use cuda-11.1"
    export CUDA_HOME=/usr/local/cuda-11.1
    export CUDNN_HOME=/esat/dragon/liqianma/workspace/cudnn-11.1-linux-x64-v8.0.4.30
    export PATH=$CUDA_HOME/bin:$PATH 
    # for torch
    export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME
    export CUDA_BIN_PATH=$CUDA_HOME
    # libs for deep learning framework
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH"
    # for CUDA & atlas
    export CUDNN_INCLUDE="$CUDNN_HOME/include"
    export CUDNN_INCLUDE_DIR="$CUDNN_HOME/include"
    export INCLUDE_DIR="$CUDA_HOME/include:$CUDNN_HOME/include:$INCLUDE_DIR"
fi
cd ~/workspace/Gitlab/spconv/
rm -rf build
python setup.py bdist_wheel
cd ./dist
pip uninstall spconv -y
pip install spconv-1.2.1-cp38-cp38-linux_x86_64.whl
cd ~/workspace/Gitlab/detectron2/projects/DensePose
 

cfg_name='densepose_CondInst_R_50_s1x'
CUDA_LAUNCH_BLOCKING=1 python train_net.py --config-file configs/${cfg_name}.yaml \
    --resume --num-gpus 1 \
    SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 SOLVER.ACCUMULATE_GRAD_ITER 1 \
    OUTPUT_DIR ./output/${cfg_name}_1chSeg_IUVSparsePooler2Head_AggFea_V1ConvXGNSparseInsINLowMemNoOverlapTrueResInput_resIUVOnly_addGTSkeletonFea_GTinsDilated3_amp  \
    MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS 1 \
    MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS True \
    SOLVER.CHECKPOINT_PERIOD 5000 \
    DATALOADER.NUM_WORKERS 4 \
    MODEL.CONDINST.IUVHead.NAME "IUVSparsePooler2Head" \
    MODEL.ROI_DENSEPOSE_HEAD.LOSS_NAME "DensePoseChartGlobalIUVSeparatedSPoolerLoss" \
    MODEL.ROI_DENSEPOSE_HEAD.NAME "DensePoseV1ConvXGNSparseGNHead" \
    MODEL.ROI_DENSEPOSE_HEAD.CONV_HEAD_DIM 256 \
    MODEL.CONDINST.MASK_BRANCH.AGG_CHANNELS 256 \
    MODEL.CONDINST.IUVHead.MASK_OUT_BG_FEATURES "hard" \
    MODEL.CONDINST.IUVHead.DILATE_FGMASK_KENERAL_SIZE 3 \
    MODEL.ROI_DENSEPOSE_HEAD.DEEPLAB.NORM "InsIN" \
    MODEL.CONDINST.IUVHead.USE_AGG_FEATURES True \
    MODEL.CONDINST.IUVHead.GT_INSTANCES True \
    MODEL.CONDINST.IUVHead.INSTANCE_AWARE_GN True \
    MODEL.CONDINST.IUVHead.REMOVE_MASK_OVERLAP True \
    MODEL.CONDINST.v2 True \
    SOLVER.AMP.ENABLED True \
    MODEL.CONDINST.IUVHead.RESIDUAL_INPUT True \
    MODEL.KEYPOINT_ON True\
    MODEL.CONDINST.IUVHead.SKELETON_FEATURES True \
    MODEL.CONDINST.IUVHead.GT_SKELETON True \
    # MODEL.CONDINST.MASK_BRANCH.RESIDUAL_SKIP_AFTER_RELU True \

# cd ~/workspace/Gitlab/spconv/
# rm -rf build
# python setup.py bdist_wheel
# cd ./dist
# pip uninstall spconv -y
# pip install spconv-1.2.1-cp38-cp38-linux_x86_64.whl
# cd ~/workspace/Gitlab/detectron2/projects/DensePose

# cfg_name='densepose_CondInst_R_50_s1x_ft'
# # export CUDA_VISIBLE_DEVICES=1
# # CUDA_VISIBLE_DEVICES=1 
# python train_net.py --config-file configs/${cfg_name}.yaml \
#     --resume \
#     SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 \
#     OUTPUT_DIR ./output/${cfg_name}_1chSeg_IUVSparsePooler2Head_AggFea_V1ConvXGNSparseInsINLowMemNoOverlapTrue_GTinsDilated3_ftIUVheadOnly  \
#     MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS 1 \
#     MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS True \
#     SOLVER.CHECKPOINT_PERIOD 1000 \
#     DATALOADER.NUM_WORKERS 2 \
#     MODEL.CONDINST.IUVHead.NAME "IUVSparsePooler2Head" \
#     MODEL.ROI_DENSEPOSE_HEAD.LOSS_NAME "DensePoseChartGlobalIUVSeparatedSPoolerLoss" \
#     MODEL.ROI_DENSEPOSE_HEAD.NAME "DensePoseV1ConvXGNSparseGNHead" \
#     MODEL.ROI_DENSEPOSE_HEAD.CONV_HEAD_DIM 256 \
#     MODEL.CONDINST.MASK_BRANCH.AGG_CHANNELS 256 \
#     MODEL.CONDINST.IUVHead.MASK_OUT_BG_FEATURES "hard" \
#     MODEL.CONDINST.IUVHead.DILATE_FGMASK_KENERAL_SIZE 3 \
#     MODEL.ROI_DENSEPOSE_HEAD.DEEPLAB.NORM "InsIN" \
#     MODEL.CONDINST.IUVHead.USE_AGG_FEATURES True \
#     MODEL.CONDINST.IUVHead.INSTANCE_AWARE_GN True \
#     MODEL.CONDINST.IUVHead.REMOVE_MASK_OVERLAP True \
#     MODEL.CONDINST.FINETUNE_IUVHead_ONLY True \
#     MODEL.FCOS.INFERENCE_TH_TEST 0.2 \
#     MODEL.WEIGHTS "./output/densepose_CondInst_R_50_s1x_1chSeg_IUVSparsePooler2Head_AggFea_V1ConvXGNSparseInsINLowMemNoOverlapTrue_GTinsDilated3/model_final.pth"

#     # MODEL.FCOS.YIELD_PROPOSAL True \
#     # MODEL.CONDINST.IUVHead.GT_INSTANCES True \

# cfg_name='densepose_CondInst_R_50_s1x'
# CUDA_LAUNCH_BLOCKING=1 python train_net.py --config-file configs/${cfg_name}.yaml \
#     --resume \
#     SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 \
#     OUTPUT_DIR ./output/${cfg_name}_1chSeg_IUVSparsePooler2Head_AggFea_V1ConvXGNSparseInsINLowMemNoOverlapTrue256_GTinsDilated3_fewProposal_fewLayer  \
#     MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS 1 \
#     MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS True \
#     SOLVER.CHECKPOINT_PERIOD 5000 \
#     DATALOADER.NUM_WORKERS 2 \
#     MODEL.CONDINST.IUVHead.NAME "IUVSparsePooler2Head" \
#     MODEL.ROI_DENSEPOSE_HEAD.LOSS_NAME "DensePoseChartGlobalIUVSeparatedSPoolerLoss" \
#     MODEL.ROI_DENSEPOSE_HEAD.NAME "DensePoseV1ConvXGNSparseGNHead" \
#     MODEL.ROI_DENSEPOSE_HEAD.CONV_HEAD_DIM 256 \
#     MODEL.CONDINST.MASK_BRANCH.AGG_CHANNELS 256 \
#     MODEL.CONDINST.IUVHead.MASK_OUT_BG_FEATURES "hard" \
#     MODEL.CONDINST.IUVHead.DILATE_FGMASK_KENERAL_SIZE 3 \
#     MODEL.ROI_DENSEPOSE_HEAD.DEEPLAB.NORM "InsIN" \
#     MODEL.CONDINST.IUVHead.USE_AGG_FEATURES True \
#     MODEL.CONDINST.IUVHead.GT_INSTANCES True \
#     MODEL.CONDINST.IUVHead.INSTANCE_AWARE_GN True \
#     MODEL.CONDINST.IUVHead.REMOVE_MASK_OVERLAP True \
#     MODEL.CONDINST.MASK_BRANCH.NUM_CONVS 3\
#     MODEL.CONDINST.MAX_PROPOSALS 160 \
#     MODEL.FCOS.PRE_NMS_TOPK_TRAIN 160 \
#     MODEL.FCOS.PRE_NMS_TOPK_TEST 160 \
#     MODEL.FCOS.POST_NMS_TOPK_TRAIN 32 \
#     MODEL.FCOS.POST_NMS_TOPK_TEST 32 \
#     # MODEL.FCOS.INFERENCE_TH_TRAIN 0.05 \
#     # MODEL.FCOS.INFERENCE_TH_TEST 0.05 \

# _C.MODEL.FCOS.NUM_CLASSES = 80
# _C.MODEL.FCOS.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]
# _C.MODEL.FCOS.FPN_STRIDES = [8, 16, 32, 64, 128]
# _C.MODEL.FCOS.PRIOR_PROB = 0.01
# _C.MODEL.FCOS.INFERENCE_TH_TRAIN = 0.05
# _C.MODEL.FCOS.INFERENCE_TH_TEST = 0.05
# _C.MODEL.FCOS.NMS_TH = 0.6
# _C.MODEL.FCOS.PRE_NMS_TOPK_TRAIN = 1000
# _C.MODEL.FCOS.PRE_NMS_TOPK_TEST = 1000
# _C.MODEL.FCOS.POST_NMS_TOPK_TRAIN = 100
# _C.MODEL.FCOS.POST_NMS_TOPK_TEST = 100


