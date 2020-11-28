source ~/.bashrc_liqianma

# cfg_name='densepose_CondInst_R_50_BiFPN_s1x'
# CUDA_LAUNCH_BLOCKING=1 python train_net.py --config-file configs/${cfg_name}.yaml \
#     --resume \
#     SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 \
#     OUTPUT_DIR ./output/${cfg_name}_1chSeg_cropResizeNew_deeperIUVHead_RelCoordPE10 \
#     MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS 1 \
#     MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS True \
#     SOLVER.CHECKPOINT_PERIOD 5000 \
#     DATALOADER.NUM_WORKERS 2 \
#     MODEL.ROI_DENSEPOSE_HEAD.LOSS_NAME "DensePoseChartGlobalIUVSeparatedSCropResizeLoss" \
#     MODEL.CONDINST.IUVHead.NUM_CONVS 14 \
#     MODEL.CONDINST.IUVHead.REL_COORDS True \
#     MODEL.CONDINST.IUVHead.POSE_EMBEDDING_NUM_FREQS 10 \
#     # SOLVER.CLIP_GRADIENTS.ENABLED True \
#     # MODEL.ROI_DENSEPOSE_HEAD.NAME "DensePoseDeepLabHead" \
#     # MODEL.ROI_DENSEPOSE_HEAD.POINT_REGRESSION_WEIGHTS 1.0 \
#     # SOLVER.WARMUP_ITERS 2000
#     # SOLVER.CLIP_GRADIENTS.ENABLED True SOLVER.CLIP_GRADIENTS.CLIP_TYPE norm SOLVER.CLIP_GRADIENTS.CLIP_VALUE 100.0\


# cfg_name='densepose_CondInst_R_50_s1x'
# CUDA_LAUNCH_BLOCKING=1 python train_net.py --config-file configs/${cfg_name}.yaml \
#   --resume \
#     SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 \
#     OUTPUT_DIR ./output/${cfg_name}_1chSeg_cropResizeNew_deepIUVHead_AbsRelCoordPE10 \
#     MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS 1 \
#     MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS True \
#     SOLVER.CHECKPOINT_PERIOD 5000 \
#     DATALOADER.NUM_WORKERS 2 \
#   MODEL.ROI_DENSEPOSE_HEAD.LOSS_NAME "DensePoseChartGlobalIUVSeparatedSCropResizeLoss" \
#   MODEL.CONDINST.IUVHead.NUM_CONVS 7 \
#   MODEL.CONDINST.IUVHead.ABS_COORDS True \
#   MODEL.CONDINST.IUVHead.REL_COORDS True \
#     MODEL.CONDINST.IUVHead.POSE_EMBEDDING_NUM_FREQS 10 \
#     # MODEL.CONDINST.IUVHead.NUM_LAMBDA_LAYER 2 \
#     # SOLVER.CLIP_GRADIENTS.ENABLED True \
#   # MODEL.ROI_DENSEPOSE_HEAD.NAME "DensePoseDeepLabHead" \
#   # MODEL.ROI_DENSEPOSE_HEAD.POINT_REGRESSION_WEIGHTS 1.0 \
#   # SOLVER.WARMUP_ITERS 2000
#     # SOLVER.CLIP_GRADIENTS.ENABLED True SOLVER.CLIP_GRADIENTS.CLIP_TYPE norm SOLVER.CLIP_GRADIENTS.CLIP_VALUE 100.0\


# cfg_name='densepose_CondInst_R_50_BiFPN_s1x'
# CUDA_LAUNCH_BLOCKING=1 python train_net.py --config-file configs/${cfg_name}.yaml \
#     --resume \
#     SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 \
#     OUTPUT_DIR ./output/${cfg_name}_1chSeg_cropResizeNew_deepIUVHead_lambda1_AbsRelCoordPE10MultiLayer \
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
#     MODEL.ROI_DENSEPOSE_HEAD.NAME "DensePoseMultilayercoordHead" \


# cfg_name='densepose_CondInst_R_50_BiFPN_s1x'
# CUDA_LAUNCH_BLOCKING=1 python train_net.py --config-file configs/${cfg_name}.yaml \
#     --resume \
#     SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 \
#     OUTPUT_DIR ./output/${cfg_name}_1chSeg_cropResizeNew_deepWide2IUVHeadDownUp_lambda1_AbsRelCoordPE10_GTins_dpFilter \
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
#     MODEL.CONDINST.IUVHead.DOWN_UP_SAMPLING True \
#     # MODEL.CONDINST.IUVHead.LAMBDA_LAYER_R 51 \
#     # MODEL.ROI_DENSEPOSE_HEAD.NAME "DensePoseMultilayermaskHead" \



# cfg_name='densepose_CondInst_R_50_s1x'
# CUDA_LAUNCH_BLOCKING=1 python train_net.py --config-file configs/${cfg_name}.yaml \
#     --resume \
#     SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 \
#     OUTPUT_DIR ./output/${cfg_name}_1chSeg_IUVSparsePooler2Head_V1ConvXBNSparse256_GTinsErode3Dilated6_AbsRelCoordPE10_normCoordBoxHW_v2  \
#     MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS 1 \
#     MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS True \
#     SOLVER.CHECKPOINT_PERIOD 2000 \
#     DATALOADER.NUM_WORKERS 2 \
#     MODEL.CONDINST.IUVHead.NAME "IUVSparsePooler2Head" \
#     MODEL.ROI_DENSEPOSE_HEAD.LOSS_NAME "DensePoseChartGlobalIUVSeparatedSPoolerLoss" \
#     MODEL.ROI_DENSEPOSE_HEAD.NAME "DensePoseV1ConvXGNSparseHead" \
#     MODEL.ROI_DENSEPOSE_HEAD.CONV_HEAD_DIM 256 \
#     MODEL.CONDINST.IUVHead.MASK_OUT_BG_FEATURES "hard" \
#     MODEL.CONDINST.IUVHead.DILATE_FGMASK_KENERAL_SIZE 3 \
#     MODEL.ROI_DENSEPOSE_HEAD.DEEPLAB.NORM "BN" \
#     MODEL.CONDINST.IUVHead.ABS_COORDS True \
#     MODEL.CONDINST.IUVHead.REL_COORDS True \
#     MODEL.CONDINST.IUVHead.POSE_EMBEDDING_NUM_FREQS 10 \
#     MODEL.CONDINST.IUVHead.GT_INSTANCES True \
#     # MODEL.CONDINST.IUVHead.NUM_LAMBDA_LAYER 1 \


# cd ~/workspace/Gitlab/spconv/
# rm -rf build
# python setup.py bdist_wheel
# cd ./dist
# pip uninstall spconv -y
# pip install spconv-1.2.1-cp38-cp38-linux_x86_64.whl
# cd ~/workspace/Gitlab/detectron2/projects/DensePose
 
# cfg_name='densepose_CondInst_R_50_s1x'
# CUDA_LAUNCH_BLOCKING=1 python train_net.py --config-file configs/${cfg_name}.yaml \
#     --resume \
#     SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 \
#     OUTPUT_DIR ./output/${cfg_name}_1chSeg_IUVSparsePooler2Head_AggFea_V1ConvXGNSparseInsINLowMemNoOverlapTrueRes3_GTinsDilated3  \
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
#     MODEL.CONDINST.IUVHead.RESIDUAL_SKIP True \
#     # MODEL.CONDINST.IUVHead.Efficient_Channel_Attention True \
#     # MODEL.CONDINST.IUVHead.WEIGHT_STANDARDIZATION True \
#     # MODEL.BACKBONE.NAME "build_fcos_resnet_fpnws_backbone" \
 

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
 
# cfg_name='densepose_CondInst_R_50_s1x'
# CUDA_LAUNCH_BLOCKING=1 python train_net.py --config-file configs/${cfg_name}.yaml \
#     --resume --num-gpus 1 \
#     SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 SOLVER.ACCUMULATE_GRAD_ITER 8 \
#     OUTPUT_DIR ./output/${cfg_name}_1chSeg_IUVSparsePooler2Head_AggFea_V1ConvXGNSparseInsINLowMemNoOverlapTrueResInput_GTinsDilated3_amp_BS2x8  \
#     MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS 1 \
#     MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS True \
#     SOLVER.CHECKPOINT_PERIOD 2000 \
#     DATALOADER.NUM_WORKERS 4 \
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
#     MODEL.CONDINST.v2 True \
#     SOLVER.AMP.ENABLED True \
#     MODEL.CONDINST.IUVHead.RESIDUAL_INPUT True \
#     MODEL.CONDINST.MASK_BRANCH.RESIDUAL_SKIP_AFTER_RELU True \

cfg_name='densepose_CondInst_R_50_s1x'
CUDA_LAUNCH_BLOCKING=1 python train_net.py --config-file configs/${cfg_name}.yaml \
    --resume  --num-gpus 1 \
    SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 SOLVER.ACCUMULATE_GRAD_ITER 1 \
    OUTPUT_DIR ./output/${cfg_name}_1chSeg_IUVSparsePooler2Head_AggFea_V1ConvXGNSparseInsINLowMemNoOverlapTrueResInputInsECA_resIUVOnly_GTinsDilated3_amp  \
    MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS 1 \
    MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS True \
    SOLVER.CHECKPOINT_PERIOD 2000 \
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
    MODEL.CONDINST.IUVHead.INSTANCE_EFFICIENT_CHANNEL_ATTENTION True \
    # MODEL.CONDINST.MASK_BRANCH.RESIDUAL_SKIP_AFTER_RELU True \
# MODEL.CONDINST.IUVHead.DILATION_CONV


# cfg_name='densepose_CondInst_R_50_s1x'
# CUDA_LAUNCH_BLOCKING=1 python train_net.py --config-file configs/${cfg_name}.yaml \
#     --resume \
#     SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 \
#     OUTPUT_DIR ./output/${cfg_name}_1chSeg_IUVSparsePooler2Head_AggFea_V1ConvXBNSparseInsINLowMemNoOverlapTrue_GTinsDilated3_RelCoordPE5_normCoordBoxHW_v3sumcoord  \
#     MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS 1 \
#     MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS True \
#     SOLVER.CHECKPOINT_PERIOD 5000 \
#     DATALOADER.NUM_WORKERS 2 \
#     MODEL.CONDINST.IUVHead.NAME "IUVSparsePooler2Head" \
#     MODEL.ROI_DENSEPOSE_HEAD.LOSS_NAME "DensePoseChartGlobalIUVSeparatedSPoolerLoss" \
#     MODEL.ROI_DENSEPOSE_HEAD.NAME "DensePoseV1ConvXGNSparseGNHead" \
#     MODEL.ROI_DENSEPOSE_HEAD.CONV_HEAD_DIM 256 \
#     MODEL.CONDINST.MASK_BRANCH.AGG_CHANNELS 256 \
#     MODEL.CONDINST.IUVHead.GT_INSTANCES True \
#     MODEL.CONDINST.IUVHead.MASK_OUT_BG_FEATURES "hard" \
#     MODEL.CONDINST.IUVHead.DILATE_FGMASK_KENERAL_SIZE 3 \
#     MODEL.ROI_DENSEPOSE_HEAD.DEEPLAB.NORM "InsIN" \
#     MODEL.CONDINST.IUVHead.USE_AGG_FEATURES True \
#     MODEL.CONDINST.IUVHead.GT_INSTANCES True \
#     MODEL.CONDINST.IUVHead.INSTANCE_AWARE_GN True \
#     MODEL.CONDINST.IUVHead.REL_COORDS True \
#     MODEL.CONDINST.IUVHead.POSE_EMBEDDING_NUM_FREQS 5 \
#     MODEL.CONDINST.IUVHead.REMOVE_MASK_OVERLAP True \
#     # MODEL.CONDINST.IUVHead.ABS_COORDS True \
#     # MODEL.CONDINST.MASK_BRANCH.USE_ASPP True
#     # MODEL.CONDINST.IUVHead.NUM_LAMBDA_LAYER 1 \

# cfg_name='densepose_CondInst_R_50_s1x'
# CUDA_LAUNCH_BLOCKING=1 python train_net.py --config-file configs/${cfg_name}.yaml \
#     --resume \
#     SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 \
#     OUTPUT_DIR ./output/${cfg_name}_1chSeg_IUVSparsePooler2Head_AggFea_V1ConvXGNSparseInsGNLowMemNoOverlap_GTinsDilated3  \
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
#     MODEL.ROI_DENSEPOSE_HEAD.DEEPLAB.NORM "InsGN" \
#     MODEL.CONDINST.IUVHead.USE_AGG_FEATURES True \
#     MODEL.CONDINST.IUVHead.GT_INSTANCES True \
#     MODEL.CONDINST.IUVHead.INSTANCE_AWARE_GN True \
#     MODEL.CONDINST.IUVHead.REMOVE_MASK_OVERLAP True \

# cfg_name='densepose_CondInst_R_50_s1x'
# CUDA_LAUNCH_BLOCKING=1 python train_net.py --config-file configs/${cfg_name}.yaml \
#     --resume \
#     SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 \
#     OUTPUT_DIR ./output/${cfg_name}_1chSeg_IUVSparsePooler2Head_V1ConvXBNSparse256_GTinsDilated5  \
#     MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS 1 \
#     MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS True \
#     SOLVER.CHECKPOINT_PERIOD 2000 \
#     DATALOADER.NUM_WORKERS 2 \
#     MODEL.CONDINST.IUVHead.NAME "IUVSparsePooler2Head" \
#     MODEL.ROI_DENSEPOSE_HEAD.LOSS_NAME "DensePoseChartGlobalIUVSeparatedSPoolerLoss" \
#     MODEL.ROI_DENSEPOSE_HEAD.NAME "DensePoseV1ConvXGNSparseHead" \
#     MODEL.ROI_DENSEPOSE_HEAD.CONV_HEAD_DIM 256 \
#     MODEL.CONDINST.IUVHead.GT_INSTANCES True \
#     MODEL.CONDINST.IUVHead.MASK_OUT_BG_FEATURES "hard" \
#     MODEL.CONDINST.IUVHead.DILATE_FGMASK_KENERAL_SIZE 5 \
#     MODEL.ROI_DENSEPOSE_HEAD.DEEPLAB.NORM "BN" \
#     # MODEL.CONDINST.IUVHead.NUM_LAMBDA_LAYER 1 \
#     # MODEL.CONDINST.IUVHead.ABS_COORDS True \
#     # MODEL.CONDINST.IUVHead.REL_COORDS True \
#     # MODEL.CONDINST.IUVHead.POSE_EMBEDDING_NUM_FREQS 10 \


# cfg_name='densepose_CondInst_R_50_s1x'
# CUDA_LAUNCH_BLOCKING=1 python train_net.py --config-file configs/${cfg_name}.yaml \
#     --resume \
#     SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 \
#     OUTPUT_DIR ./output/${cfg_name}_1chSeg_IUVSparsePooler2Head_AggFea_V1ConvXGNSparseGN256_GTinsDilated3_AbsRelCoordPE10_normCoordBoxHW  \
#     MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS 1 \
#     MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS True \
#     SOLVER.CHECKPOINT_PERIOD 5000 \
#     DATALOADER.NUM_WORKERS 2 \
#     MODEL.CONDINST.IUVHead.NAME "IUVSparsePooler2Head" \
#     MODEL.ROI_DENSEPOSE_HEAD.LOSS_NAME "DensePoseChartGlobalIUVSeparatedSPoolerLoss" \
#     MODEL.ROI_DENSEPOSE_HEAD.NAME "DensePoseV1ConvXGNSparseGNHead" \
#     MODEL.ROI_DENSEPOSE_HEAD.CONV_HEAD_DIM 256 \
#     MODEL.CONDINST.MASK_BRANCH.AGG_CHANNELS 256 \
#     MODEL.CONDINST.IUVHead.GT_INSTANCES True \
#     MODEL.CONDINST.IUVHead.MASK_OUT_BG_FEATURES "hard" \
#     MODEL.CONDINST.IUVHead.DILATE_FGMASK_KENERAL_SIZE 3 \
#     MODEL.CONDINST.IUVHead.USE_AGG_FEATURES True \
#     MODEL.CONDINST.IUVHead.ABS_COORDS True \
#     MODEL.CONDINST.IUVHead.REL_COORDS True \
#     MODEL.CONDINST.IUVHead.POSE_EMBEDDING_NUM_FREQS 10 \
#     MODEL.CONDINST.MASK_BRANCH.NUM_LAMBDA_LAYER 1 \
#     # MODEL.CONDINST.MASK_BRANCH.USE_ASPP True \
#     # MODEL.CONDINST.IUVHead.NUM_LAMBDA_LAYER 1 \
#     # MODEL.ROI_DENSEPOSE_HEAD.DEEPLAB.NORM "GN" \