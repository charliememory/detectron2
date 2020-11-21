source ~/.bashrc_liqianma

# cfg_name='densepose_CondInst_R_101_s1x'
# CUDA_LAUNCH_BLOCKING=1 python train_net.py --config-file configs/${cfg_name}.yaml \
# 	--resume \
#     SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 \
#     OUTPUT_DIR ./output/${cfg_name}_1chSeg_RelCoordDetach \
#     MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS 1 \
#     MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS True \
#     SOLVER.CHECKPOINT_PERIOD 2000 \
#     MODEL.CONDINST.IUVHead.REL_COORDS True \
#     DATALOADER.NUM_WORKERS 2 \
#     # MODEL.CONDINST.IUVHead.NUM_CONVS 7 \
#     # MODEL.ROI_DENSEPOSE_HEAD.INDEX_WEIGHTS 3.0 \
#     # MODEL.WEIGHTS ../../pretrain/CondInst_MS_R_101_3x_sem.pth \
#     # MODEL.CONDINST.MASK_BRANCH.NORM "BN"\
#     # MODEL.ROI_DENSEPOSE_HEAD.POINT_REGRESSION_WEIGHTS 3.0 \
#     # MODEL.ROI_DENSEPOSE_HEAD.INDEX_WEIGHTS 0.
#     # DATALOADER.NUM_WORKERS 1 \

# cfg_name='densepose_CondInst_R_50_s1x'
# CUDA_LAUNCH_BLOCKING=1 python train_net.py --config-file configs/${cfg_name}.yaml \
# 	--resume \
#     SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 \
#     OUTPUT_DIR ./output/${cfg_name}_1chSeg_uvMean_cropResize_deepIUVHead \
#     MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS 1 \
#     MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS True \
# 	MODEL.ROI_DENSEPOSE_HEAD.POINT_REGRESSION_WEIGHTS 1.0 \
#     SOLVER.CHECKPOINT_PERIOD 5000 \
#     DATALOADER.NUM_WORKERS 2 \
# 	MODEL.ROI_DENSEPOSE_HEAD.LOSS_NAME "DensePoseChartGlobalIUVSeparatedSCropResizeLoss" \
# 	MODEL.CONDINST.IUVHead.NUM_CONVS 7 \
# 	# MODEL.ROI_DENSEPOSE_HEAD.NAME "DensePoseDeepLabHead" \
# 	# MODEL.CONDINST.IUVHead.REL_COORDS True \

# cfg_name='densepose_CondInst_R_50_s1x'
# CUDA_LAUNCH_BLOCKING=1 python train_net.py --config-file configs/${cfg_name}.yaml \
# 	--resume \
#     SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 SOLVER.WARMUP_ITERS 4000 \
#     OUTPUT_DIR ./output/${cfg_name}_1chSeg_cropResize_deepLabHead_warmup4k \
#     MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS 1 \
#     MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS True \
#     SOLVER.CHECKPOINT_PERIOD 1000 \
#     DATALOADER.NUM_WORKERS 2 \
# 	MODEL.ROI_DENSEPOSE_HEAD.LOSS_NAME "DensePoseChartGlobalIUVSeparatedSCropResizeLoss" \
# 	MODEL.ROI_DENSEPOSE_HEAD.NAME "DensePoseDeepLabHead" \
#     # MODEL.CONDINST.IUVHead.NUM_LAMBDA_LAYER 1 \
# 	# MODEL.ROI_DENSEPOSE_HEAD.POINT_REGRESSION_WEIGHTS 1.0 \
# 	# MODEL.CONDINST.IUVHead.NUM_CONVS 7 \
# 	# MODEL.CONDINST.IUVHead.REL_COORDS True \


# cfg_name='densepose_CondInst_R_50_BiFPN_s1x'
# CUDA_LAUNCH_BLOCKING=1 python train_net.py --config-file configs/${cfg_name}.yaml \
#     --resume \
#     SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 \
#     OUTPUT_DIR ./output/${cfg_name}_1chSeg_cropResizeNew_deepIUVHead_lambda1_AbsRelCoordPE10 \
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
#     # MODEL.CONDINST.IUVHead.CHANNELS 256 MODEL.CONDINST.MASK_BRANCH.CHANNELS 256\
#     # SOLVER.CLIP_GRADIENTS.ENABLED True \
#     # MODEL.ROI_DENSEPOSE_HEAD.NAME "DensePoseDeepLabHead" \
#     # MODEL.ROI_DENSEPOSE_HEAD.POINT_REGRESSION_WEIGHTS 1.0 \
#     # SOLVER.WARMUP_ITERS 2000
#     # SOLVER.CLIP_GRADIENTS.ENABLED True SOLVER.CLIP_GRADIENTS.CLIP_TYPE norm SOLVER.CLIP_GRADIENTS.CLIP_VALUE 100.0\


# cfg_name='densepose_CondInst_R_50_BiFPN_s1x'
# CUDA_LAUNCH_BLOCKING=1 python train_net.py --config-file configs/${cfg_name}.yaml \
#     --resume \
#     SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 \
#     OUTPUT_DIR ./output/${cfg_name}_1chSeg_cropResizeNew_deepIUVHead_lambda1_AbsRelCoord \
#     MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS 1 \
#     MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS True \
#     SOLVER.CHECKPOINT_PERIOD 5000 \
#     DATALOADER.NUM_WORKERS 2 \
#     MODEL.ROI_DENSEPOSE_HEAD.LOSS_NAME "DensePoseChartGlobalIUVSeparatedSCropResizeLoss" \
#     MODEL.CONDINST.IUVHead.NUM_CONVS 7 \
#     MODEL.CONDINST.IUVHead.ABS_COORDS True \
#     MODEL.CONDINST.IUVHead.REL_COORDS True \
#     MODEL.CONDINST.IUVHead.NUM_LAMBDA_LAYER 1 \
#     # MODEL.CONDINST.IUVHead.POSE_EMBEDDING_NUM_FREQS 10 \

# cfg_name='densepose_CondInst_R_50_BiFPN_s1x'
# CUDA_LAUNCH_BLOCKING=1 python train_net.py --config-file configs/${cfg_name}.yaml \
#     --resume \
#     SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 \
#     OUTPUT_DIR ./output/${cfg_name}_1chSeg_cropResizeNew_deepIUVHead_lambda1_RelCoordPE10 \
#     MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS 1 \
#     MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS True \
#     SOLVER.CHECKPOINT_PERIOD 5000 \
#     DATALOADER.NUM_WORKERS 2 \
#     MODEL.ROI_DENSEPOSE_HEAD.LOSS_NAME "DensePoseChartGlobalIUVSeparatedSCropResizeLoss" \
#     MODEL.CONDINST.IUVHead.NUM_CONVS 7 \
#     MODEL.CONDINST.IUVHead.REL_COORDS True \
#     MODEL.CONDINST.IUVHead.POSE_EMBEDDING_NUM_FREQS 10 \
#     MODEL.CONDINST.IUVHead.NUM_LAMBDA_LAYER 1 \
#     # MODEL.CONDINST.IUVHead.ABS_COORDS True \
#     # MODEL.CONDINST.IUVHead.CHANNELS 256 MODEL.CONDINST.MASK_BRANCH.CHANNELS 256\
#     # SOLVER.CLIP_GRADIENTS.ENABLED True \
#     # MODEL.ROI_DENSEPOSE_HEAD.NAME "DensePoseDeepLabHead" \
#     # MODEL.ROI_DENSEPOSE_HEAD.POINT_REGRESSION_WEIGHTS 1.0 \
#     # SOLVER.WARMUP_ITERS 2000
#     # SOLVER.CLIP_GRADIENTS.ENABLED True SOLVER.CLIP_GRADIENTS.CLIP_TYPE norm SOLVER.CLIP_GRADIENTS.CLIP_VALUE 100.0\

# cfg_name='densepose_CondInst_R_50_BiFPN_s1x'
# CUDA_LAUNCH_BLOCKING=1 python train_net.py --config-file configs/${cfg_name}.yaml \
#     --resume \
#     SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 \
#     OUTPUT_DIR ./output/${cfg_name}_1chSeg_cropResizeNew_deepWide2IUVHead_lambda1_AbsRelCoordPE10_normCoordBoxHW \
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
#     MODEL.CONDINST.IUVHead.NORM_COORD_BOXHW True \
#     # MODEL.CONDINST.IUVHead.GT_INSTANCES True \

# cfg_name='densepose_CondInst_R_50_BiFPN_s1x'
# CUDA_LAUNCH_BLOCKING=1 python train_net.py --config-file configs/${cfg_name}.yaml \
#     --resume \
#     SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 \
#     OUTPUT_DIR ./output/${cfg_name}_1chSeg_cropResizeNew_deepWide2IUVHeadBN_lambda1_AbsRelCoordPE10_GTins_normCoordBoxHW_hardMaskBG_parConv \
#     MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS 1 \
#     MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS True \
#     SOLVER.CHECKPOINT_PERIOD 5000 \
#     DATALOADER.NUM_WORKERS 4 \
#     MODEL.ROI_DENSEPOSE_HEAD.LOSS_NAME "DensePoseChartGlobalIUVSeparatedSCropResizeLoss" \
#     MODEL.CONDINST.IUVHead.NUM_CONVS 7 \
#     MODEL.CONDINST.MASK_BRANCH.AGG_CHANNELS 256\
#     MODEL.CONDINST.IUVHead.CHANNELS 256 \
#     MODEL.CONDINST.IUVHead.ABS_COORDS True \
#     MODEL.CONDINST.IUVHead.REL_COORDS True \
#     MODEL.CONDINST.IUVHead.POSE_EMBEDDING_NUM_FREQS 10 \
#     MODEL.CONDINST.IUVHead.NUM_LAMBDA_LAYER 1 \
#     MODEL.CONDINST.IUVHead.GT_INSTANCES True \
#     MODEL.CONDINST.IUVHead.NORM_COORD_BOXHW True \
#     MODEL.CONDINST.IUVHead.OUT_CHANNELS 78 \
#     MODEL.CONDINST.IUVHead.MASK_OUT_BG_FEATURES 'hard' \
#     MODEL.CONDINST.IUVHead.PARTIAL_CONV True \
#     MODEL.CONDINST.IUVHead.NORM 'BN' \
#     # MODEL.ROI_DENSEPOSE_HEAD.AUX_REL_COORDS_WEIGHTS 1.0 \
#     # MODEL.ROI_DENSEPOSE_HEAD.PART_WEIGHTS 2.0\
#     # MODEL.ROI_DENSEPOSE_HEAD.POINT_REGRESSION_WEIGHTS 0.005\
#     # SOLVER.CLIP_GRADIENTS.ENABLED True \
#     # MODEL.ROI_DENSEPOSE_HEAD.NAME "DensePoseDeepLabHead" \
#     # MODEL.ROI_DENSEPOSE_HEAD.POINT_REGRESSION_WEIGHTS 1.0 \
#     # SOLVER.WARMUP_ITERS 2000
#     # SOLVER.CLIP_GRADIENTS.ENABLED True SOLVER.CLIP_GRADIENTS.CLIP_TYPE norm SOLVER.CLIP_GRADIENTS.CLIP_VALUE 100.0\
 
# cfg_name='densepose_rcnn_R_50_FPN_CondInst_s1x'
# CUDA_LAUNCH_BLOCKING=1 python train_net.py --config-file configs/${cfg_name}.yaml \
#     --resume \
#     SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 \
#     OUTPUT_DIR ./output/${cfg_name}_noInsLoss_noRoiSLoss \
#     MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS True \
#     SOLVER.CHECKPOINT_PERIOD 5000 \
#     DATALOADER.NUM_WORKERS 2 \
#     MODEL.CONDINST.IUVHead.DISABLE True \
#     MODEL.CONDINST.MASK_BRANCH.AGG_CHANNELS 256\
#     MODEL.CONDINST.IUVHead.CHANNELS 256 \
#     MODEL.ROI_DENSEPOSE_HEAD.DECODER_ON True \
#     # MODEL.ROI_DENSEPOSE_HEAD.POINT_REGRESSION_WEIGHTS \
#     # MODEL.ROI_DENSEPOSE_HEAD.DECODER_CONV_DIMS 128
#     # MODEL.ROI_DENSEPOSE_HEAD.NAME "DensePoseMultilayermaskHead" \
#     # MODEL.ROI_DENSEPOSE_HEAD.LOSS_NAME "DensePoseChartGlobalIUVSeparatedSCropResizeLoss" \
#     # MODEL.CONDINST.IUVHead.NUM_CONVS 7 \
#     # MODEL.CONDINST.MASK_BRANCH.AGG_CHANNELS 256\
#     # MODEL.CONDINST.IUVHead.CHANNELS 256 \
#     # MODEL.CONDINST.IUVHead.ABS_COORDS True \
#     # MODEL.CONDINST.IUVHead.REL_COORDS True \
#     # MODEL.CONDINST.IUVHead.POSE_EMBEDDING_NUM_FREQS 10 \
#     # MODEL.CONDINST.IUVHead.NUM_LAMBDA_LAYER 1 \
#     # MODEL.CONDINST.IUVHead.GT_INSTANCES True \
#     # MODEL.CONDINST.IUVHead.NORM_COORD_BOXHW True \
#     # MODEL.CONDINST.IUVHead.OUT_CHANNELS 78 \
#     # MODEL.CONDINST.IUVHead.MASK_OUT_BG_FEATURES 'hard' \
#     # MODEL.CONDINST.IUVHead.PARTIAL_CONV True \
#     # MODEL.CONDINST.IUVHead.NORM 'BN' \
#     # MODEL.ROI_DENSEPOSE_HEAD.AUX_REL_COORDS_WEIGHTS 1.0 \
#     # MODEL.ROI_DENSEPOSE_HEAD.PART_WEIGHTS 2.0\
#     # MODEL.ROI_DENSEPOSE_HEAD.POINT_REGRESSION_WEIGHTS 0.005\
#     # SOLVER.CLIP_GRADIENTS.ENABLED True \
#     # MODEL.ROI_DENSEPOSE_HEAD.NAME "DensePoseDeepLabHead" \
#     # MODEL.ROI_DENSEPOSE_HEAD.POINT_REGRESSION_WEIGHTS 1.0 \
#     # SOLVER.WARMUP_ITERS 2000
#     # SOLVER.CLIP_GRADIENTS.ENABLED True SOLVER.CLIP_GRADIENTS.CLIP_TYPE norm SOLVER.CLIP_GRADIENTS.CLIP_VALUE 100.0\
 
# cfg_name='densepose_CondInst_R_50_s1x'
# CUDA_LAUNCH_BLOCKING=1 python train_net.py --config-file configs/${cfg_name}.yaml \
#     --resume \
#     SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 \
#     OUTPUT_DIR ./output/${cfg_name}_1chSeg_IUVSparsePooler2Head_V1ConvXGNSparse256_lambda1True_AbsRelCoordPE10_GTins_normCoordBoxHW \
#     MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS 1 \
#     MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS True \
#     SOLVER.CHECKPOINT_PERIOD 2000 \
#     DATALOADER.NUM_WORKERS 2 \
#     MODEL.CONDINST.IUVHead.NAME "IUVSparsePooler2Head" \
#     MODEL.ROI_DENSEPOSE_HEAD.LOSS_NAME "DensePoseChartGlobalIUVSeparatedSPoolerLoss" \
#     MODEL.ROI_DENSEPOSE_HEAD.NAME "DensePoseV1ConvXGNSparseHead" \
#     MODEL.ROI_DENSEPOSE_HEAD.CONV_HEAD_DIM 256 \
#     MODEL.CONDINST.IUVHead.ABS_COORDS True \
#     MODEL.CONDINST.IUVHead.REL_COORDS True \
#     MODEL.CONDINST.IUVHead.POSE_EMBEDDING_NUM_FREQS 10 \
#     MODEL.CONDINST.IUVHead.NUM_LAMBDA_LAYER 1 \
#     MODEL.CONDINST.IUVHead.GT_INSTANCES True \
#     MODEL.CONDINST.IUVHead.NORM_COORD_BOXHW True \
#     MODEL.CONDINST.IUVHead.MASK_OUT_BG_FEATURES "hard" \
#     MODEL.ROI_DENSEPOSE_HEAD.DEEPLAB.NORM "BN" \
#     # MODEL.ROI_DENSEPOSE_HEAD.NAME "DensePoseV1ConvXHead" \
#     # MODEL.CONDINST.IUVHead.NUM_CONVS 7 \
#     # MODEL.CONDINST.MASK_BRANCH.AGG_CHANNELS 256\
#     # MODEL.CONDINST.IUVHead.CHANNELS 256 \
#     # MODEL.ROI_DENSEPOSE_HEAD.INDEX_WEIGHTS 0. \
#     # MODEL.ROI_DENSEPOSE_HEAD.POINT_REGRESSION_WEIGHTS 1.0 \
#     # MODEL.CONDINST.IUVHead.ABS_COORDS True \

# cfg_name='densepose_CondInst_R_50_s1x'
# CUDA_LAUNCH_BLOCKING=1 python train_net.py --config-file configs/${cfg_name}.yaml \
#     --resume \
#     SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 \
#     OUTPUT_DIR ./output/${cfg_name}_1chSeg_IUVSparsePooler2Head_V1ConvXGNSparse256_GTins \
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
#     MODEL.ROI_DENSEPOSE_HEAD.DEEPLAB.NORM "GN" \
#     # MODEL.CONDINST.IUVHead.DILATE_FGMASK_KENERAL_SIZE 3 \


# cfg_name='densepose_CondInst_R_50_lowres_s1x'
# CUDA_LAUNCH_BLOCKING=1 python train_net.py --config-file configs/${cfg_name}.yaml \
#     --resume \
#     SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 \
#     OUTPUT_DIR ./output/${cfg_name}_1chSeg_IUVSparsePooler2Head_AggFea_V1ConvXBNSparseInsINLowMemNoOverlap_GTinsDilated3_RelCoordPE5_normCoordBoxHW_v3sumcoord  \
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
#     --resume \
#     SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 \
#     OUTPUT_DIR ./output/${cfg_name}_1chSeg_IUVSparsePooler2Head_AggFea_V1ConvXGNSparseInsINLowMemNoOverlapTrue_GTinsDilated3_ckptGrad1  \
#     MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS 1 \
#     MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS True \
#     SOLVER.CHECKPOINT_PERIOD 5000 \
#     DATALOADER.NUM_WORKERS 2 \
#     MODEL.CONDINST.IUVHead.NAME "IUVPooler2Head" \
#     MODEL.ROI_DENSEPOSE_HEAD.LOSS_NAME "DensePoseChartGlobalIUVSeparatedSPoolerLoss" \
#     MODEL.ROI_DENSEPOSE_HEAD.NAME "DensePoseDeepLabHead" \
#     MODEL.ROI_DENSEPOSE_HEAD.CONV_HEAD_DIM 256 \
#     MODEL.CONDINST.MASK_BRANCH.AGG_CHANNELS 256 \
#     MODEL.CONDINST.IUVHead.MASK_OUT_BG_FEATURES "hard" \
#     MODEL.CONDINST.IUVHead.DILATE_FGMASK_KENERAL_SIZE 3 \
#     MODEL.CONDINST.IUVHead.USE_AGG_FEATURES True \
#     MODEL.CONDINST.IUVHead.GT_INSTANCES True \
#     MODEL.CONDINST.v2 True \
#     SOLVER.AMP.ENABLED True \
#     MODEL.CONDINST.CHECKPOINT_GRAD_NUM 1\

#     # MODEL.CONDINST.IUVHead.NAME "IUVSparsePooler2Head" \
#     # MODEL.ROI_DENSEPOSE_HEAD.NAME "DensePoseV1ConvXGNSparseGNHead" \
#     # MODEL.ROI_DENSEPOSE_HEAD.DEEPLAB.NORM "InsIN" \
#     # MODEL.CONDINST.IUVHead.INSTANCE_AWARE_GN True \
#     # MODEL.CONDINST.IUVHead.REMOVE_MASK_OVERLAP True \

# cfg_name='densepose_CondInst_R_50_s1x'
# CUDA_LAUNCH_BLOCKING=1 python train_net.py --config-file configs/${cfg_name}.yaml \
#     --resume \
#     SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.0025 \
#     OUTPUT_DIR ./output/${cfg_name}_1chSeg_IUVSparsePooler2Head_AggFea_V1ConvXGNSparseInsINLowMemNoOverlapTrue_GTinsDilated3_amp_BS8  \
#     MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS 1 \
#     MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS True \
#     SOLVER.CHECKPOINT_PERIOD 5000 \
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
#     # MODEL.CONDINST.CHECKPOINT_GRAD_NUM 1\
  
cfg_name='densepose_CondInst_R_50_s1x'
CUDA_LAUNCH_BLOCKING=1 python train_net.py --config-file configs/${cfg_name}.yaml \
    --resume  --num-gpus 1 \
    SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.0025 SOLVER.ACCUMULATE_GRAD_ITER 2 \
    OUTPUT_DIR ./output/${cfg_name}_1chSeg_IUVSparsePooler2Head_AggFea_V1ConvXGNSparseInsINLowMemNoOverlapTrueResInput_GTinsDilated3_amp_BS8x2  \
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
    MODEL.CONDINST.MASK_BRANCH.RESIDUAL_SKIP_AFTER_RELU True \

    # MODEL.CONDINST.CHECKPOINT_GRAD_NUM 1\


# cfg_name='densepose_CondInst_R_50_s1x'
# CUDA_LAUNCH_BLOCKING=1 python train_net.py --config-file configs/${cfg_name}.yaml \
#     --resume \
#     SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 \
#     OUTPUT_DIR ./output/${cfg_name}_1chSeg_IUVSparsePooler2Head_AggFea_V1ConvXGNSparseInsINLowMemNoOverlapTrue_GTinsDilated3_ckptGrad1  \
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
#     MODEL.CONDINST.CHECKPOINT_GRAD_NUM 1\
#     MODEL.CONDINST.v2 True \

# cfg_name='densepose_CondInst_R_50_s1x'
# CUDA_LAUNCH_BLOCKING=1 python train_net.py --config-file configs/${cfg_name}.yaml \
#     --resume \
#     SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 \
#     OUTPUT_DIR ./output/${cfg_name}_1chSeg_IUVSparsePooler2Head_V1ConvXGNSparse256Dilate3 \
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
#     MODEL.ROI_DENSEPOSE_HEAD.DEEPLAB.NORM "BN" \
#     MODEL.CONDINST.IUVHead.DILATE_FGMASK_KENERAL_SIZE 3 \



# cfg_name='densepose_CondInst_R_50_BiFPN_s1x'
# CUDA_LAUNCH_BLOCKING=1 python train_net.py --config-file configs/${cfg_name}.yaml \
#     --resume \
#     SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 \
#     OUTPUT_DIR ./output/${cfg_name}_1chSeg_cropResizeNew_deepWide2IUVHeadBN_lambda1_AbsRelCoordPE10_GTins_normCoordBoxHW_hardMaskGNOnlyMultiLayer \
#     MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS 1 \
#     MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS True \
#     SOLVER.CHECKPOINT_PERIOD 5000 \
#     DATALOADER.NUM_WORKERS 4 \
#     MODEL.ROI_DENSEPOSE_HEAD.LOSS_NAME "DensePoseChartGlobalIUVSeparatedSCropResizeLoss" \
#     MODEL.CONDINST.IUVHead.NUM_CONVS 7 \
#     MODEL.CONDINST.MASK_BRANCH.AGG_CHANNELS 256\
#     MODEL.CONDINST.IUVHead.CHANNELS 256 \
#     MODEL.CONDINST.IUVHead.ABS_COORDS True \
#     MODEL.CONDINST.IUVHead.REL_COORDS True \
#     MODEL.CONDINST.IUVHead.POSE_EMBEDDING_NUM_FREQS 10 \
#     MODEL.CONDINST.IUVHead.NUM_LAMBDA_LAYER 1 \
#     MODEL.CONDINST.IUVHead.GT_INSTANCES True \
#     MODEL.CONDINST.IUVHead.NORM_COORD_BOXHW True \
#     MODEL.CONDINST.IUVHead.OUT_CHANNELS 78 \
#     MODEL.ROI_DENSEPOSE_HEAD.NAME "DensePoseMultilayermaskHead" \
#     MODEL.CONDINST.IUVHead.NORM 'GN' \
#     # MODEL.CONDINST.IUVHead.PARTIAL_CONV True \
#     # MODEL.CONDINST.IUVHead.PARTIAL_NORM True \
#     # MODEL.CONDINST.IUVHead.MASK_OUT_BG_FEATURES 'hard' \


# cfg_name='densepose_CondInst_R_50_s1x'
# CUDA_LAUNCH_BLOCKING=1 python train_net.py --config-file configs/${cfg_name}.yaml \
# 	--resume \
#     SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 \
#     OUTPUT_DIR ./output/${cfg_name}_1chSeg_cropResizeNew_deeperIUVHead_RelCoordNormFea \
#     MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS 1 \
#     MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS True \
#     SOLVER.CHECKPOINT_PERIOD 5000 \
#     DATALOADER.NUM_WORKERS 2 \
# 	MODEL.ROI_DENSEPOSE_HEAD.LOSS_NAME "DensePoseChartGlobalIUVSeparatedSCropResizeLoss" \
# 	MODEL.CONDINST.IUVHead.NUM_CONVS 14 \
#     MODEL.CONDINST.IUVHead.REL_COORDS True \
#     MODEL.CONDINST.IUVHead.NORM_FEATURES True \
#     # SOLVER.CLIP_GRADIENTS.ENABLED True \
# 	# MODEL.ROI_DENSEPOSE_HEAD.NAME "DensePoseDeepLabHead" \
# 	# MODEL.ROI_DENSEPOSE_HEAD.POINT_REGRESSION_WEIGHTS 1.0 \
# 	# SOLVER.WARMUP_ITERS 2000
#     # SOLVER.CLIP_GRADIENTS.ENABLED True SOLVER.CLIP_GRADIENTS.CLIP_TYPE norm SOLVER.CLIP_GRADIENTS.CLIP_VALUE 100.0\
#  # ./run_train_dp_1.sh

# cfg_name='densepose_CondInst_R_101_s1x'
# CUDA_LAUNCH_BLOCKING=1 python train_net.py --config-file configs/${cfg_name}.yaml \
# 	--resume \
#     SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 \
#     OUTPUT_DIR ./output/${cfg_name}_1chSeg_noIUVLoss \
#     MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS 1 \
#     MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS True \
#     MODEL.ROI_DENSEPOSE_HEAD.POINT_REGRESSION_WEIGHTS 0. \
#     MODEL.ROI_DENSEPOSE_HEAD.PART_WEIGHTS 0. \
#     # MODEL.ROI_DENSEPOSE_HEAD.INDEX_WEIGHTS 1.0 \
#     # MODEL.ROI_DENSEPOSE_HEAD.INDEX_WEIGHTS 0.
#     # MODEL.WEIGHTS ./output/${cfg_name}/model_0129999.pth \
#     # DATALOADER.NUM_WORKERS 1 \