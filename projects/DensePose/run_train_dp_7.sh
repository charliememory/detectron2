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





cd ~/workspace/Gitlab/spconv/
rm -rf build
python setup.py bdist_wheel
cd ./dist
pip uninstall spconv -y
pip install spconv-1.2.1-cp38-cp38-linux_x86_64.whl
cd ~/workspace/Gitlab/detectron2/projects/DensePose

cfg_name='densepose_CondInst_R_50_s1x'
CUDA_LAUNCH_BLOCKING=1 python train_net.py --config-file configs/${cfg_name}.yaml \
    --resume \
    SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 \
    OUTPUT_DIR ./output/${cfg_name}_1chSeg_IUVSparsePooler2Head_AggFeaASPP_V1ConvXBNSparse256_GTinsDilated3  \
    MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS 1 \
    MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS True \
    SOLVER.CHECKPOINT_PERIOD 2000 \
    DATALOADER.NUM_WORKERS 2 \
    MODEL.CONDINST.IUVHead.NAME "IUVSparsePooler2Head" \
    MODEL.ROI_DENSEPOSE_HEAD.LOSS_NAME "DensePoseChartGlobalIUVSeparatedSPoolerLoss" \
    MODEL.ROI_DENSEPOSE_HEAD.NAME "DensePoseV1ConvXGNSparseHead" \
    MODEL.ROI_DENSEPOSE_HEAD.CONV_HEAD_DIM 256 \
    MODEL.CONDINST.MASK_BRANCH.AGG_CHANNELS 256 \
    MODEL.CONDINST.IUVHead.GT_INSTANCES True \
    MODEL.CONDINST.IUVHead.MASK_OUT_BG_FEATURES "hard" \
    MODEL.CONDINST.IUVHead.DILATE_FGMASK_KENERAL_SIZE 3 \
    MODEL.ROI_DENSEPOSE_HEAD.DEEPLAB.NORM "BN" \
    MODEL.CONDINST.IUVHead.USE_AGG_FEATURES True \
    MODEL.CONDINST.MASK_BRANCH.USE_ASPP True
    # MODEL.CONDINST.IUVHead.NUM_LAMBDA_LAYER 1 \
    # MODEL.CONDINST.IUVHead.ABS_COORDS True \
    # MODEL.CONDINST.IUVHead.REL_COORDS True \
    # MODEL.CONDINST.IUVHead.POSE_EMBEDDING_NUM_FREQS 10 \


