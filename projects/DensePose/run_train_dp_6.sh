source ~/.bashrc_liqianma


# cfg_name='densepose_CondInst_R_101_s1x'
# CUDA_LAUNCH_BLOCKING=1 python train_net.py --config-file configs/${cfg_name}.yaml \
#     --resume \
#     SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 \
#     OUTPUT_DIR ./output/${cfg_name}_1chSeg_cropResizeNew_deepWide2IUVHead_lambda1_AbsRelCoordPE10_aux0.1Fg \
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
#     MODEL.ROI_DENSEPOSE_HEAD.AUX_FG_SEGM_WEIGHTS 0.1 \
#     MODEL.CONDINST.IUVHead.OUT_CHANNELS 78 \
#     # MODEL.ROI_DENSEPOSE_HEAD.AUX_REL_COORDS_WEIGHTS 0.1 \
#     # MODEL.ROI_DENSEPOSE_HEAD.NAME "DensePoseMultilayermaskHead" \ 

cfg_name='densepose_CondInst_R_50_s1x'
CUDA_LAUNCH_BLOCKING=1 python train_net.py --config-file configs/${cfg_name}.yaml \
    --resume \
    SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 \
    OUTPUT_DIR ./output/${cfg_name}_1chSeg_IUVPooler2Head_V1ConvXGN256 \
    MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS 1 \
    MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS True \
    SOLVER.CHECKPOINT_PERIOD 2000 \
    DATALOADER.NUM_WORKERS 2 \
    MODEL.CONDINST.IUVHead.NAME "IUVPooler2Head" \
    MODEL.ROI_DENSEPOSE_HEAD.LOSS_NAME "DensePoseChartGlobalIUVSeparatedSPoolerLoss" \
    MODEL.ROI_DENSEPOSE_HEAD.NAME "DensePoseV1ConvXGNHead" \
    MODEL.ROI_DENSEPOSE_HEAD.CONV_HEAD_DIM 256 \
    MODEL.CONDINST.IUVHead.GT_INSTANCES True \
    # MODEL.CONDINST.IUVHead.NUM_CONVS 7 \
    # MODEL.CONDINST.MASK_BRANCH.AGG_CHANNELS 256\
    # MODEL.CONDINST.IUVHead.CHANNELS 256 \
    # MODEL.CONDINST.IUVHead.REL_COORDS True \
    # MODEL.CONDINST.IUVHead.POSE_EMBEDDING_NUM_FREQS 10 \
    # MODEL.CONDINST.IUVHead.NUM_LAMBDA_LAYER 1 \
    # MODEL.CONDINST.IUVHead.NORM_COORD_BOXHW True \
    # MODEL.ROI_DENSEPOSE_HEAD.NAME "DensePoseDeepLabHead" \
    # MODEL.ROI_DENSEPOSE_HEAD.INDEX_WEIGHTS 0. \
    # MODEL.ROI_DENSEPOSE_HEAD.POINT_REGRESSION_WEIGHTS 1.0 \
    # MODEL.CONDINST.IUVHead.ABS_COORDS True \