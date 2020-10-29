source ~/.bashrc_liqianma

# cfg_name='densepose_rcnn_R_50_FPN_DL_s1x'
# CUDA_LAUNCH_BLOCKING=1 python train_net.py --config-file configs/${cfg_name}.yaml \
# 	--resume \
#     SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 \
#     OUTPUT_DIR ./output/${cfg_name}_InsSeg_1GPU \
#     SOLVER.CHECKPOINT_PERIOD 5000 \
#     DATALOADER.NUM_WORKERS 2 \
#     MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS True \
#     


cfg_name='densepose_CondInst_R_101_s1x'
CUDA_LAUNCH_BLOCKING=1 python train_net.py --config-file configs/${cfg_name}.yaml \
    --resume \
    SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 \
    OUTPUT_DIR ./output/${cfg_name}_1chSeg_cropResizeNew_maskOutBgMultiLayerAndOut_deepWide2IUVHead_lambda1_AbsRelCoordPE10 \
    MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS 1 \
    MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS True \
    SOLVER.CHECKPOINT_PERIOD 5000 \
    DATALOADER.NUM_WORKERS 2 \
    MODEL.ROI_DENSEPOSE_HEAD.LOSS_NAME "DensePoseChartGlobalIUVSeparatedSCropResizeLoss" \
    MODEL.CONDINST.IUVHead.NUM_CONVS 7 \
    MODEL.CONDINST.MASK_BRANCH.AGG_CHANNELS 256\
    MODEL.CONDINST.IUVHead.CHANNELS 256 \
    MODEL.CONDINST.IUVHead.ABS_COORDS True \
    MODEL.CONDINST.IUVHead.REL_COORDS True \
    MODEL.CONDINST.IUVHead.POSE_EMBEDDING_NUM_FREQS 10 \
    MODEL.CONDINST.IUVHead.NUM_LAMBDA_LAYER 1 \
    MODEL.ROI_DENSEPOSE_HEAD.NAME "DensePoseMultilayermaskHead" \