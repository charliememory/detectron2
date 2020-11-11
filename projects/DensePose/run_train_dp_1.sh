source ~/.bashrc_liqianma


# cfg_name='densepose_CondInst_R_50_res2_s1x'
# CUDA_LAUNCH_BLOCKING=1 python train_net.py --config-file configs/${cfg_name}.yaml \
# 	--resume \
#     SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 \
#     OUTPUT_DIR ./output/${cfg_name}_1chSeg_cropResizeNew_deepIUVHead_RelCoord \
#     MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS 1 \
#     MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS True \
#     SOLVER.CHECKPOINT_PERIOD 5000 \
#     DATALOADER.NUM_WORKERS 2 \
# 	MODEL.ROI_DENSEPOSE_HEAD.LOSS_NAME "DensePoseChartGlobalIUVSeparatedSCropResizeLoss" \
# 	MODEL.CONDINST.IUVHead.NUM_CONVS 7 \
# 	MODEL.CONDINST.IUVHead.REL_COORDS True 
#     # SOLVER.CLIP_GRADIENTS.ENABLED True \
# 	# MODEL.ROI_DENSEPOSE_HEAD.NAME "DensePoseDeepLabHead" \
# 	# MODEL.ROI_DENSEPOSE_HEAD.POINT_REGRESSION_WEIGHTS 1.0 \
# 	# SOLVER.WARMUP_ITERS 2000
#     # SOLVER.CLIP_GRADIENTS.ENABLED True SOLVER.CLIP_GRADIENTS.CLIP_TYPE norm SOLVER.CLIP_GRADIENTS.CLIP_VALUE 100.0\

# cfg_name='densepose_rcnn_R_50_FPN_s1x'
# python train_net.py --config-file configs/${cfg_name}.yaml \
#     SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025\
#     OUTPUT_DIR ./output/${cfg_name}_CondIns_ROIGlobalHeads_noSloss \
#   	MODEL.ROI_HEADS.NAME "DensePoseROIGlobalHeads"\
#     MODEL.ROI_DENSEPOSE_HEAD.INDEX_WEIGHTS 0. \
#     DATALOADER.NUM_WORKERS 2 \

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
    OUTPUT_DIR ./output/${cfg_name}_1chSeg_IUVSparsePooler2Head_DeepLabGNSparse512_GTinsDilated3  \
    MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS 1 \
    MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS True \
    SOLVER.CHECKPOINT_PERIOD 2000 \
    DATALOADER.NUM_WORKERS 2 \
    MODEL.CONDINST.IUVHead.NAME "IUVSparsePooler2Head" \
    MODEL.ROI_DENSEPOSE_HEAD.LOSS_NAME "DensePoseChartGlobalIUVSeparatedSPoolerLoss" \
    MODEL.ROI_DENSEPOSE_HEAD.NAME "DensePoseDeepLabSparseHead" \
    MODEL.ROI_DENSEPOSE_HEAD.CONV_HEAD_DIM 512 \
    MODEL.CONDINST.IUVHead.GT_INSTANCES True \
    MODEL.CONDINST.IUVHead.MASK_OUT_BG_FEATURES "hard" \
    MODEL.CONDINST.IUVHead.DILATE_FGMASK_KENERAL_SIZE 3 \
    MODEL.ROI_DENSEPOSE_HEAD.DEEPLAB.NORM "GN" \
    # MODEL.CONDINST.IUVHead.ABS_COORDS True \
    # MODEL.CONDINST.IUVHead.REL_COORDS True \
    # MODEL.CONDINST.IUVHead.POSE_EMBEDDING_NUM_FREQS 10 \

# cfg_name='densepose_rcnn_R_50_FPN_s1x'
# python train_net.py --config-file configs/${cfg_name}.yaml \
#     SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025\
#     OUTPUT_DIR ./output/${cfg_name}_noSloss \
#     MODEL.ROI_DENSEPOSE_HEAD.INDEX_WEIGHTS 0. \

# cfg_name='densepose_rcnn_R_50_FPN_s1x'
# python train_net.py --config-file configs/${cfg_name}.yaml \
#     SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025\
#     OUTPUT_DIR ./output/${cfg_name}_ROIGlobalHeads \
#     MODEL.ROI_DENSEPOSE_HEAD.CONV_HEAD_DIM 256 \
#     MODEL.ROI_HEADS.NAME "DensePoseROIHeads" \

# cfg_name='densepose_CondInst_R_50_BiFPN_s1x'
# CUDA_LAUNCH_BLOCKING=1 python train_net.py --config-file configs/${cfg_name}.yaml \
#     --resume \
#     SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 \
#     OUTPUT_DIR ./output/${cfg_name}_deepWide2IUVHeadBN_lambda1_AbsRelCoordPE10_GTins_normCoordBoxHW_parNormDilate3_4meanUV \
#     MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS 1 \
#     MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS True \
#     SOLVER.CHECKPOINT_PERIOD 1000 \
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
#     MODEL.CONDINST.IUVHead.NORM 'BN' \
#     MODEL.CONDINST.IUVHead.PARTIAL_NORM True \
# 	MODEL.ROI_DENSEPOSE_HEAD.POINT_REGRESSION_WEIGHTS 4.0 \
#     # MODEL.CONDINST.IUVHead.PARTIAL_CONV True \
#     # MODEL.CONDINST.IUVHead.MASK_OUT_BG_FEATURES 'hard' \

# cfg_name='densepose_CondInst_R_101_s1x'
# CUDA_LAUNCH_BLOCKING=1 python train_net.py --config-file configs/${cfg_name}.yaml \
#     --resume \
#     SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 \
#     OUTPUT_DIR ./output/${cfg_name}_1chSeg_cropResizeNew_Decoder_deepWide2IUVHead_lambda1_AbsRelCoordPE10 \
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
#     MODEL.ROI_DENSEPOSE_HEAD.DECODER_ON True\
#     MODEL.ROI_DENSEPOSE_HEAD.DECODER_COMMON_STRIDE 8 \

# cfg_name='densepose_CondInst_R_50_s1x'
# CUDA_LAUNCH_BLOCKING=1 python train_net.py --config-file configs/${cfg_name}.yaml \
# 	--resume \
#     SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 \
#     OUTPUT_DIR ./output/${cfg_name}_1chSeg_cropResizeNew_deeperIUVHead_RelCoord \
#     MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS 1 \
#     MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS True \
#     SOLVER.CHECKPOINT_PERIOD 5000 \
#     DATALOADER.NUM_WORKERS 2 \
# 	MODEL.ROI_DENSEPOSE_HEAD.LOSS_NAME "DensePoseChartGlobalIUVSeparatedSCropResizeLoss" \
# 	MODEL.CONDINST.IUVHead.NUM_CONVS 14 \
# 	MODEL.CONDINST.IUVHead.REL_COORDS True \
#     # SOLVER.CLIP_GRADIENTS.ENABLED True \
# 	# MODEL.ROI_DENSEPOSE_HEAD.NAME "DensePoseDeepLabHead" \
# 	# MODEL.ROI_DENSEPOSE_HEAD.POINT_REGRESSION_WEIGHTS 1.0 \
# 	# SOLVER.WARMUP_ITERS 2000
#     # SOLVER.CLIP_GRADIENTS.ENABLED True SOLVER.CLIP_GRADIENTS.CLIP_TYPE norm SOLVER.CLIP_GRADIENTS.CLIP_VALUE 100.0\
    
# # ./run_train_dp_7.sh