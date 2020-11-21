source ~/.bashrc_liqianma

# 49999
# cfg_name='densepose_CondInst_R_50_s1x'
# mode_name=${cfg_name}_1chSeg_IUVSparsePooler2Head_DeepLab2BNSparse256_GTinsDilated3
# python train_net.py --config-file configs/${cfg_name}.yaml \
#     --eval-only MODEL.WEIGHTS ./output/${mode_name}/model_0049999.pth \
#     OUTPUT_DIR ./output/${mode_name} \
#     MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS 1 \
#     MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS True \
#     SOLVER.CHECKPOINT_PERIOD 2000 \
#     DATALOADER.NUM_WORKERS 2 \
#     MODEL.CONDINST.IUVHead.NAME "IUVSparsePooler2Head" \
#     MODEL.ROI_DENSEPOSE_HEAD.LOSS_NAME "DensePoseChartGlobalIUVSeparatedSPoolerLoss" \
#     MODEL.ROI_DENSEPOSE_HEAD.NAME "DensePoseDeepLab2SparseHead" \
#     MODEL.ROI_DENSEPOSE_HEAD.CONV_HEAD_DIM 256 \
#     MODEL.CONDINST.IUVHead.MASK_OUT_BG_FEATURES "hard" \
#     MODEL.CONDINST.IUVHead.DILATE_FGMASK_KENERAL_SIZE 3 \
#     MODEL.ROI_DENSEPOSE_HEAD.DEEPLAB.NORM "BN" \
#     > ./output/${mode_name}/eval_log.txt
#     # MODEL.CONDINST.IUVHead.GT_INSTANCES True \



# # 49999
# cfg_name='densepose_CondInst_R_50_s1x'
# mode_name=${cfg_name}_1chSeg_IUVSparsePooler2Head_AggFea_V1ConvXBNSparseInsINLowMem_GTinsDilated3_RelCoordPE5_normCoordBoxHW_v3sumcoord
# python train_net.py --config-file configs/${cfg_name}.yaml \
#     --eval-only MODEL.WEIGHTS ./output/${mode_name}/model_0049999.pth \
#     OUTPUT_DIR ./output/${mode_name} \
#     MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS 1 \
#     MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS True \
#     SOLVER.CHECKPOINT_PERIOD 2000 \
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
#     MODEL.CONDINST.IUVHead.REL_COORDS True \
#     MODEL.CONDINST.IUVHead.POSE_EMBEDDING_NUM_FREQS 5 \
#     MODEL.FCOS.INFERENCE_TH_TEST 0.2 \
#     # > ./output/${mode_name}/eval_log.txt
#     # MODEL.CONDINST.IUVHead.REMOVE_MASK_OVERLAP True \

    
# # 49999
# cfg_name='densepose_CondInst_R_50_lowres_s1x'
# mode_name=${cfg_name}_1chSeg_IUVSparsePooler2Head_AggFea_V1ConvXGNSparseInsINLowMemNoOverlapTrue128_GTinsDilated3_fewProposal_fewLayer_noClsLoss
# python train_net.py --config-file configs/${cfg_name}.yaml \
#     --eval-only MODEL.WEIGHTS ./output/${mode_name}/model_0049999.pth \
#     OUTPUT_DIR ./output/${mode_name} \
#     MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS 1 \
#     MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS True \
#     SOLVER.CHECKPOINT_PERIOD 5000 \
#     DATALOADER.NUM_WORKERS 2 \
#     MODEL.CONDINST.IUVHead.NAME "IUVSparsePooler2Head" \
#     MODEL.ROI_DENSEPOSE_HEAD.LOSS_NAME "DensePoseChartGlobalIUVSeparatedSPoolerLoss" \
#     MODEL.ROI_DENSEPOSE_HEAD.NAME "DensePoseV1ConvXGNSparseGNHead" \
#     MODEL.ROI_DENSEPOSE_HEAD.CONV_HEAD_DIM 128 \
#     MODEL.CONDINST.MASK_BRANCH.AGG_CHANNELS 128 \
#     MODEL.CONDINST.IUVHead.MASK_OUT_BG_FEATURES "hard" \
#     MODEL.CONDINST.IUVHead.DILATE_FGMASK_KENERAL_SIZE 3 \
#     MODEL.ROI_DENSEPOSE_HEAD.DEEPLAB.NORM "InsIN" \
#     MODEL.CONDINST.IUVHead.USE_AGG_FEATURES True \
#     MODEL.CONDINST.IUVHead.INSTANCE_AWARE_GN True \
#     MODEL.CONDINST.IUVHead.REMOVE_MASK_OVERLAP True \
#     MODEL.CONDINST.MASK_BRANCH.NUM_CONVS 3\
#     MODEL.CONDINST.MAX_PROPOSALS 160 \
#     MODEL.FCOS.PRE_NMS_TOPK_TRAIN 160 \
#     MODEL.FCOS.PRE_NMS_TOPK_TEST 160 \
#     MODEL.FCOS.POST_NMS_TOPK_TRAIN 32 \
#     MODEL.FCOS.POST_NMS_TOPK_TEST 32 \
#     MODEL.FCOS.DISABLE_CLS_LOSS True \
#     MODEL.FCOS.INFERENCE_TH_TEST 0.2 \
#     MODEL.CONDINST.MASK_OUT_STRIDE 8\
#     # > ./output/${mode_name}/eval_log.txt


# cd ~/workspace/Gitlab/spconv/
# rm -rf build
# python setup.py bdist_wheel
# cd ./dist
# pip uninstall spconv -y
# pip install spconv-1.2.1-cp38-cp38-linux_x86_64.whl
# cd ~/workspace/Gitlab/detectron2/projects/DensePose


# # 49999
# cfg_name='densepose_CondInst_R_50_s1x'
# mode_name=${cfg_name}_1chSeg_IUVSparsePooler2Head_AggFea_V1ConvXGNSparseInsINLowMemNoOverlapTrue_GTinsDilated3
# # CUDA_VISIBLE_DEVICES=6 
# python train_net.py --config-file configs/${cfg_name}.yaml \
#     --eval-only MODEL.WEIGHTS ./output/${mode_name}/model_final.pth \
#     OUTPUT_DIR ./output/${mode_name} \
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
#     MODEL.CONDINST.IUVHead.INSTANCE_AWARE_GN True \
#     MODEL.CONDINST.IUVHead.REMOVE_MASK_OVERLAP True \
#     MODEL.FCOS.INFERENCE_TH_TEST 0.2 \
#     # > ./output/${mode_name}/eval_log.txt

# 49999
cfg_name='densepose_CondInst_R_50_s1x_ft'
mode_name=${cfg_name}_1chSeg_IUVSparsePooler2Head_AggFea_V1ConvXGNSparseInsINLowMemNoOverlapTrue_GTinsDilated3_ftIUVheadOnly
# CUDA_VISIBLE_DEVICES=6 
python train_net.py --config-file configs/${cfg_name}.yaml \
    --eval-only MODEL.WEIGHTS ./output/${mode_name}/model_final.pth \
    OUTPUT_DIR ./output/${mode_name} \
    MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS 1 \
    MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS True \
    SOLVER.CHECKPOINT_PERIOD 5000 \
    DATALOADER.NUM_WORKERS 2 \
    MODEL.CONDINST.IUVHead.NAME "IUVSparsePooler2Head" \
    MODEL.ROI_DENSEPOSE_HEAD.LOSS_NAME "DensePoseChartGlobalIUVSeparatedSPoolerLoss" \
    MODEL.ROI_DENSEPOSE_HEAD.NAME "DensePoseV1ConvXGNSparseGNHead" \
    MODEL.ROI_DENSEPOSE_HEAD.CONV_HEAD_DIM 256 \
    MODEL.CONDINST.MASK_BRANCH.AGG_CHANNELS 256 \
    MODEL.CONDINST.IUVHead.MASK_OUT_BG_FEATURES "hard" \
    MODEL.CONDINST.IUVHead.DILATE_FGMASK_KENERAL_SIZE 3 \
    MODEL.ROI_DENSEPOSE_HEAD.DEEPLAB.NORM "InsIN" \
    MODEL.CONDINST.IUVHead.USE_AGG_FEATURES True \
    MODEL.CONDINST.IUVHead.INSTANCE_AWARE_GN True \
    MODEL.CONDINST.IUVHead.REMOVE_MASK_OVERLAP True \
    MODEL.FCOS.INFERENCE_TH_TEST 0.2 \
    # > ./output/${mode_name}/eval_log.txt
    # MODEL.CONDINST.FINETUNE_IUVHead_ONLY True \

