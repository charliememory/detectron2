source ~/.bashrc_liqianma

# 47999
cfg_name='densepose_CondInst_R_50_s1x'
mode_name=${cfg_name}_1chSeg_IUVSparsePooler2Head_DeepLab2BNSparse256_GTinsDilated3
python train_net.py --config-file configs/${cfg_name}.yaml \
    --eval-only MODEL.WEIGHTS ./output/${mode_name}/model_0049999.pth \
    OUTPUT_DIR ./output/${mode_name} \
    MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS 1 \
    MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS True \
    SOLVER.CHECKPOINT_PERIOD 2000 \
    DATALOADER.NUM_WORKERS 2 \
    MODEL.CONDINST.IUVHead.NAME "IUVSparsePooler2Head" \
    MODEL.ROI_DENSEPOSE_HEAD.LOSS_NAME "DensePoseChartGlobalIUVSeparatedSPoolerLoss" \
    MODEL.ROI_DENSEPOSE_HEAD.NAME "DensePoseDeepLab2SparseHead" \
    MODEL.ROI_DENSEPOSE_HEAD.CONV_HEAD_DIM 256 \
    MODEL.CONDINST.IUVHead.MASK_OUT_BG_FEATURES "hard" \
    MODEL.CONDINST.IUVHead.DILATE_FGMASK_KENERAL_SIZE 3 \
    MODEL.ROI_DENSEPOSE_HEAD.DEEPLAB.NORM "BN" \
    > ./output/${mode_name}/eval_log.txt
    # MODEL.CONDINST.IUVHead.GT_INSTANCES True \


    
./run_train_dp_3.sh