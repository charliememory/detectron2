source ~/.bashrc_liqianma

# python train_net.py --config-file configs/densepose_rcnn_R_101_FPN_DL_WC1_s1x_ft.yaml \
#     --eval-only MODEL.WEIGHTS output/model_0129999.pth


# cfg_name='densepose_rcnn_R_101_FPN_DL_s1x'
# python train_net.py --config-file configs/${cfg_name}.yaml \
#     --eval-only MODEL.WEIGHTS ./output/${cfg_name}/model_final.pth \
# 	MODEL.FCOS.INFERENCE_TH_TEST 0.3 \

# # cfg_name='densepose_rcnn_R_101_FPN_DL_s1x'
# cfg_name='densepose_rcnn_R_101_FPN_DL_s1x_InsSeg'
# mode_name=${cfg_name}_1GPU
# python train_net.py --config-file configs/${cfg_name}.yaml \
#     --eval-only MODEL.WEIGHTS ./output/${mode_name}/model_final.pth \
#     > ./output/${mode_name}/eval_log.txt


# cfg_name='densepose_CondInst_R_50_s1x'
# mode_name=${cfg_name}_1chSeg_cropResizeNew_deepIUVHead_lambda1
# python train_net.py --config-file configs/${cfg_name}.yaml \
#     --eval-only MODEL.WEIGHTS ./output/${mode_name}/model_0117999.pth \
#     OUTPUT_DIR ./output/${mode_name} \
#     MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS 1 \
# 	MODEL.CONDINST.IUVHead.NUM_CONVS 7 \
#     MODEL.CONDINST.IUVHead.NUM_LAMBDA_LAYER 1 \
#     # > ./output/${mode_name}/eval_log.txt
#     # MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS True \


# cfg_name='densepose_CondInst_R_50_s1x'
# mode_name=${cfg_name}_1chSeg_cropResizeNew_deepIUVHead_RelCoord
# python train_net.py --config-file configs/${cfg_name}.yaml \
#     --eval-only MODEL.WEIGHTS ./output/${mode_name}/model_0114999.pth \
#     OUTPUT_DIR ./output/${mode_name} \
#     MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS 1 \
# 	MODEL.CONDINST.IUVHead.NUM_CONVS 7 \
# 	MODEL.CONDINST.IUVHead.REL_COORDS True \
#     > ./output/${mode_name}/eval_log.txt
#     # MODEL.CONDINST.IUVHead.NUM_LAMBDA_LAYER 1 \
#     # MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS True \

# cfg_name='densepose_CondInst_R_50_s1x'
# mode_name=${cfg_name}_1chSeg_cropResizeNew_deeperIUVHead_RelCoord
# python train_net.py --config-file configs/${cfg_name}.yaml \
#     --eval-only MODEL.WEIGHTS ./output/${mode_name}/model_0059999.pth \
#     OUTPUT_DIR ./output/${mode_name} \
#     MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS 1 \
# 	MODEL.CONDINST.IUVHead.NUM_CONVS 14 \
# 	MODEL.CONDINST.IUVHead.REL_COORDS True \
#     > ./output/${mode_name}/eval_log.txt
#     # MODEL.CONDINST.IUVHead.NUM_LAMBDA_LAYER 1 \
#     # MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS True \


# cfg_name='densepose_CondInst_R_50_s1x'
# mode_name=${cfg_name}_1chSeg_cropResizeNew_deeperIUVHead_RelCoordNormFea
# python train_net.py --config-file configs/${cfg_name}.yaml \
#     --eval-only MODEL.WEIGHTS ./output/${mode_name}/model_0059999.pth \
#     OUTPUT_DIR ./output/${mode_name} \
#     MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS 1 \
# 	MODEL.CONDINST.IUVHead.NUM_CONVS 14 \
# 	MODEL.CONDINST.IUVHead.REL_COORDS True \
#     MODEL.CONDINST.IUVHead.NORM_FEATURES True \
#     > ./output/${mode_name}/eval_log.txt
#     # MODEL.CONDINST.IUVHead.NUM_LAMBDA_LAYER 1 \
#     # MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS True \

# cfg_name='densepose_CondInst_R_50_s1x'
# mode_name=${cfg_name}_1chSeg_cropResizeNew_deeperIUVHead_RelCoordPE10
# python train_net.py --config-file configs/${cfg_name}.yaml \
#     --eval-only MODEL.WEIGHTS ./output/${mode_name}/model_0059999.pth \
#     OUTPUT_DIR ./output/${mode_name} \
#     MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS 1 \
# 	MODEL.CONDINST.IUVHead.NUM_CONVS 14 \
# 	MODEL.CONDINST.IUVHead.REL_COORDS True \
#     MODEL.CONDINST.IUVHead.POSE_EMBEDDING_NUM_FREQS 10 \
#     # > ./output/${mode_name}/eval_log.txt
#     # MODEL.CONDINST.IUVHead.NUM_LAMBDA_LAYER 1 \
#     # MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS True \
    
    

# cfg_name='densepose_CondInst_R_50_BiFPN_s1x'
# mode_name=${cfg_name}_1chSeg_cropResizeNew_deepIUVHead_lambda1_AbsRelCoordPE10
# python train_net.py --config-file configs/${cfg_name}.yaml \
#     --eval-only MODEL.WEIGHTS ./output/${mode_name}/model_0059999.pth \
#     OUTPUT_DIR ./output/${mode_name} \
#     MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS 1 \
#     MODEL.CONDINST.IUVHead.NUM_CONVS 7 \
#     MODEL.CONDINST.IUVHead.ABS_COORDS True \
#     MODEL.CONDINST.IUVHead.REL_COORDS True \
#     MODEL.CONDINST.IUVHead.POSE_EMBEDDING_NUM_FREQS 10 \
#     MODEL.CONDINST.IUVHead.NUM_LAMBDA_LAYER 1 \
#     > ./output/${mode_name}/eval_log.txt


# cfg_name='densepose_CondInst_R_50_BiFPN_s1x'
# mode_name=${cfg_name}_1chSeg_cropResizeNew_deepIUVHead_lambda1_RelCoordPE10
# python train_net.py --config-file configs/${cfg_name}.yaml \
#     --eval-only MODEL.WEIGHTS ./output/${mode_name}/model_0059999.pth \
#     OUTPUT_DIR ./output/${mode_name} \
#     MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS 1 \
#     MODEL.CONDINST.IUVHead.NUM_CONVS 7 \
#     MODEL.CONDINST.IUVHead.REL_COORDS True \
#     MODEL.CONDINST.IUVHead.POSE_EMBEDDING_NUM_FREQS 10 \
#     MODEL.CONDINST.IUVHead.NUM_LAMBDA_LAYER 1 \
#     # > ./output/${mode_name}/eval_log.txt
#     # MODEL.CONDINST.IUVHead.ABS_COORDS True \


cfg_name='densepose_CondInst_R_50_BiFPN_s1x'
mode_name=${cfg_name}_1chSeg_cropResizeNew_deepIUVHead_lambda1_AbsRelCoordPE10
python train_net.py --config-file configs/${cfg_name}.yaml \
    --eval-only MODEL.WEIGHTS ./output/${mode_name}/model_0089999.pth \
    OUTPUT_DIR ./output/${mode_name} \
    MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS 1 \
    MODEL.CONDINST.IUVHead.NUM_CONVS 7 \
    MODEL.CONDINST.IUVHead.ABS_COORDS True \
    MODEL.CONDINST.IUVHead.REL_COORDS True \
    MODEL.CONDINST.IUVHead.POSE_EMBEDDING_NUM_FREQS 10 \
    MODEL.CONDINST.IUVHead.NUM_LAMBDA_LAYER 1 \
    > ./output/${mode_name}/eval_log.txt
    # MODEL.CONDINST.IUVHead.MASK_OUT_BG_FEATURES "hard" \

cfg_name='densepose_CondInst_R_50_BiFPN_s1x'
mode_name=${cfg_name}_1chSeg_cropResizeNew_deepIUVHead_lambda1_RelCoordPE10
python train_net.py --config-file configs/${cfg_name}.yaml \
    --eval-only MODEL.WEIGHTS ./output/${mode_name}/model_0089999.pth \
    OUTPUT_DIR ./output/${mode_name} \
    MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS 1 \
    MODEL.CONDINST.IUVHead.NUM_CONVS 7 \
    MODEL.CONDINST.IUVHead.REL_COORDS True \
    MODEL.CONDINST.IUVHead.POSE_EMBEDDING_NUM_FREQS 10 \
    MODEL.CONDINST.IUVHead.NUM_LAMBDA_LAYER 1 \
    > ./output/${mode_name}/eval_log.txt
    # MODEL.CONDINST.IUVHead.ABS_COORDS True \
    # MODEL.CONDINST.IUVHead.MASK_OUT_BG_FEATURES "hard" \

cfg_name='densepose_CondInst_R_50_BiFPN_s1x'
mode_name=${cfg_name}_1chSeg_cropResizeNew_deepIUVHead_lambda1_AbsRelCoordPE
python train_net.py --config-file configs/${cfg_name}.yaml \
    --eval-only MODEL.WEIGHTS ./output/${mode_name}/model_0089999.pth \
    OUTPUT_DIR ./output/${mode_name} \
    MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS 1 \
    MODEL.CONDINST.IUVHead.NUM_CONVS 7 \
    MODEL.CONDINST.IUVHead.ABS_COORDS True \
    MODEL.CONDINST.IUVHead.REL_COORDS True \
    MODEL.CONDINST.IUVHead.NUM_LAMBDA_LAYER 1 \
    > ./output/${mode_name}/eval_log.txt
    # MODEL.CONDINST.IUVHead.POSE_EMBEDDING_NUM_FREQS 10 \
    # MODEL.CONDINST.IUVHead.MASK_OUT_BG_FEATURES "hard" \

cfg_name='densepose_CondInst_R_50_BiFPN_s1x'
mode_name=${cfg_name}_1chSeg_cropResizeNew_deepWideIUVHead_lambda1_AbsRelCoordPE10
python train_net.py --config-file configs/${cfg_name}.yaml \
    --eval-only MODEL.WEIGHTS ./output/${mode_name}/model_0089999.pth \
    OUTPUT_DIR ./output/${mode_name} \
    MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS 1 \
    MODEL.CONDINST.IUVHead.NUM_CONVS 7 \
    MODEL.CONDINST.MASK_BRANCH.AGG_CHANNELS 256\
    MODEL.CONDINST.IUVHead.CHANNELS 256 MODEL.CONDINST.MASK_BRANCH.CHANNELS 256\
    MODEL.CONDINST.IUVHead.ABS_COORDS True \
    MODEL.CONDINST.IUVHead.REL_COORDS True \
    MODEL.CONDINST.IUVHead.POSE_EMBEDDING_NUM_FREQS 10 \
    MODEL.CONDINST.IUVHead.NUM_LAMBDA_LAYER 1 \
    > ./output/${mode_name}/eval_log.txt
    # MODEL.CONDINST.IUVHead.MASK_OUT_BG_FEATURES "hard" \

# cfg_name='densepose_CondInst_R_50_BiFPN_s1x'
# mode_name=${cfg_name}_1chSeg_cropResizeNew_maskSoftOutBg_deepIUVHead_lambda1_AbsRelCoordPE10
# python train_net.py --config-file configs/${cfg_name}.yaml \
#     --eval-only MODEL.WEIGHTS ./output/${mode_name}/model_0059999.pth \
#     OUTPUT_DIR ./output/${mode_name} \
#     MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS 1 \
#     MODEL.CONDINST.IUVHead.NUM_CONVS 7 \
#     MODEL.CONDINST.IUVHead.ABS_COORDS True \
#     MODEL.CONDINST.IUVHead.REL_COORDS True \
#     MODEL.CONDINST.IUVHead.POSE_EMBEDDING_NUM_FREQS 10 \
#     MODEL.CONDINST.IUVHead.NUM_LAMBDA_LAYER 1 \
#     MODEL.CONDINST.IUVHead.MASK_OUT_BG_FEATURES "soft" \
#     > ./output/${mode_name}/eval_log.txt


