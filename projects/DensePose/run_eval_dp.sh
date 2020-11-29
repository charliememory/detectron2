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


# cfg_name='densepose_CondInst_R_50_BiFPN_s1x'
# mode_name=${cfg_name}_1chSeg_cropResizeNew_deepIUVHead_lambda1_AbsRelCoordPE10
# python train_net.py --config-file configs/${cfg_name}.yaml \
#     --eval-only MODEL.WEIGHTS ./output/${mode_name}/model_0089999.pth \
#     OUTPUT_DIR ./output/${mode_name} \
#     MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS 1 \
#     MODEL.CONDINST.IUVHead.NUM_CONVS 7 \
#     MODEL.CONDINST.IUVHead.ABS_COORDS True \
#     MODEL.CONDINST.IUVHead.REL_COORDS True \
#     MODEL.CONDINST.IUVHead.POSE_EMBEDDING_NUM_FREQS 10 \
#     MODEL.CONDINST.IUVHead.NUM_LAMBDA_LAYER 1 \
#     > ./output/${mode_name}/eval_log.txt
#     # MODEL.CONDINST.IUVHead.MASK_OUT_BG_FEATURES "hard" \

# cfg_name='densepose_CondInst_R_50_BiFPN_s1x'
# mode_name=${cfg_name}_1chSeg_cropResizeNew_deepIUVHead_lambda1_RelCoordPE10
# python train_net.py --config-file configs/${cfg_name}.yaml \
#     --eval-only MODEL.WEIGHTS ./output/${mode_name}/model_0089999.pth \
#     OUTPUT_DIR ./output/${mode_name} \
#     MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS 1 \
#     MODEL.CONDINST.IUVHead.NUM_CONVS 7 \
#     MODEL.CONDINST.IUVHead.REL_COORDS True \
#     MODEL.CONDINST.IUVHead.POSE_EMBEDDING_NUM_FREQS 10 \
#     MODEL.CONDINST.IUVHead.NUM_LAMBDA_LAYER 1 \
#     > ./output/${mode_name}/eval_log.txt
#     # MODEL.CONDINST.IUVHead.ABS_COORDS True \
#     # MODEL.CONDINST.IUVHead.MASK_OUT_BG_FEATURES "hard" \

# cfg_name='densepose_CondInst_R_50_BiFPN_s1x'
# mode_name=${cfg_name}_1chSeg_cropResizeNew_deepIUVHead_lambda1_AbsRelCoord
# python train_net.py --config-file configs/${cfg_name}.yaml \
#     --eval-only MODEL.WEIGHTS ./output/${mode_name}/model_0089999.pth \
#     OUTPUT_DIR ./output/${mode_name} \
#     MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS 1 \
#     MODEL.CONDINST.IUVHead.NUM_CONVS 7 \
#     MODEL.CONDINST.IUVHead.ABS_COORDS True \
#     MODEL.CONDINST.IUVHead.REL_COORDS True \
#     MODEL.CONDINST.IUVHead.NUM_LAMBDA_LAYER 1 \
#     > ./output/${mode_name}/eval_log.txt
#     # MODEL.CONDINST.IUVHead.POSE_EMBEDDING_NUM_FREQS 10 \
#     # MODEL.CONDINST.IUVHead.MASK_OUT_BG_FEATURES "hard" \


# cfg_name='densepose_CondInst_R_50_BiFPN_s1x'
# mode_name=${cfg_name}_1chSeg_cropResizeNew_deepIUVHead_lambda1_AbsRelCoordPE10MultiLayer
# python train_net.py --config-file configs/${cfg_name}.yaml \
#     --eval-only MODEL.WEIGHTS ./output/${mode_name}/model_final.pth \
#     OUTPUT_DIR ./output/${mode_name} \
#     MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS 1 \
#     MODEL.CONDINST.IUVHead.NUM_CONVS 7 \
#     MODEL.CONDINST.IUVHead.ABS_COORDS True \
#     MODEL.CONDINST.IUVHead.REL_COORDS True \
#     MODEL.CONDINST.IUVHead.POSE_EMBEDDING_NUM_FREQS 10 \
#     MODEL.CONDINST.IUVHead.NUM_LAMBDA_LAYER 1 \
#     MODEL.ROI_DENSEPOSE_HEAD.NAME "DensePoseMultilayercoordHead" \
#     > ./output/${mode_name}/eval_log.txt


# cfg_name='densepose_CondInst_R_101_s1x'
# mode_name=${cfg_name}_1chSeg_cropResizeNew_deepWide2IUVHead_lambda1_AbsRelCoordPE10
# python train_net.py --config-file configs/${cfg_name}.yaml \
#     --eval-only MODEL.WEIGHTS ./output/${mode_name}/model_final.pth \
#     OUTPUT_DIR ./output/${mode_name} \
#     MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS 1 \
#     MODEL.CONDINST.IUVHead.NUM_CONVS 7 \
#     MODEL.CONDINST.MASK_BRANCH.AGG_CHANNELS 256\
#     MODEL.CONDINST.IUVHead.CHANNELS 256\
#     MODEL.CONDINST.IUVHead.ABS_COORDS True \
#     MODEL.CONDINST.IUVHead.REL_COORDS True \
#     MODEL.CONDINST.IUVHead.POSE_EMBEDDING_NUM_FREQS 10 \
#     MODEL.CONDINST.IUVHead.NUM_LAMBDA_LAYER 1 \
#     > ./output/${mode_name}/eval_log.txt


# cfg_name='densepose_CondInst_R_50_BiFPN_s1x'
# mode_name=${cfg_name}_1chSeg_cropResizeNew_deepWideIUVHead_lambda1_AbsRelCoordPE10
# python train_net.py --config-file configs/${cfg_name}.yaml \
#     --eval-only MODEL.WEIGHTS ./output/${mode_name}/model_0124999.pth \
#     OUTPUT_DIR ./output/${mode_name} \
#     MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS 1 \
#     MODEL.CONDINST.IUVHead.NUM_CONVS 7 \
#     MODEL.CONDINST.MASK_BRANCH.AGG_CHANNELS 256\
#     MODEL.CONDINST.IUVHead.CHANNELS 256 MODEL.CONDINST.MASK_BRANCH.CHANNELS 256\
#     MODEL.CONDINST.IUVHead.ABS_COORDS True \
#     MODEL.CONDINST.IUVHead.REL_COORDS True \
#     MODEL.CONDINST.IUVHead.POSE_EMBEDDING_NUM_FREQS 10 \
#     MODEL.CONDINST.IUVHead.NUM_LAMBDA_LAYER 1 \
#     > ./output/${mode_name}/eval_log.txt
#     # MODEL.CONDINST.IUVHead.MASK_OUT_BG_FEATURES "hard" \


# cfg_name='densepose_rcnn_R_50_FPN_s1x'
# mode_name=${cfg_name}_BS8
# python train_net.py --config-file configs/${cfg_name}.yaml \
#     --eval-only MODEL.WEIGHTS ./output/${mode_name}/model_0007999.pth \
#     OUTPUT_DIR ./output/${mode_name} \
#     > ./output/${mode_name}/eval_log.txt

# cfg_name='densepose_CondInst_R_50_s1x'
# mode_name=${cfg_name}_1chSeg_cropResizeNew_deepWide2IUVHead_lambda1_AbsRelCoordPE10_GTins_normCoordBoxHW_BS8
# python train_net.py --config-file configs/${cfg_name}.yaml \
#     --eval-only MODEL.WEIGHTS ./output/${mode_name}/model_0053999.pth \
#     OUTPUT_DIR ./output/${mode_name} \
#     MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS 1 \
#     MODEL.CONDINST.IUVHead.NUM_CONVS 7 \
#     MODEL.CONDINST.MASK_BRANCH.AGG_CHANNELS 256\
#     MODEL.CONDINST.IUVHead.CHANNELS 256 \
#     MODEL.CONDINST.IUVHead.ABS_COORDS True \
#     MODEL.CONDINST.IUVHead.REL_COORDS True \
#     MODEL.CONDINST.IUVHead.POSE_EMBEDDING_NUM_FREQS 10 \
#     MODEL.CONDINST.IUVHead.NUM_LAMBDA_LAYER 1 \
#     MODEL.CONDINST.IUVHead.NORM_COORD_BOXHW True \
#     > ./output/${mode_name}/eval_log.txt
#     # MODEL.CONDINST.IUVHead.GT_INSTANCES True \


# # 69999
# cfg_name='densepose_CondInst_R_50_s1x'
# mode_name=${cfg_name}_1chSeg_IUVSparsePooler2Head_V1ConvXBNSparse512_GTinsDilated3_AbsRelCoordPE10_normCoordBoxHW
# python train_net.py --config-file configs/${cfg_name}.yaml \
#     --eval-only MODEL.WEIGHTS ./output/${mode_name}/model_0049999.pth \
#     OUTPUT_DIR ./output/${mode_name} \
#     MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS 1 \
#     MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS True \
#     SOLVER.CHECKPOINT_PERIOD 2000 \
#     DATALOADER.NUM_WORKERS 2 \
#     MODEL.CONDINST.IUVHead.NAME "IUVSparsePooler2Head" \
#     MODEL.ROI_DENSEPOSE_HEAD.LOSS_NAME "DensePoseChartGlobalIUVSeparatedSPoolerLoss" \
#     MODEL.ROI_DENSEPOSE_HEAD.NAME "DensePoseV1ConvXGNSparseHead" \
#     MODEL.ROI_DENSEPOSE_HEAD.CONV_HEAD_DIM 512 \
#     MODEL.CONDINST.IUVHead.MASK_OUT_BG_FEATURES "hard" \
#     MODEL.CONDINST.IUVHead.DILATE_FGMASK_KENERAL_SIZE 3 \
#     MODEL.ROI_DENSEPOSE_HEAD.DEEPLAB.NORM "BN" \
#     MODEL.CONDINST.IUVHead.ABS_COORDS True \
#     MODEL.CONDINST.IUVHead.REL_COORDS True \
#     MODEL.CONDINST.IUVHead.POSE_EMBEDDING_NUM_FREQS 10 \
#     > ./output/${mode_name}/eval_log.txt
#     # MODEL.CONDINST.IUVHead.GT_INSTANCES True \

# # 63999
# cfg_name='densepose_CondInst_R_50_s1x'
# mode_name=${cfg_name}_1chSeg_IUVSparsePooler2Head_V1ConvXBNSparse256_GTinsDilated3_AbsRelCoordPE10_normCoordBoxHW
# python train_net.py --config-file configs/${cfg_name}.yaml \
#     --eval-only MODEL.WEIGHTS ./output/${mode_name}/model_0049999.pth \
#     OUTPUT_DIR ./output/${mode_name} \
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
#     > ./output/${mode_name}/eval_log.txt
#     # MODEL.CONDINST.IUVHead.GT_INSTANCES True \

# # 47999
# cfg_name='densepose_CondInst_R_50_s1x'
# mode_name=${cfg_name}_1chSeg_IUVSparsePooler2Head_DeepLabBNSparse256_GTinsDilated3
# python train_net.py --config-file configs/${cfg_name}.yaml \
#     --eval-only MODEL.WEIGHTS ./output/${mode_name}/model_0049999.pth \
#     OUTPUT_DIR ./output/${mode_name} \
#     MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS 1 \
#     MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS True \
#     SOLVER.CHECKPOINT_PERIOD 2000 \
#     DATALOADER.NUM_WORKERS 2 \
#     MODEL.CONDINST.IUVHead.NAME "IUVSparsePooler2Head" \
#     MODEL.ROI_DENSEPOSE_HEAD.LOSS_NAME "DensePoseChartGlobalIUVSeparatedSPoolerLoss" \
#     MODEL.ROI_DENSEPOSE_HEAD.NAME "DensePoseDeepLabSparseHead" \
#     MODEL.ROI_DENSEPOSE_HEAD.CONV_HEAD_DIM 256 \
#     MODEL.CONDINST.IUVHead.MASK_OUT_BG_FEATURES "hard" \
#     MODEL.CONDINST.IUVHead.DILATE_FGMASK_KENERAL_SIZE 3 \
#     MODEL.ROI_DENSEPOSE_HEAD.DEEPLAB.NORM "BN" \
#     > ./output/${mode_name}/eval_log.txt
#     # MODEL.CONDINST.IUVHead.GT_INSTANCES True \


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

# cfg_name='densepose_rcnn_R_50_FPN_s1x'
# mode_name=${cfg_name}_BS8
# python train_net.py --config-file configs/${cfg_name}.yaml \
#     --eval-only MODEL.WEIGHTS ./output/${mode_name}/model_final.pth \
#     MODEL.ROI_DENSEPOSE_HEAD.DECODER_ON False \
#     # >> ./output/${mode_name}/eval_log.txt


# ./run_train_dp_9.sh

# cfg_name='densepose_CondInst_R_50_s1x'
# mode_name=${cfg_name}_1chSeg_IUVSparsePooler2Head_AggFea_V1ConvXGNSparseInsINLowMemNoOverlapTrueResInput_11Layers_GTinsDilated3_amp
# python train_net.py --config-file configs/${cfg_name}.yaml \
#     --eval-only MODEL.WEIGHTS ./output/${mode_name}/model_final.pth \
#     OUTPUT_DIR ./output/${mode_name} \
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
#     MODEL.CONDINST.IUVHead.INSTANCE_AWARE_GN True \
#     MODEL.CONDINST.IUVHead.REMOVE_MASK_OVERLAP True \
#     MODEL.CONDINST.v2 True \
#     MODEL.FCOS.INFERENCE_TH_TEST 0.2 \
#     MODEL.CONDINST.IUVHead.RESIDUAL_INPUT True \
#     MODEL.CONDINST.MASK_BRANCH.RESIDUAL_SKIP_AFTER_RELU True \
#     MODEL.ROI_DENSEPOSE_HEAD.NUM_STACKED_CONVS 11 \
#     # SOLVER.AMP.ENABLED True \
#     # > ./output/${mode_name}/eval_log.txt

# cfg_name='densepose_rcnn_R_50_FPN_DL_s1x'
# mode_name=Base_${cfg_name}_InsSeg_1GPU
# python train_net.py --config-file configs/${cfg_name}.yaml \
#     --eval-only MODEL.WEIGHTS ./output/${mode_name}/model_final.pth \
#     # MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS True \
#     # >> ./output/${mode_name}/eval_log.txt

# cfg_name='densepose_rcnn_R_50_FPN_DL_s1x'
# CUDA_LAUNCH_BLOCKING=1 python train_net.py --config-file configs/${cfg_name}.yaml \
#   --resume  --num-gpus 1 \
#     SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 SOLVER.ACCUMULATE_GRAD_ITER 1 \
#     OUTPUT_DIR ./output/${cfg_name}_InsSeg_amp_tmp \
#     SOLVER.CHECKPOINT_PERIOD 2000 \
#     DATALOADER.NUM_WORKERS 6 \
#     MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS True \
#     SOLVER.AMP.ENABLED True \
#     # MODEL.CONDINST.CHECKPOINT_GRAD_NUM 1\

# # 49999
# cfg_name='densepose_CondInst_R_50_s1x'
# mode_name=${cfg_name}_1chSeg_IUVSparsePooler2Head_AggFea_V1ConvXGNSparseInsINLowMemNoOverlapTrueResInput_resIUVOnly_GTinsDilated3_amp
# python train_net.py --config-file configs/${cfg_name}.yaml \
#     --eval-only MODEL.WEIGHTS ./output/${mode_name}/model_final.pth \
#     OUTPUT_DIR ./output/${mode_name} \
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
#     MODEL.CONDINST.IUVHead.INSTANCE_AWARE_GN True \
#     MODEL.CONDINST.IUVHead.REMOVE_MASK_OVERLAP True \
#     MODEL.CONDINST.v2 True \
#     MODEL.FCOS.INFERENCE_TH_TEST 0.2 \
#     MODEL.CONDINST.IUVHead.RESIDUAL_INPUT True \
#     SOLVER.AMP.ENABLED True \
#     MODEL.CONDINST.INFERENCE_GLOBAL_SIUV True \
#     # MODEL.CONDINST.IUVHead.GT_INSTANCES True \
#     # MODEL.CONDINST.MASK_BRANCH.RESIDUAL_SKIP_AFTER_RELU True \
#     # > ./output/${mode_name}/eval_log.txt


# cfg_name='densepose_CondInst_R_50_s1x'
# mode_name=${cfg_name}_1chSeg_IUVSparsePooler2Head_AggFea_V1ConvXGNSparseInsINLowMemNoOverlapTrueResInput_resIUVOnly_GTinsDilated3_amp
# python train_net.py --config-file configs/${cfg_name}.yaml \
#     --eval-only MODEL.WEIGHTS ./output/${mode_name}/model_0109999.pth \
#     OUTPUT_DIR ./output/${mode_name} \
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
#     MODEL.CONDINST.IUVHead.INSTANCE_AWARE_GN True \
#     MODEL.CONDINST.IUVHead.REMOVE_MASK_OVERLAP True \
#     MODEL.CONDINST.v2 True \
#     MODEL.FCOS.INFERENCE_TH_TEST 0.2 \
#     MODEL.CONDINST.IUVHead.RESIDUAL_INPUT True \
#     SOLVER.AMP.ENABLED True \
#     MODEL.CONDINST.INFERENCE_GLOBAL_SIUV True \
#     # MODEL.CONDINST.IUVHead.GT_INSTANCES True \
#     # MODEL.CONDINST.MASK_BRANCH.RESIDUAL_SKIP_AFTER_RELU True \
#     # > ./output/${mode_name}/eval_log.txt

# cfg_name='densepose_CondInst_R_50_s1x'
# mode_name=${cfg_name}_1chSeg_IUVSparsePooler2Head_AggFea_V1ConvXGNSparseInsINLowMemNoOverlapTrueResInputDilate_resIUVOnly_GTinsDilated3_amp
# python train_net.py --config-file configs/${cfg_name}.yaml \
#     --eval-only MODEL.WEIGHTS ./output/${mode_name}/model_final.pth \
#     OUTPUT_DIR ./output/${mode_name} \
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
#     MODEL.CONDINST.IUVHead.INSTANCE_AWARE_GN True \
#     MODEL.CONDINST.IUVHead.REMOVE_MASK_OVERLAP True \
#     MODEL.CONDINST.v2 True \
#     MODEL.FCOS.INFERENCE_TH_TEST 0.2 \
#     MODEL.CONDINST.IUVHead.RESIDUAL_INPUT True \
#     MODEL.CONDINST.IUVHead.DILATION_CONV True \
#     SOLVER.AMP.ENABLED True \
#     MODEL.CONDINST.INFERENCE_GLOBAL_SIUV True \
#     # MODEL.CONDINST.IUVHead.GT_INSTANCES True \
#     # MODEL.CONDINST.MASK_BRANCH.RESIDUAL_SKIP_AFTER_RELU True \
#     # > ./output/${mode_name}/eval_log.txt


cfg_name='densepose_CondInst_R_50_s1x'
mode_name=${cfg_name}_1chSeg_IUVSparsePooler2Head_AggFea_V1ConvXGNSparseInsINLowMemNoOverlapTrueResInputInsEcaAfterRelu_resIUVOnly_GTinsDilated3_amp
python train_net.py --config-file configs/${cfg_name}.yaml \
    --eval-only MODEL.WEIGHTS ./output/${mode_name}/model_0059999.pth \
    OUTPUT_DIR ./output/${mode_name} \
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
    MODEL.CONDINST.IUVHead.INSTANCE_AWARE_GN True \
    MODEL.CONDINST.IUVHead.REMOVE_MASK_OVERLAP True \
    MODEL.CONDINST.v2 True \
    MODEL.FCOS.INFERENCE_TH_TEST 0.2 \
    MODEL.CONDINST.IUVHead.RESIDUAL_INPUT True \
    MODEL.CONDINST.IUVHead.INSTANCE_EFFICIENT_CHANNEL_ATTENTION "AfterRelu" \
    SOLVER.AMP.ENABLED True \
    MODEL.CONDINST.INFERENCE_GLOBAL_SIUV True \
    # MODEL.CONDINST.IUVHead.GT_INSTANCES True \
    # MODEL.CONDINST.MASK_BRANCH.RESIDUAL_SKIP_AFTER_RELU True \
    # > ./output/${mode_name}/eval_log.txt

cfg_name='densepose_CondInst_R_50_s1x'
mode_name=${cfg_name}_1chSeg_IUVSparsePooler2Head_AggFea_V1ConvXGNSparseInsINLowMemNoOverlapTrueResInputInsEcaAfterConv_resIUVOnly_GTinsDilated3_amp
python train_net.py --config-file configs/${cfg_name}.yaml \
    --eval-only MODEL.WEIGHTS ./output/${mode_name}/model_0059999.pth \
    OUTPUT_DIR ./output/${mode_name} \
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
    MODEL.CONDINST.IUVHead.INSTANCE_AWARE_GN True \
    MODEL.CONDINST.IUVHead.REMOVE_MASK_OVERLAP True \
    MODEL.CONDINST.v2 True \
    MODEL.FCOS.INFERENCE_TH_TEST 0.2 \
    MODEL.CONDINST.IUVHead.RESIDUAL_INPUT True \
    MODEL.CONDINST.IUVHead.INSTANCE_EFFICIENT_CHANNEL_ATTENTION "AfterConv" \
    SOLVER.AMP.ENABLED True \
    MODEL.CONDINST.INFERENCE_GLOBAL_SIUV True \
    # MODEL.CONDINST.IUVHead.GT_INSTANCES True \
    # MODEL.CONDINST.MASK_BRANCH.RESIDUAL_SKIP_AFTER_RELU True \
    # > ./output/${mode_name}/eval_log.txt


# ./run_train_dp_5.sh

# cfg_name='densepose_CondInst_R_50_s1x'
# mode_name=${cfg_name}_1chSeg_IUVSparsePooler2Head_AggFea_V1ConvXGNSparseInsINLowMemNoOverlapTrueResInput_IFocalGamma1.5_GTinsDilated3_amp
# python train_net.py --config-file configs/${cfg_name}.yaml \
#     --eval-only MODEL.WEIGHTS ./output/${mode_name}/model_final.pth \
#     OUTPUT_DIR ./output/${mode_name} \
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
#     MODEL.CONDINST.IUVHead.INSTANCE_AWARE_GN True \
#     MODEL.CONDINST.IUVHead.REMOVE_MASK_OVERLAP True \
#     MODEL.CONDINST.v2 True \
#     MODEL.FCOS.INFERENCE_TH_TEST 0.2 \
#     MODEL.CONDINST.IUVHead.RESIDUAL_INPUT True \
#     MODEL.CONDINST.MASK_BRANCH.RESIDUAL_SKIP_AFTER_RELU True \
#     MODEL.CONDINST.INFERENCE_GLOBAL_SIUV True \
#     # SOLVER.AMP.ENABLED True \
#     # > ./output/${mode_name}/eval_log.txt

# cfg_name='densepose_CondInst_R_50_s1x'
# mode_name=${cfg_name}_1chSeg_IUVSparsePooler2Head_AggFea_V1ConvXGNSparseInsINLowMemNoOverlapTrueResInput_IFocalGamma1_GTinsDilated3_amp
# python train_net.py --config-file configs/${cfg_name}.yaml \
#     --eval-only MODEL.WEIGHTS ./output/${mode_name}/model_final.pth \
#     OUTPUT_DIR ./output/${mode_name} \
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
#     MODEL.CONDINST.IUVHead.INSTANCE_AWARE_GN True \
#     MODEL.CONDINST.IUVHead.REMOVE_MASK_OVERLAP True \
#     MODEL.CONDINST.v2 True \
#     MODEL.FCOS.INFERENCE_TH_TEST 0.2 \
#     MODEL.CONDINST.IUVHead.RESIDUAL_INPUT True \
#     MODEL.CONDINST.MASK_BRANCH.RESIDUAL_SKIP_AFTER_RELU True \
#     MODEL.CONDINST.INFERENCE_GLOBAL_SIUV True \
#     # SOLVER.AMP.ENABLED True \
#     # > ./output/${mode_name}/eval_log.txt


# ./run_train_dp_8.sh

# cfg_name='densepose_CondInst_R_50_s1x'
# mode_name=${cfg_name}_1chSeg_IUVSparsePooler2Head_AggFea_V1ConvXGNSparseInsINLowMemNoOverlapTrueResInput_resIUVOnly_addGTSkeletonFea_GTinsDilated3_amp
# python train_net.py --config-file configs/${cfg_name}.yaml \
#     --eval-only MODEL.WEIGHTS ./output/${mode_name}/model_final.pth \
#     OUTPUT_DIR ./output/${mode_name} \
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
#     MODEL.CONDINST.IUVHead.INSTANCE_AWARE_GN True \
#     MODEL.CONDINST.IUVHead.REMOVE_MASK_OVERLAP True \
#     MODEL.CONDINST.v2 True \
#     MODEL.FCOS.INFERENCE_TH_TEST 0.2 \
#     SOLVER.AMP.ENABLED True \
#     MODEL.CONDINST.IUVHead.RESIDUAL_INPUT True \
#     MODEL.CONDINST.IUVHead.SKELETON_FEATURES True \
#     MODEL.CONDINST.IUVHead.GT_SKELETON True \
#     MODEL.CONDINST.INFERENCE_GLOBAL_SIUV True \
#     # MODEL.KEYPOINT_ON True\
#     # MODEL.CONDINST.MASK_BRANCH.RESIDUAL_SKIP_AFTER_RELU True \
#     # > ./output/${mode_name}/eval_log.txt
#     # MODEL.CONDINST.IUVHead.GT_INSTANCES True \
# # 49999
# cfg_name='densepose_CondInst_R_50_s1x'
# mode_name=${cfg_name}_1chSeg_IUVSparsePooler2Head_AggFea_V1ConvXGNSparseInsINLowMemNoOverlapTrueResInput_GTinsDilated3
# python train_net.py --config-file configs/${cfg_name}.yaml \
#     --eval-only MODEL.WEIGHTS ./output/${mode_name}/model_0039999.pth \
#     OUTPUT_DIR ./output/${mode_name} \
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
#     MODEL.CONDINST.IUVHead.INSTANCE_AWARE_GN True \
#     MODEL.CONDINST.IUVHead.REMOVE_MASK_OVERLAP True \
#     MODEL.CONDINST.v2 True \
#     MODEL.FCOS.INFERENCE_TH_TEST 0.2 \
#     MODEL.CONDINST.IUVHead.RESIDUAL_INPUT True \
#     MODEL.CONDINST.MASK_BRANCH.RESIDUAL_SKIP_AFTER_RELU True \
#     # SOLVER.AMP.ENABLED True \
#     # > ./output/${mode_name}/eval_log.txt
#     # MODEL.CONDINST.IUVHead.GT_INSTANCES True \

# # 49999
# cfg_name='densepose_CondInst_R_50_s1x'
# mode_name=${cfg_name}_1chSeg_IUVSparsePooler2Head_AggFea_V1ConvXGNSparseInsINLowMemNoOverlapTrue_GTinsDilated3
# python train_net.py --config-file configs/${cfg_name}.yaml \
#     --eval-only MODEL.WEIGHTS ./output/${mode_name}/model_0039999.pth \
#     OUTPUT_DIR ./output/${mode_name} \
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
#     MODEL.CONDINST.IUVHead.INSTANCE_AWARE_GN True \
#     MODEL.CONDINST.IUVHead.REMOVE_MASK_OVERLAP True \
#     MODEL.CONDINST.v2 False \
#     MODEL.FCOS.INFERENCE_TH_TEST 0.2 \
#     # MODEL.CONDINST.IUVHead.RESIDUAL_SKIP_AFTER_RELU True \
#     # MODEL.CONDINST.MASK_BRANCH.RESIDUAL_INPUT True \
#     # SOLVER.AMP.ENABLED True \
#     # > ./output/${mode_name}/eval_log.txt





# cfg_name='densepose_CondInst_R_50_s1x'
# mode_name=${cfg_name}_1chSeg_IUVSparsePooler2Head_AggFea_V1ConvXGNSparseInsINLowMemNoOverlapTrueResInput_GTinsDilated3_amp_BS1x16
# python train_net.py --config-file configs/${cfg_name}.yaml \
#     --eval-only MODEL.WEIGHTS ./output/${mode_name}/model_0019999.pth \
#     OUTPUT_DIR ./output/${mode_name} \
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
#     MODEL.CONDINST.IUVHead.INSTANCE_AWARE_GN True \
#     MODEL.CONDINST.IUVHead.REMOVE_MASK_OVERLAP True \
#     MODEL.CONDINST.v2 True \
#     SOLVER.AMP.ENABLED True \
#     MODEL.CONDINST.IUVHead.RESIDUAL_INPUT True \
#     MODEL.CONDINST.MASK_BRANCH.RESIDUAL_SKIP_AFTER_RELU True \
#     MODEL.FCOS.INFERENCE_TH_TEST 0.2 \
#     # >> ./output/${mode_name}/eval_log.txt


# cfg_name='densepose_CondInst_R_50_s1x'
# mode_name=${cfg_name}_1chSeg_IUVSparsePooler2Head_AggFea_V1ConvXGNSparseInsINLowMemNoOverlapTrueResInput_GTinsDilated3_amp_BS8x2
# python train_net.py --config-file configs/${cfg_name}.yaml \
#     --eval-only MODEL.WEIGHTS ./output/${mode_name}/model_0019999.pth \
#     OUTPUT_DIR ./output/${mode_name} \
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
#     MODEL.CONDINST.IUVHead.INSTANCE_AWARE_GN True \
#     MODEL.CONDINST.IUVHead.REMOVE_MASK_OVERLAP True \
#     MODEL.CONDINST.v2 True \
#     SOLVER.AMP.ENABLED True \
#     MODEL.CONDINST.IUVHead.RESIDUAL_INPUT True \
#     MODEL.CONDINST.MASK_BRANCH.RESIDUAL_SKIP_AFTER_RELU True \
#     MODEL.FCOS.INFERENCE_TH_TEST 0.2 \
#     # MODEL.CONDINST.IUVHead.GT_INSTANCES True \
#     # >> ./output/${mode_name}/eval_log.txt

# cfg_name='densepose_rcnn_R_50_FPN_DL_s1x'
# mode_name=${cfg_name}_InsSeg_amp_BS8x2_2gpu
# python train_net.py --config-file configs/${cfg_name}.yaml \
#     --eval-only MODEL.WEIGHTS ./output/${mode_name}/model_0019999.pth \
#     MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS True \
#     SOLVER.AMP.ENABLED True \
#     # >> ./output/${mode_name}/eval_log.txt

# cfg_name='densepose_CondInst_R_50_s1x'
# mode_name=${cfg_name}_1chSeg_IUVSparsePooler2Head_AggFea_V1ConvXGNSparseInsINLowMemNoOverlapTrueResInput_GTinsDilated3_amp_BS2x8_2gpu
# python train_net.py --config-file configs/${cfg_name}.yaml \
#     --eval-only MODEL.WEIGHTS ./output/${mode_name}/model_0015999.pth \
#     OUTPUT_DIR ./output/${mode_name} \
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
#     MODEL.CONDINST.IUVHead.INSTANCE_AWARE_GN True \
#     MODEL.CONDINST.IUVHead.REMOVE_MASK_OVERLAP True \
#     MODEL.CONDINST.v2 True \
#     SOLVER.AMP.ENABLED True \
#     MODEL.CONDINST.IUVHead.RESIDUAL_INPUT True \
#     MODEL.CONDINST.MASK_BRANCH.RESIDUAL_SKIP_AFTER_RELU True \
#     MODEL.FCOS.INFERENCE_TH_TEST 0.2 \
#     >> ./output/${mode_name}/eval_log.txt

# cfg_name='densepose_CondInst_R_50_s1x'
# mode_name=${cfg_name}_1chSeg_IUVSparsePooler2Head_AggFea_V1ConvXGNSparseInsINLowMemNoOverlapTrueResInput_GTinsDilated3_amp_BS2x8
# python train_net.py --config-file configs/${cfg_name}.yaml \
#     --eval-only MODEL.WEIGHTS ./output/${mode_name}/model_0007999.pth \
#     OUTPUT_DIR ./output/${mode_name} \
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
#     MODEL.CONDINST.IUVHead.INSTANCE_AWARE_GN True \
#     MODEL.CONDINST.IUVHead.REMOVE_MASK_OVERLAP True \
#     MODEL.CONDINST.v2 True \
#     SOLVER.AMP.ENABLED True \
#     MODEL.CONDINST.IUVHead.RESIDUAL_INPUT True \
#     MODEL.CONDINST.MASK_BRANCH.RESIDUAL_SKIP_AFTER_RELU True \
#     MODEL.FCOS.INFERENCE_TH_TEST 0.2 \
#     >> ./output/${mode_name}/eval_log.txt

#  ./run_train_dp_2.sh




# cfg_name='densepose_CondInst_R_50_s1x'
# mode_name=${cfg_name}_1chSeg_IUVSparsePooler2Head_AggFea_V1ConvXGNSparseInsINLowMem_GTinsDilated3
# python train_net.py --config-file configs/${cfg_name}.yaml \
#     --eval-only MODEL.WEIGHTS ./output/Base_${mode_name}/model_final.pth \
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
#     MODEL.FCOS.INFERENCE_TH_TEST 0.2 \
#     # > ./output/${mode_name}/eval_log.txt
#     # MODEL.CONDINST.IUVHead.GT_INSTANCES True \
#     # MODEL.CONDINST.IUVHead.GT_INSTANCES True \
#     # MODEL.ROI_DENSEPOSE_HEAD.DEEPLAB.NORM "BN" \



# # 49999
# cfg_name='densepose_CondInst_R_50_s1x'
# mode_name=${cfg_name}_1chSeg_IUVSparsePooler2Head_AggFea_V1ConvXBNSparseInsINLowMemNoOverlap128_GTinsDilated3_RelCoordPE5_normCoordBoxHW_v3sumcoord
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
#     MODEL.ROI_DENSEPOSE_HEAD.CONV_HEAD_DIM 128 \
#     MODEL.CONDINST.MASK_BRANCH.AGG_CHANNELS 128 \
#     MODEL.CONDINST.IUVHead.MASK_OUT_BG_FEATURES "hard" \
#     MODEL.CONDINST.IUVHead.DILATE_FGMASK_KENERAL_SIZE 3 \
#     MODEL.ROI_DENSEPOSE_HEAD.DEEPLAB.NORM "InsIN" \
#     MODEL.CONDINST.IUVHead.USE_AGG_FEATURES True \
#     MODEL.CONDINST.IUVHead.INSTANCE_AWARE_GN True \
#     MODEL.CONDINST.IUVHead.REL_COORDS True \
#     MODEL.CONDINST.IUVHead.POSE_EMBEDDING_NUM_FREQS 5 \
#     MODEL.CONDINST.IUVHead.REMOVE_MASK_OVERLAP True \
#     MODEL.FCOS.INFERENCE_TH_TEST 0.2 \
#     # MODEL.CONDINST.IUVHead.REMOVE_MASK_OVERLAP True \
#     # > ./output/${mode_name}/eval_log.txt

# # 49999
# cfg_name='densepose_CondInst_R_50_s1x'
# mode_name=${cfg_name}_1chSeg_IUVSparsePooler2Head_AggFea_V1ConvXBNSparse256_GTinsDilated3_AbsRelCoordPE10_normCoordBoxHW_v3sumcoord
# python train_net.py --config-file configs/${cfg_name}.yaml \
#     --eval-only MODEL.WEIGHTS ./output/${mode_name}/model_0049999.pth \
#     OUTPUT_DIR ./output/${mode_name} \
#     MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS 1 \
#     MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS True \
#     SOLVER.CHECKPOINT_PERIOD 2000 \
#     DATALOADER.NUM_WORKERS 2 \
#     MODEL.CONDINST.IUVHead.NAME "IUVSparsePooler2Head" \
#     MODEL.ROI_DENSEPOSE_HEAD.LOSS_NAME "DensePoseChartGlobalIUVSeparatedSPoolerLoss" \
#     MODEL.ROI_DENSEPOSE_HEAD.NAME "DensePoseV1ConvXGNSparseHead" \
#     MODEL.ROI_DENSEPOSE_HEAD.CONV_HEAD_DIM 256 \
#     MODEL.CONDINST.MASK_BRANCH.AGG_CHANNELS 256 \
#     MODEL.CONDINST.IUVHead.MASK_OUT_BG_FEATURES "hard" \
#     MODEL.CONDINST.IUVHead.DILATE_FGMASK_KENERAL_SIZE 3 \
#     MODEL.ROI_DENSEPOSE_HEAD.DEEPLAB.NORM "BN" \
#     MODEL.CONDINST.IUVHead.USE_AGG_FEATURES True \
#     MODEL.CONDINST.IUVHead.ABS_COORDS True \
#     MODEL.CONDINST.IUVHead.REL_COORDS True \
#     MODEL.CONDINST.IUVHead.POSE_EMBEDDING_NUM_FREQS 10 \
#     MODEL.FCOS.INFERENCE_TH_TEST 0.2 \
#     # MODEL.CONDINST.IUVHead.REMOVE_MASK_OVERLAP True \
#     # > ./output/${mode_name}/eval_log.txt


# cfg_name='densepose_CondInst_R_50_s1x'
# mode_name=${cfg_name}_1chSeg_IUVSparsePooler2Head_AggFea_V1ConvXBNSparse256_GTinsDilated3
# python train_net.py --config-file configs/${cfg_name}.yaml \
#     --eval-only MODEL.WEIGHTS ./output/${mode_name}/model_0049999.pth \
#     OUTPUT_DIR ./output/${mode_name} \
#     MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS 1 \
#     MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS True \
#     SOLVER.CHECKPOINT_PERIOD 2000 \
#     DATALOADER.NUM_WORKERS 2 \
#     MODEL.CONDINST.IUVHead.NAME "IUVSparsePooler2Head" \
#     MODEL.ROI_DENSEPOSE_HEAD.LOSS_NAME "DensePoseChartGlobalIUVSeparatedSPoolerLoss" \
#     MODEL.ROI_DENSEPOSE_HEAD.NAME "DensePoseV1ConvXGNSparseHead" \
#     MODEL.ROI_DENSEPOSE_HEAD.CONV_HEAD_DIM 256 \
#     MODEL.CONDINST.MASK_BRANCH.AGG_CHANNELS 256 \
#     MODEL.CONDINST.IUVHead.MASK_OUT_BG_FEATURES "hard" \
#     MODEL.CONDINST.IUVHead.DILATE_FGMASK_KENERAL_SIZE 3 \
#     MODEL.ROI_DENSEPOSE_HEAD.DEEPLAB.NORM "BN" \
#     MODEL.CONDINST.IUVHead.USE_AGG_FEATURES True \
#     > ./output/${mode_name}/eval_log.txt
#     # MODEL.CONDINST.IUVHead.GT_INSTANCES True \


# # 47999
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
#     MODEL.ROI_DENSEPOSE_HEAD.NAME "DeepLab2SparseHead" \
#     MODEL.ROI_DENSEPOSE_HEAD.CONV_HEAD_DIM 256 \
#     MODEL.CONDINST.IUVHead.MASK_OUT_BG_FEATURES "hard" \
#     MODEL.CONDINST.IUVHead.DILATE_FGMASK_KENERAL_SIZE 3 \
#     MODEL.ROI_DENSEPOSE_HEAD.DEEPLAB.NORM "BN" \
#     > ./output/${mode_name}/eval_log.txt
#     # MODEL.CONDINST.IUVHead.GT_INSTANCES True \


# cfg_name='densepose_CondInst_R_50_s1x'
# mode_name=${cfg_name}_1chSeg_IUVSparsePooler2Head_AggFea_V1ConvXBNSparse256_GTinsDilated3
# python train_net.py --config-file configs/${cfg_name}.yaml \
#     --eval-only MODEL.WEIGHTS ./output/${mode_name}/model_0049999.pth \
#     OUTPUT_DIR ./output/${mode_name} \
#     MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS 1 \
#     MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS True \
#     SOLVER.CHECKPOINT_PERIOD 2000 \
#     DATALOADER.NUM_WORKERS 2 \
#     MODEL.CONDINST.IUVHead.NAME "IUVSparsePooler2Head" \
#     MODEL.ROI_DENSEPOSE_HEAD.LOSS_NAME "DensePoseChartGlobalIUVSeparatedSPoolerLoss" \
#     MODEL.ROI_DENSEPOSE_HEAD.NAME "DensePoseV1ConvXGNSparseHead" \
#     MODEL.ROI_DENSEPOSE_HEAD.CONV_HEAD_DIM 256 \
#     MODEL.CONDINST.MASK_BRANCH.AGG_CHANNELS 256 \
#     MODEL.CONDINST.IUVHead.MASK_OUT_BG_FEATURES "hard" \
#     MODEL.CONDINST.IUVHead.DILATE_FGMASK_KENERAL_SIZE 3 \
#     MODEL.ROI_DENSEPOSE_HEAD.DEEPLAB.NORM "BN" \
#     MODEL.CONDINST.IUVHead.USE_AGG_FEATURES True \
#     > ./output/${mode_name}/eval_log.txt
#     # MODEL.CONDINST.IUVHead.GT_INSTANCES True \


# # 79999
# cfg_name='densepose_CondInst_R_50_s1x'
# mode_name=${cfg_name}_1chSeg_IUVSparsePooler2Head_V1ConvXBNSparse256_GTinsDilated3_lambda1
# python train_net.py --config-file configs/${cfg_name}.yaml \
#     --eval-only MODEL.WEIGHTS ./output/${mode_name}/model_0049999.pth \
#     OUTPUT_DIR ./output/${mode_name} \
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
#     MODEL.CONDINST.IUVHead.NUM_LAMBDA_LAYER 1 \
#     > ./output/${mode_name}/eval_log.txt
#     # MODEL.CONDINST.IUVHead.GT_INSTANCES True \

# # 63999
# cfg_name='densepose_CondInst_R_50_s1x'
# mode_name=${cfg_name}1chSeg_IUVSparsePooler2Head_V1ConvXBNSparse512_GTinsDilated3
# python train_net.py --config-file configs/${cfg_name}.yaml \
#     --eval-only MODEL.WEIGHTS ./output/${mode_name}/model_0049999.pth \
#     OUTPUT_DIR ./output/${mode_name} \
#     MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS 1 \
#     MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS True \
#     SOLVER.CHECKPOINT_PERIOD 2000 \
#     DATALOADER.NUM_WORKERS 2 \
#     MODEL.CONDINST.IUVHead.NAME "IUVSparsePooler2Head" \
#     MODEL.ROI_DENSEPOSE_HEAD.LOSS_NAME "DensePoseChartGlobalIUVSeparatedSPoolerLoss" \
#     MODEL.ROI_DENSEPOSE_HEAD.NAME "DensePoseV1ConvXGNSparseHead" \
#     MODEL.ROI_DENSEPOSE_HEAD.CONV_HEAD_DIM 512 \
#     MODEL.CONDINST.IUVHead.MASK_OUT_BG_FEATURES "hard" \
#     MODEL.CONDINST.IUVHead.DILATE_FGMASK_KENERAL_SIZE 3 \
#     MODEL.ROI_DENSEPOSE_HEAD.DEEPLAB.NORM "BN" \
#     > ./output/${mode_name}/eval_log.txt
# #     # MODEL.CONDINST.IUVHead.GT_INSTANCES True \
# #     MODEL.CONDINST.IUVHead.ABS_COORDS True \
# #     MODEL.CONDINST.IUVHead.REL_COORDS True \
#     # MODEL.CONDINST.IUVHead.POSE_EMBEDDING_NUM_FREQS 10 \

# ./run_train_dp_7.sh

# cfg_name='densepose_CondInst_R_101_s1x'
# mode_name=${cfg_name}_1chSeg_cropResizeNew_deepWide2IUVHead_lambda1_AbsRelCoordPE10_aux0.1Fg
# python train_net.py --config-file configs/${cfg_name}.yaml \
#     --eval-only MODEL.WEIGHTS ./output/${mode_name}/model_final.pth \
#     OUTPUT_DIR ./output/${mode_name} \
#     MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS 1 \
#     MODEL.CONDINST.IUVHead.NUM_CONVS 7 \
#     MODEL.CONDINST.MASK_BRANCH.AGG_CHANNELS 256\
#     MODEL.CONDINST.IUVHead.CHANNELS 256 \
#     MODEL.CONDINST.IUVHead.ABS_COORDS True \
#     MODEL.CONDINST.IUVHead.REL_COORDS True \
#     MODEL.CONDINST.IUVHead.POSE_EMBEDDING_NUM_FREQS 10 \
#     MODEL.CONDINST.IUVHead.NUM_LAMBDA_LAYER 1 \
#     MODEL.CONDINST.IUVHead.OUT_CHANNELS 78 \
#     > ./output/${mode_name}/eval_log.txt
#     # MODEL.CONDINST.IUVHead.GT_INSTANCES True \

# cfg_name='densepose_CondInst_R_50_BiFPN_s1x'
# mode_name=${cfg_name}_1chSeg_cropResizeNew_deepWide2IUVHead_lambda1_AbsRelCoordPE10_GTins_normCoordBoxHW_aux1Rel
# python train_net.py --config-file configs/${cfg_name}.yaml \
#     --eval-only MODEL.WEIGHTS ./output/${mode_name}/model_0079999.pth \
#     OUTPUT_DIR ./output/${mode_name} \
#     MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS 1 \
#     MODEL.CONDINST.IUVHead.NUM_CONVS 7 \
#     MODEL.CONDINST.MASK_BRANCH.AGG_CHANNELS 256\
#     MODEL.CONDINST.IUVHead.CHANNELS 256 \
#     MODEL.CONDINST.IUVHead.ABS_COORDS True \
#     MODEL.CONDINST.IUVHead.REL_COORDS True \
#     MODEL.CONDINST.IUVHead.POSE_EMBEDDING_NUM_FREQS 10 \
#     MODEL.CONDINST.IUVHead.NUM_LAMBDA_LAYER 1 \
#     MODEL.CONDINST.IUVHead.OUT_CHANNELS 78 \
#     MODEL.CONDINST.IUVHead.NORM_COORD_BOXHW True \
#     > ./output/${mode_name}/eval_log.txt
#     # MODEL.CONDINST.IUVHead.GT_INSTANCES True \

# cfg_name='densepose_CondInst_R_50_s1x'
# mode_name=${cfg_name}_1chSeg_cropResizeNew_deepWide2IUVHead_lambda1_AbsRelCoordPE10
# python train_net.py --config-file configs/${cfg_name}.yaml \
#     --eval-only MODEL.WEIGHTS ./output/${mode_name}/model_0079999.pth \
#     OUTPUT_DIR ./output/${mode_name} \
#     MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS 1 \
#     MODEL.CONDINST.IUVHead.NUM_CONVS 7 \
#     MODEL.CONDINST.MASK_BRANCH.AGG_CHANNELS 256\
#     MODEL.CONDINST.IUVHead.CHANNELS 256 \
#     MODEL.CONDINST.IUVHead.ABS_COORDS True \
#     MODEL.CONDINST.IUVHead.REL_COORDS True \
#     MODEL.CONDINST.IUVHead.POSE_EMBEDDING_NUM_FREQS 10 \
#     MODEL.CONDINST.IUVHead.NUM_LAMBDA_LAYER 1 \
#     > ./output/${mode_name}/eval_log.txt
#     # MODEL.CONDINST.IUVHead.GT_INSTANCES True \

# cfg_name='densepose_CondInst_R_50_BiFPN_s1x'
# mode_name=${cfg_name}_1chSeg_cropResizeNew_deepWide2IUVHead_lambda1_AbsRelCoordPE10_GTins
# python train_net.py --config-file configs/${cfg_name}.yaml \
#     --eval-only MODEL.WEIGHTS ./output/${mode_name}/model_0079999.pth \
#     OUTPUT_DIR ./output/${mode_name} \
#     MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS 1 \
#     MODEL.CONDINST.IUVHead.NUM_CONVS 7 \
#     MODEL.CONDINST.MASK_BRANCH.AGG_CHANNELS 256\
#     MODEL.CONDINST.IUVHead.CHANNELS 256 \
#     MODEL.CONDINST.IUVHead.ABS_COORDS True \
#     MODEL.CONDINST.IUVHead.REL_COORDS True \
#     MODEL.CONDINST.IUVHead.POSE_EMBEDDING_NUM_FREQS 10 \
#     MODEL.CONDINST.IUVHead.NUM_LAMBDA_LAYER 1 \
#     > ./output/${mode_name}/eval_log.txt
#     # MODEL.CONDINST.IUVHead.GT_INSTANCES True \


# cfg_name='densepose_CondInst_R_50_BiFPN_s1x'
# mode_name=${cfg_name}_1chSeg_cropResizeNew_deepWide2IUVHead_lambda1_AbsRelCoordPE10_GTins_normCoordBoxHW
# python train_net.py --config-file configs/${cfg_name}.yaml \
#     --eval-only MODEL.WEIGHTS ./output/${mode_name}/model_0079999.pth \
#     OUTPUT_DIR ./output/${mode_name} \
#     MODEL.ROI_DENSEPOSE_HEAD.NUM_COARSE_SEGM_CHANNELS 1 \
#     MODEL.CONDINST.IUVHead.NUM_CONVS 7 \
#     MODEL.CONDINST.MASK_BRANCH.AGG_CHANNELS 256\
#     MODEL.CONDINST.IUVHead.CHANNELS 256 \
#     MODEL.CONDINST.IUVHead.ABS_COORDS True \
#     MODEL.CONDINST.IUVHead.REL_COORDS True \
#     MODEL.CONDINST.IUVHead.POSE_EMBEDDING_NUM_FREQS 10 \
#     MODEL.CONDINST.IUVHead.NUM_LAMBDA_LAYER 1 \
#     MODEL.CONDINST.IUVHead.NORM_COORD_BOXHW True \
#     > ./output/${mode_name}/eval_log.txt

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


