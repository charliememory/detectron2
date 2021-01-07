source /HPS/HumanBodyRetargeting7/work/For_Liqian/.bashrc_liqianma

cfg_name='MS_R_50_1x'
OMP_NUM_THREADS=1 python tools/train_net.py \
    --config-file configs/CondInst/${cfg_name}.yaml \
    OUTPUT_DIR ./output/${cfg_name} \
    # --num-gpus 8 \
