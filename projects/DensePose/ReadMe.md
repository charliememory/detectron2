ReadMe.md

requirement.txt
torch                  1.6.0
pycocotools            2.0.2
detectron2             0.3
spconv                 1.2.1 [compiled with gcc-7.0.4]
av                     8.0.2
scipy                  1.5.4
[optional] lambda-networks 0.4.0

# Build modules of AdelaiDet (incl. CondInst) into DensePose
`python setup.py build develop`

# Build spconv (trouble-shooting)

If miss lib, then add path in setup.py, e.g.
`
            cmake_args += ['-DCUDA_cufft_LIBRARY=/esat/dragon/liqianma/workspace/miniconda3/envs/pytorch/lib/libcufft.so']
`

If '-- The CUDA compiler identification is unknown', then add one line in the Cmakelist.txt before the line 5
`set(CMAKE_CUDA_COMPILER "/path_to_cuda/bin/nvcc")`
Ref: https://github.com/traveller59/spconv/issues/21#issuecomment-495889887

If 'Segmentation fault during GPU forward', it may be related to the gcc version, try 7.4.0
Ref: https://github.com/traveller59/spconv/issues/245#issuecomment-735370682


# Results
## densepose_rcnn_R_50_FPN_DL_s2x 
### ins_num=None
`Inference done 57/1508. 0.1139 s / img. ETA=0:13:33`
### ins_num=1
`Inference done 76/603. 0.0848 s / img. ETA=0:02:45`
### ins_num=14
`Inference done 26/87. 0.1764 s / img. ETA=0:01:06`

## Ours
### ins_num=None
`Inference done 218/1508. 0.1158 s / img. ETA=0:02:38`
### ins_num=1
`Inference done 330/603. 0.1062 s / img. ETA=0:00:30`
### ins_num=14
`Inference done 70/87. 0.1481 s / img. ETA=0:00:02`


# PoseTrack Results
## Official results https://github.com/facebookresearch/DensePose/tree/master/PoseTrack

## densepose_rcnn_R_50_FPN_DL_s2x
Total inference time: 0:07:34.423428 (0.584844 s / img per device, on 1 devices)
Total inference pure compute time: 0:01:23 (0.107573 s / img per device, on 1 devices)

## Ours
Total inference time: 0:02:12.386951 (0.170382 s / img per device, on 1 devices)
Total inference pure compute time: 0:01:43 (0.133565 s / img per device, on 1 devices)


