ReadMe.md

requirement.txt
torch                  1.6.0
pycocotools            2.0.2
detectron2             0.3
spconv                 1.2.1 [compiled]
av                     8.0.2
scipy                  1.5.4
[optional] lambda-networks 0.4.0

# Build modules of AdelaiDet (incl. CondInst) into DensePose
python setup.py build develop

# Build spconv (trouble-shooting)

If miss lib, then add path in setup.py, e.g.
`
            cmake_args += ['-DCUDA_cufft_LIBRARY=/esat/dragon/liqianma/workspace/miniconda3/envs/pytorch/lib/libcufft.so']
`

If '-- The CUDA compiler identification is unknown', then add one line in the Cmakelist.txt before the line 5
`set(CMAKE_CUDA_COMPILER "/path_to_cuda/bin/nvcc")`
Ref: https://github.com/traveller59/spconv/issues/21#issuecomment-495889887

If 'Segmentation fault during GPU forward', it may be related to the gcc version, try 7.4.0

