ReadMe.md

requirement.txt
torch                  1.6.0
pycocotools            2.0.2
detectron2			   0.3
spconv

# trouble-shooting when installing spconv

If miss lib, then add path in setup.py, e.g.
`
            cmake_args += ['-DCUDA_cufft_LIBRARY=/esat/dragon/liqianma/workspace/miniconda3/envs/pytorch/lib/libcufft.so']
`

If '-- The CUDA compiler identification is unknown', then add one line in the Cmakelist.txt before the line 5
`set(CMAKE_CUDA_COMPILER "/path_to_cuda/bin/nvcc")`
Ref: https://github.com/traveller59/spconv/issues/21#issuecomment-495889887

