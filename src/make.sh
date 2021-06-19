#!/usr/bin/env bash

mex_c="/home/scheiblr/Spielewiese/R2021a/bin/mex"

$mex_c -largeArrayDims -f /home/scheiblr/Spielewiese/R2021a/toolbox/parallel/gpu/extern/src/mex/glnxa64/nvcc_g++.xml NVCC_FLAGS="" \
  -v -cxx -D_DEBUG C/mex/render.cpp C/vr/volumeRender.cpp C/vr/volumeRender_kernel.cu -I"../lib/cuda-samples/Common/" -I"C/" \
  -lcuda -output matlab/VolumeRender/volumeRender 
# $mex_c -cxx -largeArrayDims C/mex/timestamp.cpp -output matlab/VolumeRender/timestamp
# $mex_c -largeArrayDims C/mex/HenyeyGreenstein.cc -I"C/" -output matlab/VolumeRender/HenyeyGreenstein
# $mex_c -largeArrayDims -f /home/scheiblr/Spielewiese/R2021a/toolbox/parallel/gpu/extern/src/mex/glnxa64/nvcc_g++.xml NVCC_FLAGS="" -cxx -D_DEBUG C/mex/mmanager.cpp -I"../lib/cuda-samples/Common/" -I"C/" -output matlab/memory_manager/mmanager
