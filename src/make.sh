#!/usr/bin/env bash

mex_c="/home/scheiblr/Spielewiese/R2020b/bin/mex"

$mex_c -largeArrayDims -f /home/scheiblr/Spielewiese/R2020b/toolbox/parallel/gpu/extern/src/mex/glnxa64/nvcc_g++.xml NVCC_FLAGS="" -v -cxx -D_DEBUG -largeArrayDims C/mex/render.cpp C/vr/volumeRender.cpp C/vr/volumeRender_kernel.cu -I"../lib/cuda-samples/Common/" -I"C/common" -I"C/vr/" -lcuda -output matlab/VolumeRender/volumeRender 
$mex_c -cxx -largeArrayDims C/mex/timestamp.cpp -output matlab/VolumeRender/timestamp
$mex_c -largeArrayDims C/mex/HenyeyGreenstein.cc -I"C/vr/illumination" -output matlab/VolumeRender/HenyeyGreenstein
$mex_c -largeArrayDims C/mex/mmanager.cpp -I"C/common" -output matlab/memory_manager/mmanager