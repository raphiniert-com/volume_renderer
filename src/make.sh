#!/usr/bin/env bash

mex_c="/home/scheiblr/Spielewiese/R2020b/bin/mex"

$mex_c -largeArrayDims -f /home/scheiblr/Spielewiese/R2020b/toolbox/parallel/gpu/extern/src/mex/glnxa64/nvcc_g++.xml NVCC_FLAGS="" -v -cxx -D_DEBUG -largeArrayDims C/render.cpp C/volumeRender.cpp C/volumeRender_kernel.cu -I"../lib/cuda-samples/Common/" -I"C/" -lcuda -output matlab/VolumeRender/volumeRender 
$mex_c -cxx -largeArrayDims C/timestamp.cpp -output matlab/VolumeRender/timestamp
$mex_c -largeArrayDims C/illumination/HenyeyGreenstein.cc -I"C/illumination/" -output matlab/VolumeRender/HenyeyGreenstein