% compiling required mex files

% build render command
mexcuda C/render.cpp C/volumeRender.cpp C/volumeRender_kernel.cu -I"../lib/cuda-samples/Common/" -I"C/" -lcuda -output matlab/VolumeRender/volumeRender

% build command to create illumination model
mex C/illumination/HenyeyGreenstein.cc -I"C/illumination/" -output matlab/VolumeRender/HenyeyGreenstein