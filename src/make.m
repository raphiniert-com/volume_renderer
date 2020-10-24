% compiling required mex files

% build render command
mexcuda -v  -cxx -D_DEBUG -largeArrayDims C/render.cpp C/volumeRender.cpp C/volumeRender_kernel.cu ...
    -I"../lib/cuda-samples/Common/" -I"C/" -lcuda -output matlab/VolumeRender/volumeRender

% build command to create illumination model
mex -largeArrayDims C/illumination/HenyeyGreenstein.cc -I"C/illumination/" -output matlab/VolumeRender/HenyeyGreenstein

% build command to create timestamp command (using c++11)
mex -cxx -largeArrayDims C/timestamp.cpp -output matlab/VolumeRender/timestamp
