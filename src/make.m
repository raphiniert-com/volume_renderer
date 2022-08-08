% compiling required mex files

setenv('MW_NVCC_PATH','C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7\bin')

% enable if debug output of renderer is desired
debug=false;

directory_content = dir; % contains everything of the current directory
exe_path = directory_content(1).folder; % returns the path that is currently open

% build render command
% if debug is desired, add " -DDEBUG", ...

str_debug='';
if debug==true
    str_debug = " -DDEBUG";
end

eval(strcat("mexcuda ",  ...
    fullfile(exe_path, "C", "render.cpp"), ... 
    " ", fullfile(exe_path, "C", "volumeRender.cpp"), ...
    " ", fullfile(exe_path, "C", "volumeRender_kernel.cu"), ...
    " -I", fullfile(exe_path, "..", "lib","cuda-samples", "Common"), ...
    " -I", fullfile(exe_path, "C"), ...
    " -lcuda", ...
    str_debug, ...
    " -DMATLAB_MEX_FILE", ...
    " -output ", fullfile(exe_path,"matlab", "VolumeRender", "volumeRender")));

% build command to create illumination model
eval(strcat("mex -R2018a -O", ...
    " -I", fullfile(exe_path, "C","illumination"), ...
    " -outdir ", fullfile(exe_path, "matlab", "VolumeRender"), " ", ... 
    fullfile(exe_path, "C","illumination","HenyeyGreenstein.cc")));
