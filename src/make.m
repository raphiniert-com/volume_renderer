% compiling required mex files

%% init
% estimate working path, so that the script runs from any location
workingpath = erase(mfilename('fullpath'), 'make');

% add Stopwatch
addpath(fullfile(workingpath, 'matlab', 'StopWatch'));

% setup compiler path
setenv('MW_NVCC_PATH','C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7\bin')

% enable if debug output of renderer is desired
debug=false;

directory_content = dir; % contains everything of the current directory
exe_path = directory_content(1).folder; % returns the path that is currently open

sw = Stopwatch('timings');

%% build render command
% if debug is desired, add " -DDEBUG", ...
sw.add('r', 'compilation renderer');

sw.start('r');

str_debug='';
if debug==true
    str_debug = " -DDEBUG";
end

eval(strcat("mexcuda ",  ...
    fullfile(exe_path, "C", "mex", "render.cpp"), ... 
    " ", fullfile(exe_path, "C", "vr", "volumeRender.cpp"), ...
    " ", fullfile(exe_path, "C", "vr", "volumeRender_kernel.cu"), ...
    " -I", fullfile(exe_path, "..", "lib","cuda-samples", "Common"), ...
    " -I", fullfile(exe_path, "C"), ...
    " -lcuda", ...
    str_debug, ...
    " -output ", fullfile(exe_path,"matlab", "VolumeRender", "volumeRender")));

sw.stop('r');

%% build command to generate timestamp
sw.add('t', 'compilation timestamp');

sw.start('t');
eval(strcat("mex -R2018a -O", ...
    " -I", fullfile(exe_path, "C"), ...
    " -outdir ", fullfile(exe_path, "matlab", "VolumeRender"), " ", ... 
    fullfile(exe_path, "C", "mex","timestamp.cpp")));
sw.stop('t');

sw.print();
