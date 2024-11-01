% This example uses two channels and two light-sources to render one static image

%% init
% estimate working path, so that the script runs from any location
workingpath = erase(mfilename('fullpath'), 'example1');

% add VolumeRender to path
addpath(fullfile(workingpath, '..', 'src', 'matlab', 'VolumeRender'));
addpath(fullfile(workingpath, '..', 'src', 'matlab', 'StopWatch'));

sw = Stopwatch('timings');
sw.add('r', 'benchmark rendering');

% folder to the volume files
path=fullfile(workingpath, 'h5-data');

filename = fullfile(path, 'ViBE-Z_72hpf_v1.h5'); 
dataset = '/anatomy/average_brain';

data_main = h5read(filename, dataset);

elementSizeUm = h5readatt(filename, dataset,'element_size_um');
emission_main = Volume(data_main);

dataset = '/expression/3A10';
data_structure = h5read(filename, dataset);

emission_structure = Volume(data_structure);

%% setup general render settings
% create render object
render = VolumeRender();

render.ScatteringWeight = 1;

% setup illumination settings
render.LightSources = [LightSource([0,0.1,0],[1,1,1], LightType.Diffuse)];
% render.VolumePhase = Volume(HenyeyGreenstein_LUT(64));

% misc parameters
render.ElementSizeUm=elementSizeUm;
render.FocalLength=3.0;
render.DistanceToObject=6;

render.rotate(125,25,0);

render.OpacityThreshold=0.9;


% setup image size (of the resulting 2D image)
render.ImageResolution=size(emission_structure.Data,[1, 2]);

%% first image (structure)

% set render volumes
render.VolumeEmission=emission_structure;
render.VolumeAbsorption=emission_structure;
render.FactorAbsorption=0.6;
render.FactorReflection=0.4;

render.Color = [1,1,0];

rendered_image_structure = render.render();

render.memInfo();

%% second image (main zebra fish)
absorptionVolume=Volume(data_main);
absorptionVolume.resize(0.5);
absorptionVolume.normalize(0,1);

render.VolumeEmission=emission_main;
render.VolumeAbsorption=absorptionVolume;
% make it kind of transparent
render.FactorEmission=1;
render.FactorAbsorption=0.4;
render.FactorReflection=0.6;
render.Color = [1,1,1];

sw.start('r');
rendered_image_main = render.render();
sw.stop('r');

sw.print();

%% display the images and the combined one
figure;
imshow(im2double(rendered_image_main+rendered_image_structure));