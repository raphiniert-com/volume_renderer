% This example uses two channels and two light-sources to render one static image

% add VolumeRender to path
addpath('VolumeRender');

path='/home/scheiblr/git/raphiniert/volumerenderer/data/';

filename = [path '/ViBE-Z_72hpf_v1.h5']; 

dataset = '/anatomy/average_brain';
data_main = h5read(filename, dataset);
elementSizeUm = h5readatt(filename, dataset,'element_size_um');
emission_main = Volume(data_main);
emission_main.resize(size(emission_main.Data)/2);

dataset = '/expression/3A10';
data_structure = h5read(filename, dataset);
% elementSizeUm = h5readatt(filename, dataset,'element_size_um');
emission_structure = Volume(data_structure);
emission_structure.resize(size(emission_structure.Data)/2);

% create render object
render = VolumeRender();

% setup illumination settings
render.VolumeIllumination=Volume(HenyeyGreenstein(64));
% render.LightSources = [LightSource([1,2,-3],[0,1,1]), LightSource([0,1,9],[1,0.5,1])];
render.ElementSizeUm=elementSizeUm;
render.FocalLength=3.0;
render.DistanceToObject=6;

render.rotate(-125,25,0);

render.OpacityThreshold=0.9;


%% perform rendering
% setup image size (of the resulting 2D image)
render.ImageResolution=size(emission_structure.Data,[1, 2]);

tic
% set render volumes
render.VolumeEmission=emission_structure;
render.VolumeAbsorption=emission_structure;
render.ScaleAbsorption=0.6;
render.ScaleReflection=0.4;

render.Color = [1,1,0];

rendered_image_structure = render.render();

absorptionVolume=Volume(data_main);
absorptionVolume.resize(0.5);
absorptionVolume.normalize(0,1);

render.VolumeEmission=emission_main;
render.VolumeAbsorption=absorptionVolume;
render.ScaleAbsorption=0.8;
render.ScaleReflection=1;
render.Color = [1,1,1];

rendered_image_main = render.render();
toc

imshow(rendered_image_main+rendered_image_structure);