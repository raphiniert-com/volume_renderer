% This example uses two channels and two light-sources to render one static image

% add VolumeRender to path
addpath('VolumeRender');

path='/home/scheiblr/git/raphiniert/volumerenderer/data/';

filename = [path '/ViBE-Z_72hpf_v1.h5']; 

dataset = '/anatomy/average_brain';
data_main = h5read(filename, dataset);
elementSizeUm = h5readatt(filename, dataset,'element_size_um');

dataset = '/expression/3A10';
data_structure = h5read(filename, dataset);
% elementSizeUm = h5readatt(filename, dataset,'element_size_um');
emission_structure = Volume(data_structure);

% create render object
render = VolumeRender();

% setup illumination settings
render.VolumeIllumination=Volume(HenyeyGreenstein(64));
render.LightSources = [LightSource([1,2,-3],[0,1,1]), LightSource([0,1,9],[1,0.5,1])];
render.ElementSizeUm=elementSizeUm;
render.FocalLength=3.0;
render.DistanceToObject=6;
render.rotate(140,25,0);
render.OpacityThreshold=0.95;

% rotate some more
%render.rotate(0,-120,0);

% setup image size (of the resulting 2D image)
render.ImageResolution=size(emission_structure.Data,[2, 1]);

% set render volumes
render.VolumeEmission=emission_structure;
render.VolumeAbsorption=emission_structure;
render.VolumeReflection=reflection;
render.ScaleReflection=1;

render.Color = [1,1,0];

rendered_image_1 = render.render();

data_absorption = data_main;
half_size=size(data_absorption)/2;
data_absorption(1:half_size(1), 1:half_size(2), 1:end) = 0;

absorptionVolume=Volume(data_main);
absorptionVolume.resize(half_size);

render.VolumeEmission=emission_main;
render.VolumeAbsorption=absorptionVolume;
render.ScaleReflection=0.2;
render.Color = [1,1,1];

rendered_image_2 = render.render();

imshow(rendered_image_1+rendered_image_2);