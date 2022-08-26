% This example uses two channels and two light-sources to render one static image

%% init
% add VolumeRender to path
addpath('matlab/VolumeRender');

path='../h5-data/';

filename = [path '/ViBE-Z_72hpf_v1.h5']; 

dataset = '/anatomy/average_brain';
data_main = h5read(filename, dataset);

elementSizeUm = h5readatt(filename, dataset,'element_size_um');
emission_main = Volume(data_main);

dataset = '/expression/3A10';
data_structure = h5read(filename, dataset);
% elementSizeUm = h5readatt(filename, dataset,'element_size_um');
emission_structure = Volume(data_structure);

[gX, gY, gZ] = emission_main.grad();


%% setup general render settings
% create render object
render = VolumeRender();

% set gradient as volumes
render.VolumeGradientX = gX;
render.VolumeGradientY = gY;
render.VolumeGradientZ = gZ;

% setup illumination settings
render.VolumeIllumination=Volume(HenyeyGreenstein(64));
render.LightSources = [LightSource([1,2,-3],[0,1,1]), LightSource([0,1,9],[1,0.5,1])];

% misc parameters
render.ElementSizeUm=elementSizeUm;
render.FocalLength=3.0;
render.DistanceToObject=6;

render.rotate(-125,25,0);

render.OpacityThreshold=0.9;


% setup image size (of the resulting 2D image)
render.ImageResolution=size(emission_structure.Data,[1, 2]);

%% first image (structure)

% set render volumes
render.VolumeEmission=emission_structure;
render.VolumeAbsorption=emission_structure;
render.ScaleAbsorption=0.6;
render.ScaleReflection=0.4;

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
render.ScaleEmission=0.1;
render.ScaleAbsorption=0.4;
render.ScaleReflection=0.1;
render.Color = [1,1,1];

rendered_image_main = render.render();


%% display the images and the combined one
figure;
imshow(rendered_image_main);
figure;
imshow(rendered_image_structure);
figure;
imshow(rendered_image_main+rendered_image_structure);

render.memInfo();

%% with gradient computation
render.resetGradientVolumes();
rendered_image_grad_computed = render.render();

figure;
imshow(rendered_image_grad_computed);

%% volume swap test
% render.VolumeEmission=absorptionVolume;
% render.VolumeAbsorption=emission_main;
% 
% figure;
% imshow(render.render());