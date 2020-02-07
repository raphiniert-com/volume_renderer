% This example uses two channels and one LightSource to render a movie.
% Furthermore, we implemented special effects.

% add VolumeRender to path
addpath('VolumeRender');

% some tests for the matlab-C++ connection
path='/home/scheiblr/git/raphiniert/volumerenderer/data/';

% l = [LightSource([1,2,3],[3,3,4]), LightSource([0,1,9],[13,4,6])];

% filename = [path 'tmp_st37_Emb2.h5']; dataset = '/t0/channel3';

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

% central_z = round(size(emission.Data,3)/2);
% imshow(emission.Data(:,:,central_z)' * 10,[]);

% setup illumination settings
render.VolumeIllumination=Volume(HenyeyGreenstein(64));
render.LightSources = LightSource([-15,15,0], [1,1,1]);
render.ElementSizeUm=elementSizeUm;
render.FocalLength=3.0;
render.DistanceToObject = 6;
render.rotate(90,0,0);
render.OpacityThreshold=0.95;

% % rotate some more
% render.rotate(0,45,0);
% 
% % setup image size (of the resulting 2D image)
% render.ImageResolution= size(emission_structure.Data,[2, 1]);
% 
% % set render volumes
% render.VolumeEmission=emission_structure;
% render.VolumeAbsorption=emission_structure;
% render.VolumeReflection=reflection;
% render.ScaleReflection=1;
% 
% render.Color = [1,1,0];
% 
% rendered_image_1 = render.render();
% 
% data_absorption = data_main;
% half_size=size(data_absorption)/2;
% data_absorption(1:end, 1:half_size(2), 1:end) = 0;
% absorptionVolume=Volume(data_absorption);
% absorptionVolume.resize(half_size);
% 
% render.VolumeEmission=emission_main;
% render.VolumeAbsorption=absorptionVolume;
% render.ScaleReflection=0.2;
% render.Color = [1,1,1];
% 
% rendered_image_2 = render.render();
% 
% imshow(rendered_image_1+rendered_image_2);
% return;

% create volumes
emission_structure = Volume(data_structure);
emission_main = Volume(data_main);

reflection=Volume([1,1,1;1,1,1;1,1,1]);
render.VolumeReflection=reflection;

% setup image size (of the resulting 2D image)
render.ImageResolution=[size(emission_structure.Data,2), size(emission_structure.Data,1)];


nStep=24;
rendered_image = zeros([size(emission_structure.Data,2),size(emission_structure.Data,1), 3, 360/nStep]);

beta = 360/nStep;
for i=1:nStep
    display(strcat('image ', num2str(i)));

    % set render volumes
    render.VolumeEmission=emission_structure;
    render.VolumeAbsorption=emission_structure;
    render.VolumeReflection=reflection;
    render.ScaleReflection=1;

    render.Color = [1,1,0];

    rendered_image_1 = render.render();

    data_absorption = data_main;
    half_size=size(data_absorption)/2;
    data_absorption(1:end, 1:half_size(2), 1:end) = 0;
    absorptionVolume=Volume(data_absorption);
    absorptionVolume.resize(half_size);
    min(absorptionVolume.Data(:))

    render.VolumeEmission=emission_main;
    render.VolumeAbsorption=absorptionVolume;
    render.ScaleReflection=0.2;
    render.Color = [1,1,1];

    rendered_image_2 = render.render();
    
    % rotate object
    render.rotate(0,beta,0);

    result = rendered_image_1+rendered_image_2;
    rendered_image(:,:,:,i) = result/max(result(:));
end

normalizedImages = VolumeRender.normalizeSequence(rendered_image);

mov = immovie(normalizedImages);
implay(mov);