% This example uses two channels and one LightSource to render a movie.
% Furthermore, we implemented special effects.

% add VolumeRender to path
addpath('VolumeRender');

% some tests for the matlab-C++ connection
path='../../h5-data/';

filename = [path '/ViBE-Z_72hpf_v1.h5']; 

dataset = '/anatomy/average_brain';
data_main = h5read(filename, dataset);
elementSizeUm = h5readatt(filename, dataset,'element_size_um');

dataset = '/expression/3A10';
data_structure = h5read(filename, dataset);

% create volumes
emission_structure = Volume(data_structure);
emission_main = Volume(data_main);

%% setup for rendering main channel

% total frames of the movie
total_frames=120;

% rotation performed in the entire movie
rotation_of_movie=1200;


% create empty structure for all rendered images
rendered_images_main = zeros([  size(emission_structure.Data,2), ...
                                size(emission_structure.Data,1), ...
                                3, ...
                                total_frames]);


% create render object
render = VolumeRender();

% setup render options
render.Color = [1,1,1];

render.ScaleEmission=1;
render.ScaleAbsorption=2;
render.ScaleReflection=1;

render.ElementSizeUm=elementSizeUm;
render.FocalLength=3.0;
render.DistanceToObject = 6;
render.OpacityThreshold=0.95;

% rotate
render.rotate(-90,0,0);
render.rotate(0,45,0);

% setup illumintation
render.LightSources = LightSource([-15,15,0], [1,1,1]);

% setup volumes
data_absorption = data_main;
absorptionVolume=Volume(data_absorption);
half_size=size(data_absorption)/2;
absorptionVolume.resize(half_size);
absorptionVolume.normalize(0,1);

render.VolumeAbsorption=absorptionVolume;
render.VolumeEmission=emission_main;

render.VolumeIllumination=Volume(HenyeyGreenstein(64));

% setup image size (of the resulting 2D image)
render.ImageResolution=size(emission_main.Data,[1 2]);

% rotation per step
beta = rotation_of_movie/total_frames;


%% start renderung main channel
disp('rendering main channel');


start_frame=1;
end_frame=total_frames/4;
for i=start_frame:end_frame
    disp(strcat(' -image ', num2str(i)));

    rendered_image = render.render();
    
    % rotate object
    render.rotate(0,beta,0);
    
    rendered_images_main(:,:,:,i) = rendered_image;
end

start_frame=(total_frames/4)+1;
end_frame=total_frames/2;
for i=start_frame:end_frame
    disp(strcat(' -image ', num2str(i)));

    absorptionVolume=Volume(data_absorption);
    half_size=size(data_absorption)/2;
    data_absorption(1:end, 1:half_size(2), 1:end) = (i/start_frame)/(end_frame);
    absorptionVolume.resize(half_size);
    absorptionVolume.normalize(0,1);

    render.VolumeAbsorption=absorptionVolume;


    rendered_image = render.render();
    
    % rotate object
    render.rotate(0,beta,0);

    rendered_images_main(:,:,:,i) = rendered_image;
end

start_frame=(total_frames/2)+1;
end_frame=total_frames;
for i=start_frame:end_frame
    disp(strcat(' -image ', num2str(i)));
    
    rendered_image = render.render();
    
    % rotate object
    render.rotate(0,beta,0);

    rendered_images_main(:,:,:,i) = rendered_image;
end


%% render structure image channel
% create empty structure for all rendered images
rendered_images_structure = zeros([  size(emission_structure.Data,2), ...
                                size(emission_structure.Data,1), ...
                                3, ...
                                total_frames]);
                            
disp('rendering structure channel');

% reset render
render.RotationMatrix = eye(3);
render.rotate(-90,0,0);
render.rotate(0,45,0);

% setup render
render.ImageResolution=size(emission_structure.Data,[1 2]);


render.VolumeAbsorption=emission_structure;
render.VolumeEmission=emission_structure;

render.ScaleEmission=1;
render.ScaleAbsorption=0.6;
render.ScaleReflection=1;

render.Color = [1,1,0];


start_frame=1;
end_frame=total_frames;
for i=start_frame:end_frame
    disp(strcat(' -image ', num2str(i)));

    rendered_image = render.render();
    
    % rotate object
    render.rotate(0,beta,0);
    
    rendered_images_structure(:,:,:,i) = rendered_image;
end


%% create one movie
rendered_images_combined = rendered_images_main+rendered_images_structure;

normalized_images = VolumeRender.normalizeSequence(rendered_images_combined);

mov = immovie(normalized_images);
implay(mov);