% create a complex movie:
%  start fading in the objects using one light source
%  rotate 360+90 degree
%  then let's fade in some structure and disappear half of the surface
%  install a second light, which improves the illumination from the brain
%  zoom slowly in and after a while of inspection zoom out again and fade
%  out the scene

% 226e710b-725b-4bec-840e-bf47be2b8a44.h5: http://mridata.org/download/226e710b-725b-4bec-840e-bf47be2b8a44

% This example uses one channel and one LightSource to render a movie.

%% init
% estimate working path, so that the script runs from any location
workingpath = erase(mfilename('fullpath'), 'example4');

% add VolumeRender to path
addpath(fullfile(workingpath, '..', 'src', 'matlab', 'VolumeRender'));
addpath(fullfile(workingpath, '..', 'src', 'matlab', 'StopWatch'));

% folder to the volume files
path=fullfile(workingpath, 'nifti-data');

% dataset: https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation/data
filename_t1 = fullfile(path, 'BraTS20_Training_001_t1.nii');
filename_seg = fullfile(path, 'BraTS20_Training_001_seg.nii');

% filename_flair = fullfile(path, 'BraTS20_Training_001_flair.nii');
% filename_t1ce = fullfile(path, 'BraTS20_Training_001_t1ce.nii');
% data_flair = niftiread(filename_flair);
% data_t1ce = niftiread(filename_t1ce);

data_t1 = niftiread(filename_t1);
data_seg = niftiread(filename_seg);

data_main = data_t1;
data_structure = data_seg;

% create volumes
emission_structure = Volume(data_structure);
emission_main = Volume(data_main);

mask_margin=5;
mask_content_size = size(emission_main.Data)-2*mask_margin;
half_size=emission_main.size()/2;

mask = Volume(data_main);
mask.resize(mask_content_size);
mask.pad(mask_margin, 0);
mask.Data(mask.Data>0.1) = 1;
mask.Data(mask.Data<=0.1) = 0;
mask.Data(half_size(2):end, 1:end, 1:end) = 1;
emission_main.Data = emission_main.Data;

sw = Stopwatch('timings');


%% setup for rendering main channel

% total frames of the movie
total_frames=240;

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
render.Shininess = 30;
render.ScatteringWeight = 0.5;
render.Color = [1,1,1];

render.FactorEmission=1;
render.FactorAbsorption=1;
render.FactorReflection=1;

render.FocalLength=4.5;
render.DistanceToObject = 4;
render.OpacityThreshold=0.95;

% rotate
render.rotate(-90,270,0);
render.rotate(-15,15,15);

% setup illumintation
render.LightSources = LightSource([-15,15,0], [1,1,1]);

% setup volumes
data_absorption = data_main;
absorptionVolume=emission_main;
% absorptionVolume=Volume(data_absorption);
% half_size=size(data_absorption)/2;
% absorptionVolume.resize(half_size);
% absorptionVolume.normalize(0,1);

render.VolumeAbsorption=absorptionVolume;
render.VolumeEmission=emission_main;

render.VolumeScattering=Volume(HenyeyGreenstein_MATLAB(512));

% setup image size (of the resulting 2D image)
render.ImageResolution=size(emission_main.Data,[1 2]);

% rotation per step
beta = rotation_of_movie/total_frames;


% uncomment this line in order to get a stereo image (anaglyph)
stereo = false;


if stereo
    render.StereoOutput = StereoRenderMode.RedCyan;
    render.CameraXOffset = 0.06;
end

%% start renderung main channel
% disp('rendering main channel');

sw.add('1', 'main channel #1');

start_frame=1;
end_frame=total_frames/8;
for i=start_frame:end_frame
    % disp(strcat(' -image ', num2str(i)));

    sw.start('1');
    rendered_image = render.render();
    sw.stop('1');

    % rotate object
    render.rotate(0,beta,0);
    
    rendered_images_main(:,:,:,i) = rendered_image;
end

% fade out half of the volume
start_frame=(total_frames/8)+1;
end_frame=total_frames-(total_frames/8);

% compute factor for dimming
n = end_frame-start_frame+1;
factor = linspace(1,0.2,n);

sw.add('2', 'main channel #2');

for i=start_frame:end_frame
    % disp(strcat(' -image ', num2str(i)));
    % data_main =  factor(i-start_frame+1) * mask.Data .* single(data_main);

    % emissionVolume=Volume(factor(i-start_frame+1) * emission_main.Data(1:end, half_size(2):end, 1:end));

    render.VolumeEmission.Data(logical(mask.Data))=factor(i-start_frame+1) * emission_main.Data(logical(mask.Data));
    
    sw.start('2');
    rendered_image = render.render();
    sw.stop('2');

    % rotate object
    render.rotate(0,beta,0);

    rendered_images_main(:,:,:,i) = rendered_image;
end

sw.add('3', 'main channel #3');

start_frame=end_frame+1;
end_frame=total_frames;
for i=start_frame:end_frame
    % disp(strcat(' -image ', num2str(i)));
    
    sw.start('3');
    rendered_image = render.render();
    sw.stop('3');

    % rotate object
    render.rotate(0,beta,0);

    rendered_images_main(:,:,:,i) = rendered_image;
end


%% setup for rendering structure image channel
% create empty structure for all rendered images
rendered_images_structure = zeros([  size(emission_structure.Data,2), ...
                                size(emission_structure.Data,1), ...
                                3, ...
                                total_frames]);
                            
% disp('rendering structure channel');

% reset render
render.RotationMatrix = eye(3);
render.rotate(-90,270,0);
render.rotate(-15,15,15);

% setup render
render.ImageResolution=size(emission_structure.Data,[1 2]);


render.VolumeAbsorption=emission_structure;
render.VolumeEmission=emission_structure;

render.FactorEmission=1;
render.FactorAbsorption=2;
render.FactorReflection=1;

render.Color = [0,1,0];

%% render structure image channel

sw.add('s', 'structure channel');

start_frame=1;
end_frame=total_frames;
for i=start_frame:end_frame
    % disp(strcat(' -image ', num2str(i)));

    sw.start('s');
    rendered_image = render.render();
    sw.stop('s');

    % rotate object
    render.rotate(0,beta,0);
    
    rendered_images_structure(:,:,:,i) = rendered_image;
end


%% create one movie
sw.print();

rendered_images_combined = rendered_images_main+rendered_images_structure;

normalized_images = VolumeRender.normalizeSequence(rendered_images_combined);

mov = immovie(normalized_images);
implay(mov, 15);

% save to file
v = VideoWriter("brain", 'MPEG-4');
v.FrameRate=15;
v.Quality = 100;
open(v);
writeVideo(v,normalized_images);
close(v);


%% save 360 degree image excerpt
frame_start=15;
n=8;
rel_i=int8(linspace(0,360,8)/5);

for i=1:1:8
    filename=fullfile(workingpath,sprintf('../png/rotation_brain_%d.png', i));

    img_inv = imcomplement(normalized_images(:,:,:,frame_start+rel_i(i)));
    imwrite(img_inv, filename);
end
