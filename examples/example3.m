% This example uses two channels and one LightSource to render a movie.
% Furthermore, let half of the main volume diffuse

%% init
% estimate working path, so that the script runs from any location
workingpath = erase(mfilename('fullpath'), 'example3');

% add VolumeRender to path
addpath(fullfile(workingpath, '..', 'src', 'matlab', 'VolumeRender'));
addpath(fullfile(workingpath, '..', 'src', 'matlab', 'StopWatch'));

% folder to the volume files
path=fullfile(workingpath, 'h5-data');

filename = fullfile(path, 'ViBE-Z_72hpf_v1.h5');

dataset = '/anatomy/average_brain';
data_main = h5read(filename, dataset);

elementSizeUm = h5readatt(filename, dataset, 'element_size_um');

dataset = '/expression/3A10';
data_structure = h5read(filename, dataset);

% create volumes
emission_structure = Volume(data_structure);
emission_main = Volume(data_main);

mask_margin=10;
mask_content_size = size(emission_main.Data)-2*mask_margin;
half_size=emission_main.size()/2;

mask = Volume(data_main);
mask.resize(mask_content_size);
mask.pad(mask_margin, 0);
mask.Data(mask.Data>0.1) = 1;
mask.Data(mask.Data<=0.1) = 0;
mask.Data(1:end, half_size(2):end, 1:end) = 1;
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
render.Color = [1,1,1];

render.ElementSizeUm=elementSizeUm;
render.FocalLength=4.5;
render.DistanceToObject = 6;
render.OpacityThreshold=0.95;

% rotate
render.rotate(90,0,0);
render.rotate(-15,15,15);

% setup illumintation
render.LightSources = LightSource([-15,15,0], [0.5,0.5,0.5]);

% setup volumes
data_absorption = data_main;
absorptionVolume=emission_main;
% absorptionVolume=Volume(data_absorption);
% half_size=size(data_absorption)/2;
% absorptionVolume.resize(half_size);
% absorptionVolume.normalize(0,1);

render.VolumeAbsorption=absorptionVolume;
render.VolumeEmission=emission_main;

render.VolumeIllumination=Volume(HenyeyGreenstein(64));

% setup image size (of the resulting 2D image)
render.ImageResolution=size(emission_main.Data,[1 2]);

% rotation per step
beta = rotation_of_movie/total_frames;

render.FactorEmission=1;
render.FactorAbsorption=1;
render.FactorReflection=1;

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

%     imshow(imcomplement(rendered_image));
%     return;
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
render.rotate(90,0,0);
render.rotate(-15,15,15);

% setup render
render.ImageResolution=size(emission_structure.Data,[1 2]);


render.VolumeAbsorption=emission_structure;
render.VolumeEmission=emission_structure;

render.Color = [0,1,0];

%% render structure image channel

render.FactorEmission=0.5;
render.FactorAbsorption=1;
render.FactorReflection=1;

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

%     imshow(imcomplement(rendered_image));
% 
%     rendered_images_combined = rendered_images_main+rendered_images_structure;
% 
%     imshow(imcomplement(rendered_images_combined(:,:,:,1)))
% 
%     return;
end


%% create one movie
sw.print();

rendered_images_combined = rendered_images_main+rendered_images_structure;

amplification_factor=3;

% multiply the values in order to make it better visible after inverting
normalized_images = sqrt(VolumeRender.normalizeSequence(rendered_images_combined));

mov = immovie(imcomplement(normalized_images));
implay(mov, 15);

% save to file
v = VideoWriter("zebrafish", 'MPEG-4');
v.FrameRate=15;
v.Quality = 100;
open(v);
writeVideo(v,imcomplement(normalized_images/max(normalized_images(:))));
close(v);


%% save 360 degree image excerpt
frame_start=15;
n=8;
rel_i=int8(linspace(0,360,8)/5);

for i=1:1:8
    filename=fullfile(workingpath,sprintf('../png/rotation_%d.png', i));

    % multiply the values in order to make it better visible after inverting
    img_inv = imcomplement(normalized_images(:,:,:,frame_start+rel_i(i)));
    imwrite(img_inv, filename);
end
