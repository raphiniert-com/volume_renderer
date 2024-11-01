% This example uses one channel and one LightSource to render a movie.

%% init
% estimate working path, so that the script runs from any location
workingpath = erase(mfilename('fullpath'), 'example2');

% add VolumeRender to path
addpath(fullfile(workingpath, '..', 'src', 'matlab', 'VolumeRender'));

% folder to the volume files
path=fullfile(workingpath, 'h5-data');

filename = fullfile(path, 'ViBE-Z_72hpf_v1.h5');
dataset = '/anatomy/average_brain';

data_main = h5read(filename, dataset);
elementSizeUm = h5readatt(filename, dataset, 'element_size_um');

%% setup renderer
% create render object
render = VolumeRender();

render.ElementSizeUm=elementSizeUm;

% setup illumination settings
render.VolumePhase = Volume(HenyeyGreenstein_LUT(64));

render.LightSources = LightSource([15,15,0], [1,1,1], LightType.Spotlight);
render.ScatteringWeight = 0.8;
render.FocalLength = 3.0;
render.DistanceToObject = 6.0;
render.rotate(90,0,0);

% rotate some more
render.rotate(-15,-15,15);

% create volumes
emission_main = Volume(data_main);

% setup image size (of the resulting 2D image)
render.ImageResolution= ...
    size(emission_main.Data,[1 2]);

% set render volumes
render.VolumeEmission = emission_main;
render.VolumeAbsorption = emission_main;
render.FactorReflection = 1;
render.FactorEmission = 1;

render.Color = [1,1,1];

%% perform rendering of video sequence
nStep=30;
rendered_image = zeros([size(emission_main.Data,2), ...
                        size(emission_main.Data,1), ...
                        3, 360/nStep]);

beta = 360/nStep;
for i=1:nStep
    disp(strcat('image ', num2str(i)));

    % rotate object
    render.rotate(0,beta,0);

    rendered_image(:,:,:,i) = render.render();
end

normalizedImages = VolumeRender.normalizeSequence(rendered_image);


%% create video and play it
mov = immovie(normalizedImages);
implay(mov);