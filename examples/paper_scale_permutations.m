% This example uses two channels and two light-sources to render one static image

%% init
% estimate working path, so that the script runs from any location
workingpath = erase(mfilename('fullpath'), 'paper_scale_permutations');

% add VolumeRender to path
addpath(fullfile(workingpath, '..', 'src', 'matlab', 'VolumeRender'));
addpath(fullfile(workingpath, '..', 'src', 'matlab', 'Stopwatch'));

sw = Stopwatch('Movie generation');

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

% setup illumination settings
render.VolumeIllumination=Volume(HenyeyGreenstein(64));
render.LightSources = [LightSource([0,5,0],[0.5,0.5,0.5])];

% misc parameters
render.ElementSizeUm=elementSizeUm;
render.FocalLength=3.0;
render.DistanceToObject=6;

render.rotate(45,25,45);

render.OpacityThreshold=0.9;


% setup image size (of the resulting 2D image)
render.ImageResolution=size(emission_structure.Data,[1, 2]);

% create empty structure for all rendered images
rendered_images_main= zeros([  size(emission_structure.Data,2), ...
                                size(emission_structure.Data,1), ...
                                3, ...
                                10,10,10]);

absorptionVolume=Volume(data_main);
absorptionVolume.resize(0.5);
absorptionVolume.normalize(0,1);

render.VolumeEmission=emission_main;
render.VolumeAbsorption=Volume(1);


% make it kind of transparent
render.FactorAbsorption=0.5;
render.FactorEmission=0.5;
render.FactorReflection=0.5;


image = render.render();

figure;
imshow(image);

sw.add('rt', 'render time');

stepsize=2;
%% main zebra fish
for r = 0:stepsize:10
    for a = 0:stepsize:10
        for e = 0:stepsize:10
            % make it kind of transparent
            render.FactorAbsorption=a*0.1;
            render.FactorEmission=e*0.1;
            render.FactorReflection=r*0.1;
        
            
            render.Color = [1,1,1];
            
            sw.start('rt');
            rendered_image = render.render();
            sw.stop('rt');

            rendered_images_main(:,:,:,r+1,a+1,e+1) = rendered_image;

            %figure;
            %imshow(rendered_image);
        end
    end
end

sw.print();

%% display the images and the combined one
x=10/stepsize + 1;
y=10/stepsize + 1;

img={};

targetSize = [400 400];
win1 = centerCropWindow2d(size(rendered_image),targetSize);

for e = 0:stepsize:10
    for r = 0:stepsize:10
        for a = 0:stepsize:10
        
            k = (r/stepsize) * x + (a/stepsize);

            current_img=imcrop(rendered_images_main(:,:,:,e+1, r+1, a+1), win1);
            filename=fullfile(workingpath,sprintf('../png/%d_%d.png', e, k));
            
            img_inv = imcomplement(current_img);
            img = [img; img_inv];

            imwrite(img_inv, filename);

            fid=fopen(fullfile(workingpath,sprintf('../png/%d_%d.json', e, k)),'w');
            s = struct("scale_reflection", round(r*0.1,2), "scale_absorption", round(a*0.1,2), "scale_emission", round(e*0.1,2)); 
            encodedJSON = jsonencode(s); 
            fprintf(fid, encodedJSON);
        end
    end
end

fclose('all');


%% crop
fontSizeLabelAxis=30;
fontSizeLabel=25;
for i = 1:6
    figure;
    collage_img=montage(img((i-1)*36+1:36*i));
    filename=fullfile(workingpath,sprintf('../png/collage_%d.png', (i-1)*stepsize));
    %caption =  sprintf("cebrafish embrio with scale emission: %0.1f", (i-1) * 0.2);
    %title(caption, 'FontSize', 10);

    ax = gca;
    ax.FontSize = fontSizeLabel;

    axis('on', 'image');
    xticks([targetSize(2)/4:targetSize(2)/2:10*targetSize(2)])
    xticklabels({'0','0.2','0.4','0.6','0.8','1.0'})

    yticks([targetSize(1)/4:targetSize(1)/2:10*targetSize(1)])
    yticklabels({'0','0.2','0.4','0.6','0.8','1.0'})

    xlabel('scale absorption', 'FontSize', fontSizeLabelAxis);
    ylabel('scale reflection', 'FontSize', fontSizeLabelAxis);

    

    % make sure that the x and ylabels are in the picture
    scale = 0.06;
    pos = get(gca, 'Position');
    pos(2) = pos(2)+scale*pos(4);
    pos(4) = (1-scale)*pos(4);
    set(gca, 'Position', pos)

    hIm = findall(collage_img,'type','image');
    image = get(hIm,'CData');

    saveas(gcf, filename);

    % imwrite(image, filename);

    
    % imshow(collage_img.CData, [], 'XData', [0, 0.2, 0.4, 0.6, 0.8, 1], 'YData', [0, 0.2, 0.4, 0.6, 0.8, 1]);
    % axis('on', 'image');
end

%%
montage(img((i-1)*36+1:36*i));

% imshow(collage_img.CData);
axis('on', 'image');
xticks([targetSize(2)/4:targetSize(2)/2:10*targetSize(2)])
xticklabels({'0','0.2','0.4','0.6','0.8','1.0'})
xlabel('');

yticks([targetSize(1)/4:targetSize(1)/2:10*targetSize(1)])
yticklabels({'0','0.2','0.4','0.6','0.8','1.0'})
ylabel('');

ax = gca;
ax.FontSize = 20;

