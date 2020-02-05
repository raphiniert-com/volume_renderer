% some tests for the matlab-C++ connection



% l = [LightSource([1,2,3],[3,3,4]), LightSource([0,1,9],[13,4,6])];

% filename ='/home/raphael/volumes/zebrafish/Tuan/material_and_methods/301_stitched.h5'; dataset = '/t0/channel0';
% img = hdf5read( filename, dataset);
% elementSizeUm = hdf5read( filename, strcat(dataset,'/element_size_um'));
% img=single(img);
% img=sqrt(img);

% connect(l, LightSource([1,2,3],[2,3,4]));

% read raw-files
raw_folder='/home/raphael/volumes/raw2/';
resemission = [356, 257, 48];
resabsorption = [356, 257, 48];
femission=fopen(strcat(raw_folder,'/emission_0.raw'), 'r');
fabsorption=fopen(strcat(raw_folder,'/emission_0.raw'), 'r');

emission=fread(femission, 'single');
emission=Volume(reshape(emission, resemission));
absorption=fread(fabsorption, 'single');
% TODO: normalize [0,1] absorption=absorption;
absorption=Volume(reshape(absorption, resabsorption));

fclose(femission);
fclose(fabsorption);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

elementSizeUm = [1.3;0.4972;0.4972];

% emission=img;
reflection=Volume([1,1,1;1,1,1;1,1,1]);
% absorption=img;

%r=size(img)/2;
%v.resize(r(1),r(2),r(3));

%central_z = round(size(v.Data,3)/2);
%imshow(v.Data(:,:,central_z)',[]);

render = VolumeRender();
render.VolumeEmission=emission;
render.VolumeAbsorption=absorption;
render.VolumeReflection=reflection;

render.VolumeIllumination=Volume(HG(64));

render.ImageResolution=[size(emission.Data,2), size(emission.Data,1)];

render.LightSources = [LightSource([0,0,1], [0.5,0,0]), ...
                       LightSource([0,0,-1], [0,0.5,0]), ...
                       LightSource([-1,0,0], [0,0,0.5])];

render.ElementSizeUm=elementSizeUm;

render.CameraXOffset=0.1;

render.FocalLength=3.0;

render.Color = [1,1,1];

render.DistanceToObject = 10;

render.rotate(0,270,0);
rendered_image = zeros([size(emission.Data,2),size(emission.Data,1), 3, 360]);

nStep=30;
beta = 360/nStep;
for i=1:nStep
    display(strcat('image ', num2str(i)));
    alpha = 0;
    gamma = 0;
    RotationX = [1,0,0;
                 0, cosd(alpha), -sin(alpha);
                 0, sind(alpha), cosd(alpha)];
    RotationY = [cosd(beta),0,sind(beta);
                 0,1,0;
                 -sind(beta),0,cosd(beta)];
    RotationZ = [cosd(gamma), -sind(gamma), 0;
                 sind(gamma), cosd(gamma), 0;
                 0,0,1];
    
    render.rotate(0,beta,0);

    render.LightSources(3).Position = RotationY * render.LightSources(3).Position';
    rendered_image(:,:,:,i) = render.render();
end

normalizedImages = VolumeRender.normalizeSequence(rendered_image);

mov = immovie(normalizedImages);
movie(mov);

% all(all(all(normalizedImages(:,:,:,1) ==  normalizedImages(:,:,:,10))))
% imshow(VolumeRender.normalizeImage(rendered_image));