classdef VolumeRender < handle
    % For explanation see documentation (pdf)
    properties
        FocalLength=0.0;            % 
        DistanceToObject=0.0;       % 
        OpacityThreshold=0.95;      % 
        
        ElementSizeUm=[1,1,1];      % 
        
        LightSources=false;         % 
        
        Color=[1,1,1];              % 
        
        VolumeEmission=false;       % 
        VolumeReflection=Volume(1);         % 
        VolumeAbsorption=false;     % 
        VolumeGradientX=false;      % 
        VolumeGradientY=false;      % 
        VolumeGradientZ=false;      % 
        VolumeIllumination=false;   % 
 
        ScaleEmission=1.0;          % 
        ScaleReflection=1.0;        % 
        ScaleAbsorption=1.0;        % 
        
        CameraXOffset=0;
        StereoOutput=StereoRenderMode.RedCyan;
        
        RotationMatrix = eye(3);   % 
        ImageResolution = [0,0];    % 
    end
    methods        
        % construct
        function obj = VolumeRender()
        end

        function rotate(obj, alpha, beta, gamma)
            % rotation of the viewmatrix
            % alpha     rotation around col
            % gamma     rotation adound lev
            % beta      rotation around row
            
            RotationX = [1,0,0;
                         0, cosd(alpha), -sind(alpha);
                         0, sind(alpha), cosd(alpha)];
            RotationY = [cosd(beta),0,sind(beta);
                         0,1,0;
                         -sind(beta),0,cosd(beta)];
            RotationZ = [cosd(gamma), -sind(gamma), 0;
                         sind(gamma), cosd(gamma), 0;
                         0,0,1];

            obj.RotationMatrix = obj.RotationMatrix * ...
                                 RotationX * RotationY * RotationZ;
        end
        
        function image = render(obj)
            % rendering image. If CmaeraXOffset does not equal 0
            % a 3D anaglyph will be returned
            % image     output (3D) image
            
            if (obj.CameraXOffset==0)
                image=prender(obj, single(obj.CameraXOffset), flip(obj.ImageResolution));
            else
                % render 3D (combine the images)
                base = obj.CameraXOffset/2;
                
                
                %angleFieldOfview=2*atan(obj.ImageResolution(2)/(2*obj.FocalLength));
                angleFieldOfview=2*atan(1/obj.FocalLength);
                delta = round((base * obj.ImageResolution(2)) / ...
                        (2*obj.FocalLength * tan(angleFieldOfview/2)));

                % render more pixel
                resolution=flip(obj.ImageResolution) + [0, delta];
                
                rightImage=prender(obj, base, resolution);
                leftImage=prender(obj, -base, resolution);
                
                % combine the 2 images, crop delta
                rect=[(delta+1) 0 size(leftImage,2) size(leftImage,1)];
                leftImage=imcrop(leftImage, rect);
                
                rect=[0 0 (size(rightImage,2)-delta) size(rightImage,1)];
                rightImage=imcrop(rightImage, rect);
                
                if( strcmp(obj.StereoOutput,StereoRenderMode.RedCyan) )
                    % RGB - anaglyph
                    image=zeros([size(leftImage,1), size(leftImage,2), 3]);
                    image(:,:,1) = leftImage(:,:,1);
                    image(:,:,2) = rightImage(:,:,2);
                    image(:,:,3) = rightImage(:,:,3);
                elseif( strcmp(obj.StereoOutput, StereoRenderMode.LeftRightHorizontal) )
                    image = [leftImage, rightImage];
                end
                    
            end
        end
    end
       
    % set methods
    methods
        function set.ElementSizeUm(obj, val)
            if (isequal(size(val), [1,3]))
                obj.ElementSizeUm = val;
            elseif (isequal(size(val), [3,1]))
                obj.ElementSizeUm = val'; % transpose it
            else
                error('dimensions of ImageResolution must be [1,3]');
            end
        end
        
        function set.Color(obj, val)
            if (isequal(size(val), [1,3]))
                obj.Color = val;
            else
                error('dimensions of ImageResolution must be [1,3] and values between 0 and 1');
            end
        end
        
        function set.LightSources(obj, val)
            % ensure correct dimension
            validate = [size(val,1) == 1, ...
                        ismatrix(val), ...
                        ndims(val) == 2, ...
                        isa(val, 'LightSource')
                        ];
            if (all(validate))
                obj.LightSources = val;
            else
                error('LightSources must be a 1xN vector with data of type LightSource!');
            end
        end
        
        function set.VolumeIllumination(obj, val)
            if (isa(val,'Volume'))
                obj.VolumeIllumination = val;
            else
                error('VolumeEmission must be of type Volume');
            end
        end
        
        function set.VolumeEmission(obj, val)
            if (isa(val,'Volume'))
                obj.VolumeEmission = val;
            else
                error('VolumeEmission must be of type Volume');
            end
        end
        
        function set.VolumeReflection(obj, val)
            if (isa(val,'Volume'))
                obj.VolumeReflection = val;
            else
                error('VolumeReflection must be of type Volume');
            end
        end
        
        function set.VolumeGradientX(obj, val)
            if (isa(val,'Volume'))
                obj.VolumeGradientX = val;
            else
                error('VolumeReflection must be of type Volume');
            end
        end
        function set.VolumeGradientY(obj, val)
            if (isa(val,'Volume'))
                obj.VolumeGradientY = val;
            else
                error('VolumeReflection must be of type Volume');
            end
        end
        function set.VolumeGradientZ(obj, val)
            if (is(val,'Volume'))
                obj.VolumeGradientZ = val;
            else
                error('VolumeReflection must be of type Volume');
            end
        end
        
        function set.VolumeAbsorption(obj, val)
            if (isa(val,'Volume'))
                minVal = min(min(min(val.Data)));
                if minVal < 0
                    warning('VolumeAbsorption is not allowed to contain data smaller than 0!');
                end
                obj.VolumeAbsorption = val;
            else
                error('VolumeAbsorption must be of type Volume');
            end
        end
        
        function set.FocalLength(obj, val)
            if (isequal(size(val), [1,1]))
                obj.FocalLength = val;
            else
                error('dimensions of ImageResolution must be a scalar');
            end
        end
        
        function set.DistanceToObject(obj, val)
            if (isequal(size(val), [1,1]))
                obj.DistanceToObject = val;
            else
                error('dimensions of ImageResolution must be a scalar');
            end
        end
        
        function set.OpacityThreshold(obj, val)
            if (isequal(size(val), [1,1]))
                obj.OpacityThreshold = val;
            else
                error('dimensions of ImageResolution must be a scalar');
            end
        end
        
        function set.CameraXOffset(obj, val)
            if (isequal(size(val), [1,1]))
                obj.CameraXOffset = val;
            else
                error('dimensions of ImageResolution must be a scalar');
            end
        end
        
        function set.StereoOutput(obj, val)
            if (strcmp(val,'red-cyan'))
                obj.StereoOutput = val;
            elseif (strcmp(val,'left-right-horizontal'))
                obj.StereoOutput = val;
            else
                error('allowed values are "red-cyan", "left-right-horizontal"');
            end
        end
        
        function set.RotationMatrix(obj, val)
            if (isequal(size(val), [3,3]))
                    obj.RotationMatrix = val;
            else
                error('dimensions of RotationMatrix must be [3,3]');
            end
        end
        
        function set.ImageResolution(obj, val)
            if (isequal(size(val), [1,2]))
                obj.ImageResolution = val;
            else
                error('dimensions of ImageResolution must be a [1,2]');
            end
        end
        
        function set.ScaleEmission(obj, val)
             if (isequal(size(val), [1,1]))
                 obj.ScaleEmission = val;
             else
                 error('dimensions of ImageResolution must be a scalar');
             end
        end
         
        function set.ScaleReflection(obj, val)
             if (isequal(size(val), [1,1]))
                 obj.ScaleReflection = val;
             else
                 error('dimensions of ImageResolution must be a scalar');
             end
        end
         
        function set.ScaleAbsorption(obj, val)
             if (isequal(size(val), [1,1]))
                 obj.ScaleAbsorption = val;
             else
                 error('dimensions of ImageResolution must be a scalar');
             end
        end
    end % methods

    methods(Access = protected)
       function image = prender(obj, CameraXOffset, resolution)
            % rendering image on GPU
            % CameraXOffset     offset between 2 cameras
            % resolution        image resolution
            % image             rendered image

            % check if all volumes are correctly set
            validate=[islogical(obj.VolumeReflection), ...
                      islogical(obj.VolumeAbsorption), ...
                      islogical(obj.VolumeEmission)];
                  
            if (islogical(obj.VolumeIllumination))
                warning('VolumeIllumination is unset. Thus no lightning will be applied!');
            end
                
            if (all(validate))
                display(validate);
                error('Not all volumes are properly set!');
            end
                  
            scales=[obj.ScaleEmission, obj.ScaleReflection, obj.ScaleAbsorption];
            props=[CameraXOffset, obj.FocalLength, obj.DistanceToObject];

            validate =  [islogical(obj.VolumeGradientX), ...
                         islogical(obj.VolumeGradientY), ...
                         islogical(obj.VolumeGradientZ)];

            matrix=flip(obj.RotationMatrix);

            if (all(not(validate)))
                image = volumeRender(obj.LightSources, single(obj.VolumeEmission.Data), ...
                               single(obj.VolumeReflection.Data), ...
                               single(obj.VolumeAbsorption.Data), ...
                               single(obj.VolumeIllumination.Data), single(scales), ...
                               single(obj.ElementSizeUm), uint64(resolution), ...
                               single(matrix), single(props), ...
                               single(obj.OpacityThreshold), single(obj.Color), ...
                               single(obj.VolumeGradientX.Data), ...
                               single(obj.VolumeGradientY.Data), ...
                               single(obj.VolumeGradientZ.Data));
            else
                image = volumeRender(obj.LightSources, single(obj.VolumeEmission.Data), ...
                               single(obj.VolumeReflection.Data), ...
                               single(obj.VolumeAbsorption.Data), ...
                               single(obj.VolumeIllumination.Data), single(scales), ...
                               single(obj.ElementSizeUm), uint64(resolution), ...
                               single(matrix), single(props), ...
                               single(obj.OpacityThreshold), single(obj.Color));
            end
       end
    end
    
    methods(Access = public,Static = true)
        function normalized=normalizeSequence(sequence)
            % Normalizing a multiframe image (4D image) between [0,1]
            % sequence      multiframe image/image sequence
            % normalized    normalized multiframe image
            
            if (ndims(sequence) < 4)
                error('input must be a multiframe image (4D)');
            end
            
            % get min/max scalar
            maxValue=max(sequence(:));
            minValue=min(sequence(:));
            
            normalized=zeros(size(sequence));
            for i=1:size(sequence,4)
                tmp=sequence(:,:,:,i);
                normalized(:,:,:,i) = ...
                    VolumeRender.normalizeImage(tmp, minValue, maxValue);
            end
        end
        
        function img=normalizeImage(ImageRGB, varargin)
            % Normalizing RGB image between [0,1]
            % ImageRGB      input image
            % img           output image
            % varargin      manually chosen minValue, maxValue

            red = ImageRGB(:,:,1);
            green = ImageRGB(:,:,2);
            blue = ImageRGB(:,:,3);

            if (nargin < 2)
                minValue = min([min(red(:)), min(green(:)), min(blue(:))]);
            else
                minValue = varargin{1};
            end
            
            if (nargin < 3)
                maxValue = max([max(red(:)), max(green(:)), max(blue(:))]);
            else
                maxValue = varargin{2};
            end
            
            if (minValue < 0)
                red = red + minValue;
                green = green + minValue;
                blue = blue + minValue;
                maxValue = maxValue + abs(minValue);
            end
            
            tmp = zeros(size(ImageRGB));
            tmp(:,:,1) = red/maxValue;
            tmp(:,:,2) = green/maxValue;
            tmp(:,:,3) = blue/maxValue;
            
            img = tmp;
        end
    end
end % classdef
