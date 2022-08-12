classdef VolumeRender < handle
    % For explanation see documentation (pdf)
    properties
        FocalLength(1,1)      = 0.0;
        DistanceToObject(1,1) = 0.0;
        OpacityThreshold(1,1) = 0.95;
        
        LightSources(1,:)     = false;
        
        Color(1,3)            = [1,1,1];
 
        ScaleEmission(1,1)    = 1.0;
        ScaleReflection(1,1)  = 1.0;
        ScaleAbsorption(1,1)  = 1.0;
        
        CameraXOffset(1,1)    = 0;
        StereoOutput StereoRenderMode = StereoRenderMode.RedCyan;
        
        ElementSizeUm(1,3)    = [1,1,1];
        RotationMatrix(3,3)   = eye(3);
        ImageResolution(1,2)  = [0,0];
        
        VolumeReflection      = Volume(1);
        
        VolumeEmission        = false;
        VolumeAbsorption      = false;
        VolumeGradientX       = false;
        VolumeGradientY       = false;
        VolumeGradientZ       = false;
        VolumeIllumination    = false;

        TimeLastMemSync = uint64(0);
    end

    properties (SetAccess = private, Hidden = true)
        objectHandle; % Handle to the underlying C++ memory manager instance
        
    end
    methods        
        % Constructor
        function this = VolumeRender(varargin)
            this.objectHandle = volumeRender('new', varargin{:});
        end

        % Destructor
        function delete(this)
            volumeRender('delete', this.objectHandle);
        end

        function syncVolumes(this)
            validate =  [islogical(this.VolumeGradientX), ...
                         islogical(this.VolumeGradientY), ...
                         islogical(this.VolumeGradientZ)];

            if (all(not(validate)))
                volumeRender(  'sync_volumes', this.objectHandle, ...
                               this.TimeLastMemSync, ...
                               this.VolumeEmission, ...
                               this.VolumeReflection, ...
                               this.VolumeAbsorption, ...
                               this.VolumeGradientX, ...
                               this.VolumeGradientY, ...
                               this.VolumeGradientZ);
            else
                volumeRender(  'sync_volumes', this.objectHandle, ...
                               this.TimeLastMemSync, ...
                               this.VolumeEmission, ...
                               this.VolumeReflection, ...
                               this.VolumeAbsorption);
            end

            % save time of render process
            this.TimeLastMemSync=timestamp;
        end

        function memInfo(this)
            volumeRender('mem_info', this.objectHandle);
        end

        function memClear(this)
            volumeRender('delete', this.objectHandle);
        end

        function rotate(this, alpha, beta, gamma)
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

            this.RotationMatrix = this.RotationMatrix * ...
                                 RotationX * RotationY * RotationZ;
        end
        
        function image = render(this)
            % rendering image. If CmaeraXOffset does not equal 0
            % a 3D anaglyph will be returned
            % image     output (3D) image
            
            if (this.CameraXOffset==0)
                image=p_render(this, single(this.CameraXOffset), flip(this.ImageResolution));
            else
                % render 3D (combine the images)
                base = this.CameraXOffset/2;
                
                
                %angleFieldOfview=2*atan(this.ImageResolution(2)/(2*this.FocalLength));
                angleFieldOfview=2*atan(1/this.FocalLength);
                delta = round((base * this.ImageResolution(2)) / ...
                        (2*this.FocalLength * tan(angleFieldOfview/2)));

                % render more pixel
                resolution=flip(this.ImageResolution) + [0, delta];
                
                rightImage=p_render(this, base, resolution);
                leftImage=p_render(this, -base, resolution);
                
                % combine the 2 images, crop delta
                rect=[(delta+1) 0 size(leftImage,2) size(leftImage,1)];
                leftImage=imcrop(leftImage, rect);
                
                rect=[0 0 (size(rightImage,2)-delta) size(rightImage,1)];
                rightImage=imcrop(rightImage, rect);
                
                if (strcmp(this.StereoOutput,StereoRenderMode.RedCyan))
                    % RGB - anaglyph
                    image=zeros([size(leftImage,1), size(leftImage,2), 3]);
                    image(:,:,1) = leftImage(:,:,1);
                    image(:,:,2) = rightImage(:,:,2);
                    image(:,:,3) = rightImage(:,:,3);
                elseif( strcmp(this.StereoOutput, StereoRenderMode.LeftRightHorizontal) )
                    image = [leftImage, rightImage];
                end
                    
            end
        end
    end
       
    % set methods
    methods
        function set.LightSources(this, val)
            % ensure correct type
            if (all(isa(val, 'LightSource')))
                this.LightSources = val;
            else
                error('LightSources must be a 1xN vector with data of type LightSource!');
            end
        end
        
        function set.VolumeIllumination(this, val)
            if (isa(val,'Volume'))
                this.VolumeIllumination = val;
            else
                error('VolumeEmission must be of type Volume');
            end
        end
        
        function set.VolumeEmission(this, val)
            if (isa(val,'Volume'))
                this.VolumeEmission = val;
            else
                error('VolumeEmission must be of type Volume');
            end
        end
        
        function set.VolumeReflection(this, val)
            if (isa(val,'Volume'))
                this.VolumeReflection = val;
            else
                error('VolumeReflection must be of type Volume');
            end
        end
        
        function set.VolumeGradientX(this, val)
            if (isa(val,'Volume'))
                this.VolumeGradientX = val;
            else
                error('VolumeReflection must be of type Volume');
            end
        end
        function set.VolumeGradientY(this, val)
            if (isa(val,'Volume'))
                this.VolumeGradientY = val;
            else
                error('VolumeReflection must be of type Volume');
            end
        end
        function set.VolumeGradientZ(this, val)
            if (is(val,'Volume'))
                this.VolumeGradientZ = val;
            else
                error('VolumeReflection must be of type Volume');
            end
        end
        
        function set.VolumeAbsorption(this, val)
            if (isa(val,'Volume'))
                minVal = min(min(min(val.Data)));
                if minVal < 0
                    warning('VolumeAbsorption is not allowed to contain data smaller than 0!');
                end
                this.VolumeAbsorption = val;
            else
                error('VolumeAbsorption must be of type Volume');
            end
        end
        
        function set.StereoOutput(this, val)
            if (strcmp(val,'red-cyan'))
                this.StereoOutput = val;
            elseif (strcmp(val,'left-right-horizontal'))
                this.StereoOutput = val;
            else
                error('allowed values are "red-cyan", "left-right-horizontal"');
            end
        end
    end % methods

    methods(Access = protected)
       function image = p_render(this, CameraXOffset, resolution)
            % rendering image on GPU
            % CameraXOffset     offset between 2 cameras
            % resolution        image resolution
            % image             rendered image

            % check if all volumes are correctly set
            validate=[islogical(this.VolumeReflection), ...
                      islogical(this.VolumeAbsorption), ...
                      islogical(this.VolumeEmission)];
                  
            if (islogical(this.VolumeIllumination))
                warning('VolumeIllumination is unset. Thus no lightning will be applied!');
            end
                
            if (all(validate))
                display(validate);
                error('Not all volumes are properly set!');
            end
                  
            scales=[this.ScaleEmission, this.ScaleReflection, this.ScaleAbsorption];
            props=[CameraXOffset, this.FocalLength, this.DistanceToObject];

            validate =  [islogical(this.VolumeGradientX), ...
                         islogical(this.VolumeGradientY), ...
                         islogical(this.VolumeGradientZ)];

            matrix=flip(this.RotationMatrix);

            if (all(not(validate)))
                image = volumeRender('render', this.objectHandle, ...
                               this.LightSources, ...
                               this.VolumeEmission, ...
                               this.VolumeReflection, ...
                               this.VolumeAbsorption, ...
                               this.VolumeIllumination, single(scales), ...
                               single(this.ElementSizeUm), uint64(resolution), ...
                               single(matrix), single(props), ...
                               single(this.OpacityThreshold), single(this.Color), ...
                               this.VolumeGradientX, ...
                               this.VolumeGradientY, ...
                               this.VolumeGradientZ);
            else
                image = volumeRender('render', this.objectHandle, ...
                               this.LightSources, ...
                               this.VolumeEmission, ...
                               this.VolumeReflection, ...
                               this.VolumeAbsorption, ...
                               this.VolumeIllumination, single(scales), ...
                               single(this.ElementSizeUm), uint64(resolution), ...
                               single(matrix), single(props), ...
                               single(this.OpacityThreshold), single(this.Color));
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
