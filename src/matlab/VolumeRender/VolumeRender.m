classdef VolumeRender < handle
    % VolumeRender Class
    %
    % This class provides a GPU-accelerated renderer for volumetric data, with
    % support for different lighting models, stereo rendering, and advanced
    % shading techniques. It includes customizable settings for light sources,
    % color intensities, object-camera distances, and rendering parameters
    % related to scattering and reflection.
    %
    % Properties:
    % - FocalLength (1x1 float): Specifies the camera's focal length, affecting
    %   field of view and depth perception in the render.
    %
    % - DistanceToObject (1x1 float): Distance from the camera to the object,
    %   used in depth and scaling calculations.
    %
    % - OpacityThreshold (1x1 float): Alpha threshold for transparency. Pixels
    %   reaching this threshold are considered opaque and terminate further
    %   ray tracing.
    %
    % - LightSources (1xN LightSource array): List of LightSource objects that
    %   affect the rendered volume.
    %
    % - Color (1x3 array): RGB color applied to the volume, affecting the
    %   appearance of emitted and scattered light.
    %
    % - FactorEmission (1x1 float): Controls the intensity of emission for the
    %   volume data.
    %
    % - FactorReflection (1x1 float): Scales the reflection intensity within the
    %   volume.
    %
    % - FactorAbsorption (1x1 float): Scales the absorption of light, affecting
    %   the attenuation of light through the volume.
    %
    % - CameraXOffset (1x1 float): Specifies the horizontal offset of the
    %   camera for stereo rendering.
    %
    % - StereoOutput (StereoRenderMode enum): Defines stereo output mode,
    %   supporting RedCyan and LeftRightHorizontal render formats.
    %
    % - ElementSizeUm (1x3 array): Physical dimensions of a voxel in micrometers.
    %
    % - RotationMatrix (3x3 matrix): Rotation matrix applied to the camera
    %   perspective, supporting 3D navigation.
    %
    % - ImageResolution (1x2 array): Specifies the resolution of the rendered
    %   image in pixels (width x height).
    %
    % - Shininess (1x1 float): Shininess exponent for Blinn-Phong shading,
    %   controlling the sharpness of specular highlights.
    %
    % - ScatteringWeight (1x1 float): Weight factor between Blinn-Phong
    %   reflection and Henyey-Greenstein scattering.
    %
    % - HgAsymmetry (1x1 float): Asymmetry factor for Henyey-Greenstein phase
    %   function, controlling forward/backward scattering.
    %
    % - TimeLastMemSync (uint64): Timestamp of the last memory synchronization
    %   with the GPU, ensuring updated volumes are used.
    %
    % Observable Properties:
    % - VolumeReflection, VolumeEmission, VolumeAbsorption, VolumeGradientX,
    %   VolumeGradientY, VolumeGradientZ, VolumePhase (Volume objects):
    %   References to volume data used for specific aspects of rendering.
    %
    %   * VolumeReflection - Volume data for reflection.
    %   * VolumeEmission - Volume data for emission effects.
    %   * VolumeAbsorption - Volume data representing absorption levels.
    %   * VolumeGradientX/Y/Z - Gradient volumes along the X, Y, and Z axes.
    %   * VolumePhase - Volume for phase function calculations.
    %
    % Private Properties:
    % - objectHandle (pointer): Handle to the C++ memory manager for direct
    %   memory management.
    
    % Methods:
    % - VolumeRender: Constructor. Initializes volumes and GPU resources.
    % - delete: Destructor. Cleans up GPU memory associated with the instance.
    % - syncVolumes: Synchronizes volumes to the GPU, ensuring they are current.
    % - memInfo: Outputs GPU memory usage information.
    % - memClear: Clears GPU memory allocated for this instance.
    % - rotate: Applies a rotation matrix to adjust the view perspective.
    % - render: Renders the volume to an image, with support for stereo
    %   rendering modes.
    %
    % Set Methods:
    % - resetGradientVolumes: Resets gradient volumes to switch back to 
    %   gradient calculation.
    %
    % Static Methods:
    % - normalizeSequence: Normalizes a 4D image sequence to [0,1].
    % - normalizeImage: Normalizes a 3D RGB image to [0,1].

    properties
        FocalLength(1,1)       = 0.0;
        DistanceToObject(1,1)  = 0.0;
        OpacityThreshold(1,1)  = 0.95;
        
        LightSources(1,:)      = false;
        
        Color(1,3)             = [1,1,1];
 
        FactorEmission(1,1)    = 1.0;
        FactorReflection(1,1)  = 1.0;
        FactorAbsorption(1,1)  = 1.0;
        
        CameraXOffset(1,1)    = 0;
        StereoOutput StereoRenderMode = StereoRenderMode.RedCyan;
        
        ElementSizeUm(1,3)    = [1,1,1];
        RotationMatrix(3,3)   = eye(3);
        ImageResolution(1,2)  = [0,0];
        
        Shininess(1,1)        = 30.0;
        ScatteringWeight(1,1) = 0.5;
        HgAsymmetry(1,1)      = 0.8;
        
        TimeLastMemSync       = uint64(0);
    end

    properties(SetObservable)
        VolumeReflection      = Volume(1);
        
        VolumeEmission        = false;
        VolumeAbsorption      = false;
        VolumeGradientX       = false;
        VolumeGradientY       = false;
        VolumeGradientZ       = false;
        VolumePhase           = false;
    end

    properties (SetAccess = private, Hidden = true)
        objectHandle; % Handle to the underlying C++ memory manager instance
        
    end
    methods        
        % Constructor
        function this = VolumeRender(varargin)
            addlistener(this,'VolumeEmission','PostSet', @propEventHandler);
            addlistener(this,'VolumeAbsorption','PostSet', @propEventHandler);
            addlistener(this,'VolumeReflection','PostSet', @propEventHandler);
            addlistener(this,'VolumeGradientX','PostSet', @propEventHandler);
            addlistener(this,'VolumeGradientY','PostSet', @propEventHandler);
            addlistener(this,'VolumeGradientZ','PostSet', @propEventHandler);
            addlistener(this,'VolumePhase','PostSet', @propEventHandler);
            
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
                
                if this.StereoOutput == StereoRenderMode.RedCyan
                    % RGB - anaglyph
                    image=zeros([size(leftImage,1), size(leftImage,2), 3]);
                    image(:,:,1) = leftImage(:,:,1);
                    image(:,:,2) = rightImage(:,:,2);
                    image(:,:,3) = rightImage(:,:,3);
                elseif this.StereoOutput == StereoRenderMode.LeftRightHorizontal
                    image = [leftImage, rightImage];
                end
                    
            end
        end
    end
    
    % set methods
    methods
        function resetGradientVolumes(this)
            % reset gradient volumes in order to switch render to gradient computation
            this.VolumeGradientX       = false;
            this.VolumeGradientY       = false;
            this.VolumeGradientZ       = false;
        end

        function set.LightSources(this, val)
            % ensure correct type
            if (all(isa(val, 'LightSource')))
                this.LightSources = val;
            else
                error('LightSources must be a 1xN vector with data of type LightSource!');
            end
        end

        function set.VolumePhase(this, val)
            if (isa(val,'Volume'))
                this.VolumePhase = val;
            else
                error('VolumePhase must be of type Volume');
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
            if (isa(val,'Volume') || val == false)
                this.VolumeGradientX = val;
            else
                error('VolumeGradientX must be of type Volume');
            end
        end
        function set.VolumeGradientY(this, val)
            if (isa(val,'Volume') || val == false)
                this.VolumeGradientY = val;
            else
                error('VolumeGradientY must be of type Volume');
            end
        end
        function set.VolumeGradientZ(this, val)
            if (isa(val,'Volume') || val == false)
                this.VolumeGradientZ = val;
            else
                error('VolumeGradientZ must be of type Volume');
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
                
            if (all(validate))
                display(validate);
                error('Not all volumes are properly set!');
            end

            % check if gradient volumes are set properly
            if (not(any([islogical(this.VolumeGradientX), ...
                         islogical(this.VolumeGradientY), ...
                         islogical(this.VolumeGradientZ)])))
                if not(all([isa(this.VolumeGradientX, 'Volume') ...
                            isa(this.VolumeGradientY, 'Volume') ...
                            isa(this.VolumeGradientZ, 'Volume')]))
                    error('All gradient dimensions need to be set and of type Volume!');
                end
            end

            % sync volumes onto the device
            this.syncVolumes()
            
            factors=[this.FactorEmission, this.FactorReflection, this.FactorAbsorption];
            props=[CameraXOffset, this.FocalLength, this.DistanceToObject];

            matrix=flip(this.RotationMatrix);

            image = volumeRender('render', this.objectHandle, ...
                           this.LightSources, ...
                           this.VolumePhase, single(factors), ...
                           single(this.ElementSizeUm), uint64(resolution), ...
                           single(matrix), single(props), ...
                           single(this.OpacityThreshold), single(this.Color), ...
                           single(this.Shininess), single(this.ScatteringWeight), ...
                           single(this.HgAsymmetry));
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

% called whenever data of a volume is set/changed
function propEventHandler(~,eventData)
    v = ["VolumeEmission", "VolumeAbsorption", "VolumeReflection", ...
         "VolumeGradientX", "VolumeGradientY", "VolumeGradientZ", ...
         "VolumeIllumination"];
    if any(v == eventData.Source.Name)
        switch eventData.EventName % Get the event name
        case 'PostSet'
            % set TimeLastUpdate to the current timestamp if volume
            if (isa(eventData.AffectedObject.(eventData.Source.Name), 'Volume'))
                eventData.AffectedObject.(eventData.Source.Name).TimeLastUpdate = timestamp;
            end
        end
    end
end

