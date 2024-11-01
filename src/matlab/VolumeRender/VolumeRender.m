classdef VolumeRender < handle
    % VOLUMERENDER Class for managing and rendering volumetric data with lighting and material properties.
    %   This class handles various rendering settings, such as light source properties,
    %   material factors (emission, reflection, and absorption), and camera parameters.
    %   The rendering supports stereoscopic output and customizable volume data for realistic
    %   rendering effects, useful in fields like microscopy and medical imaging.
    %
    % Properties:
    %   FocalLength           - Focal length of the virtual camera.
    %   DistanceToObject      - Distance from the camera to the object in the scene.
    %   OpacityThreshold      - Threshold for opacity rendering; values above this are fully opaque.
    %   LightSources          - Array of light sources influencing the scene lighting.
    %   Color                 - Base RGB color of the rendered volume.
    %   FactorEmission        - Emission factor controlling how much light the volume emits.
    %   FactorReflection      - Reflection factor determining reflective intensity.
    %   FactorAbsorption      - Absorption factor defining how much light is absorbed by the volume.
    %   CameraXOffset         - Offset between two cameras for stereoscopic rendering.
    %   StereoOutput          - Stereo mode for 3D output (e.g., RedCyan anaglyph).
    %   ElementSizeUm         - Element size in micrometers, representing voxel dimensions.
    %   RotationMatrix        - Rotation matrix for setting the view orientation.
    %   ImageResolution       - Resolution of the output image.
    %   Shininess             - Shininess value for specular highlights in Blinn-Phong shading.
    %   ScatteringWeight      - Weighting factor between reflection and scattering.
    %   HgAsymmetry           - Asymmetry parameter for the Henyey-Greenstein scattering phase function.
    %   TimeLastMemSync       - Timestamp of the last memory synchronization for the volumes.
    %
    % Observable Properties:
    %   VolumeReflection      - Volume data representing reflection characteristics.
    %   VolumeEmission        - Volume data for emission; represents light-emitting areas.
    %   VolumeAbsorption      - Volume data for absorption; higher values absorb more light.
    %   VolumeGradientX       - Volume gradient data in the X direction.
    %   VolumeGradientY       - Volume gradient data in the Y direction.
    %   VolumeGradientZ       - Volume gradient data in the Z direction.
    %
    % Private Properties:
    %   objectHandle          - Handle to the underlying C++ memory manager for volume rendering.
    %
    % Methods:
    %   VolumeRender          - Constructor to initialize volume rendering properties and listeners.
    %   delete                - Destructor to clean up resources.
    %   syncVolumes           - Synchronizes the volume data with the GPU memory.
    %   memInfo               - Displays memory usage information.
    %   memClear              - Clears GPU memory associated with the volume rendering.
    %   rotate                - Applies a rotation to the view orientation using given angles.
    %   render                - Renders an image from the volume data with optional stereoscopic effect.
    %   resetGradientVolumes  - Resets gradient volume properties to switch rendering to gradient mode.
    %
    % Static Methods:
    %   normalizeSequence     - Normalizes a 4D image sequence to values between [0, 1].
    %   normalizeImage        - Normalizes an RGB image to values between [0, 1] with optional min/max input.
    %
    % Usage:
    %   Create a VolumeRender object, configure light sources, material factors, and other
    %   properties, and then call the render method to obtain the rendered image.
    %
    % Notes:
    %   - For more detailed documentation, refer to the accompanying documentation PDF.
    %   - Ensure that volume properties (VolumeReflection, VolumeEmission, etc.) are set with
    %     compatible data types to prevent rendering errors.
    %
    % Example:
    %   renderer = VolumeRender();
    %   renderer.FactorEmission = 0.8;
    %   renderer.Shininess = 50;
    %   renderer.LightSources = [LightSource(...), LightSource(...)]; % Define light sources
    %   image = renderer.render();
    %   imshow(image);

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

