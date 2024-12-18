classdef VolumeRender < handle
    % VolumeRender Class
    %
    % This class provides tools for rendering volumetric data with advanced
    % features such as stereo rendering, lighting, and gradient-based illumination.
    % It uses an external rendering engine implemented in C++ for performance.
    %
    % Properties:
    %   - FocalLength: Focal length of the camera (scalar).
    %   - DistanceToObject: Distance from the camera to the rendered object (scalar).
    %   - OpacityThreshold: Threshold for opacity in rendering (scalar).
    %   - LightSources: Array of LightSource objects defining scene lighting.
    %   - Color: RGB color of the rendered object (1x3 array).
    %   - FactorEmission: Intensity factor for emission lighting (scalar).
    %   - FactorReflection: Intensity factor for reflection lighting (scalar).
    %   - FactorAbsorption: Intensity factor for absorption (scalar).
    %   - CameraXOffset: Horizontal offset for stereo rendering (scalar).
    %   - StereoOutput: Stereo rendering mode (StereoRenderMode enumeration).
    %   - ElementSizeUm: Physical size of a voxel in micrometers (1x3 array).
    %   - RotationMatrix: Matrix for transforming the view orientation (3x3 matrix).
    %   - ImageResolution: Resolution of the rendered image (1x2 array).
    %   - TimeLastMemSync: Timestamp of the last memory synchronization.
    %   - VolumeReflection: Volume object containing reflection data.
    %   - VolumeEmission: Volume object or logical for emission data.
    %   - VolumeAbsorption: Volume object or logical for absorption data.
    %   - VolumeGradientX, VolumeGradientY, VolumeGradientZ: Gradient volumes.
    %   - VolumeIllumination: Volume object or logical for illumination.
    %
    % Methods:
    %   - VolumeRender: Constructor to initialize the rendering engine.
    %   - delete: Destructor to release memory in the external engine.
    %   - syncVolumes: Synchronizes the volume data with the rendering engine.
    %   - memInfo: Displays memory usage information.
    %   - memClear: Clears allocated memory in the rendering engine.
    %   - rotate: Applies rotations to the view matrix.
    %   - render: Renders the volume data to an image.
    %   - resetGradientVolumes: Resets gradient volumes for switching render modes.
    %   - normalizeSequence: Normalizes a 4D image sequence between [0,1].
    %   - normalizeImage: Normalizes an RGB image between [0,1].
    %
    % Example:
    %   % Create a VolumeRender object
    %   renderer = VolumeRender();
    %
    %   % Set some properties
    %   renderer.FocalLength = 1.5;
    %   renderer.Color = [1, 0, 0]; % Red
    %   renderer.ImageResolution = [512, 512];
    %
    %   % Load volumes and set properties
    %   vol = Volume(rand(100, 100, 100));
    %   renderer.VolumeReflection = vol;
    %   renderer.VolumeEmission = vol;
    %
    %   % Render the volume
    %   image = renderer.render();
    %
    % Notes:
    %   - This class requires a C++ rendering engine backend accessible via 
    %     the `volumeRender` function. Ensure the backend is correctly installed
    %     and linked.
    %   - Gradient volumes must be set to use gradient-based illumination.
    
    properties
        % FocalLength: Focal length of the camera (scalar, default: 0.0)
        %   Determines the field of view for rendering. Smaller values result in
        %   a wider field of view, while larger values provide a narrower view.
        FocalLength(1,1)      = 0.0;

        % DistanceToObject: Distance from the camera to the object (scalar, default: 0.0)
        %   Defines how far the object appears from the camera in the rendered view.
        DistanceToObject(1,1) = 0.0;

        % OpacityThreshold: Opacity cutoff for rendering (scalar, default: 0.95)
        %   Controls the threshold at which a voxel becomes opaque. Values closer
        %   to 1 make the rendering more transparent.
        OpacityThreshold(1,1) = 0.95;
        
        % LightSources: Array of LightSource objects (1xN, default: empty)
        %   Defines the lighting setup for the scene. Each LightSource object
        %   specifies position and color of a light source.
        LightSources(1,:)     = false;
        
        % Color: RGB color of the object (1x3 array, default: [1, 1, 1])
        %   Defines the base color of the rendered object.
        Color(1,3)            = [1,1,1];
 
        % FactorEmission: Intensity factor for emission lighting (scalar, default: 1.0)
        %   Adjusts the contribution of emission (light generated by the object itself).
        FactorEmission(1,1)    = 1.0;

        % FactorReflection: Intensity factor for reflection lighting (scalar, default: 1.0)
        %   Adjusts the contribution of reflected light.
        FactorReflection(1,1)  = 1.0;

        % FactorAbsorption: Intensity factor for light absorption (scalar, default: 1.0)
        %   Controls how much light is absorbed as it passes through the object.
        FactorAbsorption(1,1)  = 1.0;
        
        % CameraXOffset: Offset for stereo rendering (scalar, default: 0)
        %   Defines the horizontal displacement between two camera views for
        %   creating stereo or 3D anaglyph images. A value of 0 disables stereo rendering.
        CameraXOffset(1,1)    = 0;

        % StereoOutput: Stereo rendering mode (StereoRenderMode, default: RedCyan)
        %   Specifies the type of stereo output. Can be:
        %       - StereoRenderMode.RedCyan: Creates an anaglyph image for red-cyan glasses.
        %       - StereoRenderMode.LeftRightHorizontal: Creates side-by-side stereo images.
        StereoOutput StereoRenderMode = StereoRenderMode.RedCyan;
        
        % ElementSizeUm: Physical size of a voxel in micrometers (1x3 array, default: [1,1,1])
        %   Specifies the real-world dimensions of a single voxel in micrometers.
        ElementSizeUm(1,3)    = [1,1,1];

        % RotationMatrix: View rotation matrix (3x3 matrix, default: identity matrix)
        %   Determines the orientation of the camera in the 3D space.
        RotationMatrix(3,3)   = eye(3);

        % ImageResolution: Resolution of the rendered image (1x2 array, default: [0,0])
        %   Defines the width and height (in pixels) of the output image.
        ImageResolution(1,2)  = [0,0];
        
        % TimeLastMemSync: Timestamp of the last synchronization with the rendering engine (uint64, default: 0)
        %   Tracks the most recent memory synchronization to ensure up-to-date rendering.
        TimeLastMemSync       = uint64(0);
    end

    properties(SetObservable)
        % VolumeReflection: Volume object containing reflection data (Volume)
        %   Represents the volume dataset for reflective properties.
        VolumeReflection      = Volume(1);
        
        % VolumeEmission: Volume object or logical for emission data
        %   Contains data for the object's emissive light properties. Can be set
        %   to `false` if not in use.
        VolumeEmission        = false;

        % VolumeAbsorption: Volume object or logical for absorption data
        %   Contains data for the object's light absorption properties. 
        %   Must be a Volume object.
        VolumeAbsorption      = false;

        % VolumeGradientX, VolumeGradientY, VolumeGradientZ: Gradient volumes (Volume or logical)
        %   Contain gradient data for X, Y, and Z axes. Used for gradient-based lighting
        %   effects.
        VolumeGradientX       = false;
        VolumeGradientY       = false;
        VolumeGradientZ       = false;

        % VolumeIllumination: Volume object or logical for illumination data
        %   Represents the overall illumination volume.
        VolumeIllumination    = false;
    end

    properties (SetAccess = private, Hidden = true)
        objectHandle; % Handle to the underlying C++ memory manager instance
        
    end
    methods        
        % Constructor
        function this = VolumeRender(varargin)
            % VolumeRender Constructor
            % Initializes a new VolumeRender object and sets up the rendering engine.
            %
            % Arguments:
            %   - varargin: Optional parameters passed to the C++ backend.
            %
            % Behavior:
            %   - Sets up property change listeners for volume-related properties.
            %   - Initializes the rendering engine using volumeRender('new', ...).

            addlistener(this,'VolumeEmission','PostSet', @propEventHandler);
            addlistener(this,'VolumeAbsorption','PostSet', @propEventHandler);
            addlistener(this,'VolumeReflection','PostSet', @propEventHandler);
            addlistener(this,'VolumeGradientX','PostSet', @propEventHandler);
            addlistener(this,'VolumeGradientY','PostSet', @propEventHandler);
            addlistener(this,'VolumeGradientZ','PostSet', @propEventHandler);
            addlistener(this,'VolumeIllumination','PostSet', @propEventHandler);
            
            this.objectHandle = volumeRender('new', varargin{:});
        end

        % Destructor
        function delete(this)
            volumeRender('delete', this.objectHandle);
        end

        function syncVolumes(this)
            % syncVolumes Synchronizes the volume data with the rendering engine.
            %
            % Behavior:
            %   - Validates the status of volume properties.
            %   - Calls the backend to synchronize volume data for rendering.
            %   - Updates the TimeLastMemSync property with the current timestamp.

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
            % memInfo Displays memory usage information for the rendering engine.
            %
            % Behavior:
            %   - Calls volumeRender('mem_info', this.objectHandle) to fetch memory info.

            volumeRender('mem_info', this.objectHandle);
        end

        function memClear(this)
            % memClear Clears allocated memory in the rendering engine.
            %
            % Behavior:
            %   - Calls volumeRender('delete', this.objectHandle) to release memory.

            volumeRender('delete', this.objectHandle);
        end

        function rotate(this, alpha, beta, gamma)
            % rotate Applies rotations to the view matrix.
            %
            % Arguments:
            %   - alpha: Rotation angle around the X-axis (degrees).
            %   - beta: Rotation angle around the Y-axis (degrees).
            %   - gamma: Rotation angle around the Z-axis (degrees).
            %
            % Behavior:
            %   - Updates the RotationMatrix property by applying the specified rotations.
            
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
            % render Renders the volume data to an image.
            %
            % Returns:
            %   - image: The rendered image. If StereoOutput is set to
            %     StereoRenderMode.RedCyan, a 3D anaglyph image is returned.
            %
            % Behavior:
            %   - Calls the rendering backend to generate a 2D or stereo image.
            %   - If CameraXOffset is non-zero, renders stereo views and combines them.
            
            if (this.CameraXOffset==0)
                image=p_render(this, single(this.CameraXOffset), flip(this.ImageResolution));
            else
                % Render stereo (combine left and right images)
                base = this.CameraXOffset/2;
                
                angleFieldOfview=2*atan(1/this.FocalLength);
                delta = round((base * this.ImageResolution(2)) / ...
                        (2*this.FocalLength * tan(angleFieldOfview/2)));

                % Adjust resolution for stereo rendering
                resolution=flip(this.ImageResolution) + [0, delta];
                rightImage=p_render(this, base, resolution);
                leftImage=p_render(this, -base, resolution);
                
                % combine the 2 images, crop delta
                rect=[(delta+1) 0 size(leftImage,2) size(leftImage,1)];
                leftImage=imcrop(leftImage, rect);
                
                rect=[0 0 (size(rightImage,2)-delta) size(rightImage,1)];
                rightImage=imcrop(rightImage, rect);
                
                % Combine left and right images based on StereoOutput mode
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
            % resetGradientVolumes Resets the gradient volume properties.
            %
            % Behavior:
            %   - Sets the `VolumeGradientX`, `VolumeGradientY`, and `VolumeGradientZ`
            %     properties to `false`.
            %   - Disables gradient-based rendering by clearing existing gradient data.
            %
            % Example:
            %   % Create a VolumeRender object
            %   renderer = VolumeRender();
            %
            %   % Assign gradient volumes
            %   renderer.VolumeGradientX = Volume(rand(100, 100, 100));
            %   renderer.VolumeGradientY = Volume(rand(100, 100, 100));
            %   renderer.VolumeGradientZ = Volume(rand(100, 100, 100));
            %
            %   % Reset gradient volumes
            %   renderer.resetGradientVolumes();
            %   % At this point, VolumeGradientX, VolumeGradientY, and VolumeGradientZ
            %   % are set to false, and gradient rendering is disabled.
            %
            % Notes:
            %   - This method is useful for switching between rendering modes that do not
            %     require gradient-based lighting.
            %   - Ensure that gradient volumes are re-assigned if gradient-based rendering
            %     is required after calling this method.

            % Reset gradient volume properties
            this.VolumeGradientX       = false;
            this.VolumeGradientY       = false;
            this.VolumeGradientZ       = false;
        end

        % Setter for LightSources
        function set.LightSources(this, val)
            % set.LightSources Ensures the LightSources property is correctly set.
            %
            % Arguments:
            %   - val: Array of LightSource objects.
            %
            % Behavior:
            %   - Validates that `val` is a 1xN array of LightSource objects.
            %   - Throws an error if validation fails.

            if (all(isa(val, 'LightSource')))
                this.LightSources = val;
            else
                error('LightSources must be a 1xN vector with data of type LightSource!');
            end
        end
        
        % Setter for VolumeIllumination
        function set.VolumeIllumination(this, val)
            % set.VolumeIllumination Validates the VolumeIllumination property.
            %
            % Arguments:
            %   - val: Must be of type `Volume`.
            %
            % Behavior:
            %   - Checks if `val` is a Volume object.
            %   - Throws an error if validation fails.

            if (isa(val,'Volume'))
                this.VolumeIllumination = val;
            else
                error('VolumeEmission must be of type Volume');
            end
        end
        
        % Setter for VolumeEmission
        function set.VolumeEmission(this, val)
            % set.VolumeEmission Validates the VolumeEmission property.
            %
            % Arguments:
            %   - val: Must be of type `Volume` or `false`.
            %
            % Behavior:
            %   - Ensures the value is either a Volume object or logical false.
            %   - Throws an error if validation fails.

            if (isa(val,'Volume'))
                this.VolumeEmission = val;
            else
                error('VolumeEmission must be of type Volume');
            end
        end
        
        % Setter for VolumeReflection
        function set.VolumeReflection(this, val)
            % set.VolumeReflection Validates the VolumeReflection property.
            %
            % Arguments:
            %   - val: Must be of type `Volume`.
            %
            % Behavior:
            %   - Ensures the value is a Volume object.
            %   - Throws an error if validation fails.

            if (isa(val,'Volume'))
                this.VolumeReflection = val;
            else
                error('VolumeReflection must be of type Volume');
            end
        end
        
        % Setter for VolumeGradientX
        function set.VolumeGradientX(this, val)
            % set.VolumeGradientX Validates the VolumeGradientX property.
            %
            % Arguments:
            %   - val: Must be of type `Volume` or `false`.
            %
            % Behavior:
            %   - Ensures the value is a Volume object or logical false.
            %   - Throws an error if validation fails.
            if (isa(val,'Volume') || val == false)
                this.VolumeGradientX = val;
            else
                error('VolumeGradientX must be of type Volume');
            end
        end

        % Setter for VolumeGradientY
        function set.VolumeGradientY(this, val)
            % set.VolumeGradientY Validates the VolumeGradientY property.
            %
            % Arguments:
            %   - val: Must be of type `Volume` or `false`.
            %
            % Behavior:
            %   - Ensures the value is a Volume object or logical false.
            %   - Throws an error if validation fails.
            if (isa(val,'Volume') || val == false)
                this.VolumeGradientY = val;
            else
                error('VolumeGradientY must be of type Volume');
            end
        end

        % Setter for VolumeGradientZ
        function set.VolumeGradientZ(this, val)
            % set.VolumeGradientZ Validates the VolumeGradientZ property.
            %
            % Arguments:
            %   - val: Must be of type `Volume` or `false`.
            %
            % Behavior:
            %   - Ensures the value is a Volume object or logical false.
            %   - Throws an error if validation fails.

            if (isa(val,'Volume') || val == false)
                this.VolumeGradientZ = val;
            else
                error('VolumeGradientZ must be of type Volume');
            end
        end

        % Setter for VolumeAbsorption
        function set.VolumeAbsorption(this, val)
            % set.VolumeAbsorption Validates the VolumeAbsorption property.
            %
            % Arguments:
            %   - val: Must be of type `Volume`.
            %
            % Behavior:
            %   - Checks if the data in the Volume object contains negative values.
            %   - Throws a warning if data contains values < 0.
            %   - Throws an error if `val` is not of type Volume.

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
            % p_render Internal rendering function (GPU backend).
            %
            % Arguments:
            %   - CameraXOffset: Horizontal offset between the two cameras for stereo rendering.
            %                    A scalar value.
            %   - resolution: Resolution of the output image, specified as a 1x2 array 
            %                 [height, width].
            %
            % Returns:
            %   - image: A rendered image based on the specified settings. The format and 
            %            dimensionality depend on the rendering mode and other properties.
            %
            % Behavior:
            %   - Validates the presence of required volume data (`VolumeEmission`, 
            %     `VolumeReflection`, and `VolumeAbsorption`).
            %   - Checks for the optional gradient volumes (`VolumeGradientX`, `VolumeGradientY`, 
            %     `VolumeGradientZ`) and determines whether gradient-based rendering is enabled.
            %   - Synchronizes volumes with the GPU memory using `syncVolumes`.
            %   - Sends rendering parameters to the GPU backend via the `volumeRender` function.
            %   - If gradient volumes are used, additional data is included in the rendering 
            %     process.
            %
            % Notes:
            %   - This is a protected method and is intended to be called internally by 
            %     `render` or other methods.
            %   - Relies on an external GPU backend accessible through `volumeRender`.


            % Validate required volume properties
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

            % Check if gradient volumes are correctly set
            renderWithgradientVolumes=false;
            if (not(any([islogical(this.VolumeGradientX), ...
                         islogical(this.VolumeGradientY), ...
                         islogical(this.VolumeGradientZ)])))
                if not(all([isa(this.VolumeGradientX, 'Volume') ...
                            isa(this.VolumeGradientY, 'Volume') ...
                            isa(this.VolumeGradientZ, 'Volume')]))
                    error('All gradient dimensions need to be set and of type Volume!');
                else
                    renderWithgradientVolumes=true;
                end
            end

            % Synchronize volumes with GPU memory
            this.syncVolumes()
            
            % Set rendering parameters
            factors=[this.FactorEmission, this.FactorReflection, this.FactorAbsorption];
            props=[CameraXOffset, this.FocalLength, this.DistanceToObject];
            matrix=flip(this.RotationMatrix);

            % Call GPU rendering backend
            if (renderWithgradientVolumes)
                % Include gradient volumes in rendering
                image = volumeRender('render', this.objectHandle, ...
                               this.LightSources, ...
                               this.VolumeIllumination, single(factors), ...
                               single(this.ElementSizeUm), uint64(resolution), ...
                               single(matrix), single(props), ...
                               single(this.OpacityThreshold), single(this.Color), ...
                               this.VolumeGradientX, ...
                               this.VolumeGradientY, ...
                               this.VolumeGradientZ);
            else
                % Standard rendering without gradients
                image = volumeRender('render', this.objectHandle, ...
                               this.LightSources, ...
                               this.VolumeIllumination, single(factors), ...
                               single(this.ElementSizeUm), uint64(resolution), ...
                               single(matrix), single(props), ...
                               single(this.OpacityThreshold), single(this.Color));
            end
       end
    end
    
    methods(Access = public,Static = true)
        function normalized=normalizeSequence(sequence)
            % normalizeSequence Normalizes a multiframe (4D) image sequence to the range [0, 1].
            %
            % Arguments:
            %   - sequence: A 4D array representing a multiframe image sequence. The
            %               dimensions are expected to be [height, width, depth, frames].
            %
            % Returns:
            %   - normalized: A 4D array of the same size as `sequence`, where the pixel
            %                 intensities are linearly scaled between 0 and 1.
            %
            % Behavior:
            %   - Computes the global minimum and maximum pixel values across all frames.
            %   - Scales all pixel values to the range [0, 1] using the global min and max.
            %
            % Example:
            %   % Normalize a multiframe sequence
            %   sequence = rand(256, 256, 50, 10); % Random 4D data
            %   normalizedSequence = VolumeRender.normalizeSequence(sequence);
            %
            % Notes:
            %   - This function requires the input to be a 4D array. If the input is not 4D,
            %     an error is thrown.
            %   - The normalization process preserves the relative intensity distribution
            %     across frames.
            
            if (ndims(sequence) < 4)
                error('input must be a multiframe image (4D)');
            end
            
            % Get global minimum and maximum values across all frames
            maxValue=max(sequence(:));
            minValue=min(sequence(:));
            
            % Preallocate the normalized sequence
            normalized=zeros(size(sequence));

            % Normalize each frame individually
            for i=1:size(sequence,4)
                tmp=sequence(:,:,:,i);
                normalized(:,:,:,i) = ...
                    VolumeRender.normalizeImage(tmp, minValue, maxValue);
            end
        end
        
        function img=normalizeImage(ImageRGB, varargin)
            % normalizeImage Normalizes an RGB image to the range [0, 1].
            %
            % Arguments:
            %   - ImageRGB: A 3D array representing an RGB image with dimensions
            %               [height, width, 3].
            %   - varargin: Optional minimum and maximum values for normalization.
            %               If not provided, the function computes the min and max values
            %               from the input image.
            %               - varargin{1}: Custom minimum value.
            %               - varargin{2}: Custom maximum value.
            %
            % Returns:
            %   - img: A 3D array of the same size as `ImageRGB`, where pixel intensities
            %          are linearly scaled between 0 and 1.
            %
            % Behavior:
            %   - Separately processes the R, G, and B channels.
            %   - Optionally accepts user-defined min and max values for scaling.
            %
            % Example:
            %   % Normalize an RGB image with auto-computed min/max
            %   img = rand(256, 256, 3) * 255; % Random RGB image
            %   normalizedImg = VolumeRender.normalizeImage(img);
            %
            %   % Normalize with custom min and max values
            %   normalizedImg = VolumeRender.normalizeImage(img, 0, 255);
            %
            % Notes:
            %   - The function can handle cases where pixel values are negative or exceed
            %     the standard [0, 255] range.
            %   - If minValue is negative, the function shifts pixel values accordingly
            %     before normalization.

            % Extract individual color channels
            red = ImageRGB(:,:,1);
            green = ImageRGB(:,:,2);
            blue = ImageRGB(:,:,3);

            % Determine the minimum value
            if (nargin < 2)
                minValue = min([min(red(:)), min(green(:)), min(blue(:))]);
            else
                minValue = varargin{1};
            end
            
            % Determine the maximum value
            if (nargin < 3)
                maxValue = max([max(red(:)), max(green(:)), max(blue(:))]);
            else
                maxValue = varargin{2};
            end
            
            % Handle negative minimum value by shifting pixel intensities
            if (minValue < 0)
                red = red + minValue;
                green = green + minValue;
                blue = blue + minValue;
                maxValue = maxValue + abs(minValue);
            end
            
            % Normalize each channel
            tmp = zeros(size(ImageRGB));
            tmp(:,:,1) = red/maxValue;
            tmp(:,:,2) = green/maxValue;
            tmp(:,:,3) = blue/maxValue;
            
            % Return the normalized image
            img = tmp;
        end
    end
end % classdef

% propEventHandler Function
%
% Handles events triggered when observable properties of the `VolumeRender`
% class are updated. Specifically, it tracks updates to properties such as 
% `VolumeEmission`, `VolumeReflection`, `VolumeAbsorption`, and gradient volumes.
%
% Arguments:
%   - ~: Unused (listener object, not needed in this case).
%   - eventData: Event data structure containing details about the property change.
%
% Behavior:
%   - If the updated property is a `Volume` object, updates its `TimeLastUpdate` 
%     property to the current timestamp.
%   - Ensures that changes to the volume-related properties are tracked consistently.
%
% Notes:
%   - This function is called automatically when a `PostSet` event is triggered
%     for observable properties.
function propEventHandler(~,eventData)
    % List of volume-related properties to monitor
    v = ["VolumeEmission", "VolumeAbsorption", "VolumeReflection", ...
         "VolumeGradientX", "VolumeGradientY", "VolumeGradientZ", ...
         "VolumeIllumination"];

    % Check if the changed property is one of the monitored properties
    if any(v == eventData.Source.Name)
        switch eventData.EventName % Get the event name
        case 'PostSet'
            % Update TimeLastUpdate for Volume objects
            if (isa(eventData.AffectedObject.(eventData.Source.Name), 'Volume'))
                % Update the TimeLastUpdate of the affected Volume object
                eventData.AffectedObject.(eventData.Source.Name).TimeLastUpdate = timestamp;
            end
        end
    end
end

