classdef Volume < handle
    % Volume Class
    %
    % This class represents a volumetric data structure with a variety of 
    % utilities for manipulation, analysis, and visualization. The class 
    % supports dynamic updates to its data and tracks changes with a timestamp.
    %
    % Properties:
    %   - Data: Volumetric data, represented as a numeric array. It supports 
    %           multi-dimensional arrays. Changes to this property trigger 
    %           an event and update the `TimeLastUpdate`.
    %
    %   - TimeLastUpdate: The timestamp of the last modification to the `Data`
    %                     property. Stored as a uint64 value.
    %
    % Methods:
    %   - Volume: Constructor that initializes the `Data` property and sets 
    %             up a listener to track updates.
    %   - set.Data: Setter for the `Data` property to ensure data is stored 
    %               as single-precision.
    %   - resize: Resizes the volumetric data using 2D or 3D interpolation.
    %   - size: Returns the size of the `Data` array.
    %   - pad: Pads the volumetric data with specified padding and values.
    %   - mip: Computes the maximum intensity projection (MIP) of the volumetric data.
    %   - mean: Computes the mean value of the data.
    %   - max: Computes the maximum value in the data.
    %   - min: Computes the minimum value in the data.
    %   - grad: Computes the gradient of the data along all dimensions, returning
    %           three new `Volume` objects for each axis.
    %   - normalize: Linearly normalizes the data to a specified range.
    %
    % Example:
    %   % Create a Volume object with random data
    %   vol = Volume(rand(100, 100, 50));
    %
    %   % Resize the volume
    %   vol.resize([200, 200, 100]);
    %
    %   % Pad the volume with zeros
    %   vol.pad(10, 0);
    %
    %   % Compute the maximum intensity projection
    %   mipImage = vol.mip();
    %
    %   % Normalize the data to the range [0, 1]
    %   vol.normalize(0, 1);
    %
    %   % Compute the gradient of the volume
    %   [gx, gy, gz] = vol.grad();
    %
    % Notes:
    %   - The `Data` property must always contain numeric arrays. If multi-dimensional,
    %     it assumes a 3D structure for most operations.
    %   - The class inherits from the `handle` class, allowing it to modify the 
    %     object directly without creating copies.
    
    properties(SetObservable)
        % Data: The main volumetric data (numeric array)
        Data=[];
    end

    properties
    % TimeLastUpdate: Timestamp of the last update to the Data property
        TimeLastUpdate = uint64(0);
    end
    
    methods
        % Constructor
        function this=Volume(data)
            % Volume Constructor
            % Initializes the Volume object and sets up a listener for the
            % `Data` property.
            %
            % Arguments:
            %   - data: Initial volumetric data as a numeric array.

            addlistener(this,'Data','PostSet',@propEventHandler);
            this.Data=single(data);
        end

        % Setter for the Data property
        function set.Data(obj, data)
            % set.Data
            % Validates and assigns the `Data` property.
            %
            % Arguments:
            %   - data: New data to assign, converted to single-precision.

            obj.Data=single(data);
        end
        
        % Resize method
        function resize(obj, newsize)
            % resize
            % Resizes the volumetric data to the specified size.
            %
            % Arguments:
            %   - newsize: New size for the data as a numeric array.

            if (ndims(obj.Data) == 3)
                obj.Data=imresize3(obj.Data, newsize);
            else
                obj.Data=imresize(obj.Data,newsize);
            end
        end

        % Size method
        function s=size(obj)
            % size
            % Returns the size of the volumetric data.
            %
            % Returns:
            %   - s: Size of the `Data` property.

            s=size(obj.Data);
        end

        % Pad method
        function pad(obj, padding, value)
            % pad
            % Pads the volumetric data with specified values.
            %
            % Arguments:
            %   - padding: Padding size for each dimension.
            %   - value: Value to use for padding.

            if (ndims(obj.Data) == 3)
                % newsize = size(obj.Data) + padding;
                obj.Data = padarray(obj.Data,[padding padding], value, 'both');
                obj.Data = permute(obj.Data,[1,3,2]);
                
                obj.Data = padarray(obj.Data,[0 padding], value, 'both');
                obj.Data = permute(obj.Data,[1,3,2]);
            end
        end
        
        % Maximum intensity projection (MIP)
        function image=mip(obj)
            % mip
            % Computes the maximum intensity projection (MIP).
            %
            % Returns:
            %   - image: 2D image from the MIP operation.

            image = max(permute(obj.Data,[2 1 3]), [], 3);
        end
        
        % Compute mean value
        function value=mean(obj)
            % mean
            % Computes the mean value of the volumetric data.
            %
            % Returns:
            %   - value: Scalar mean value.
            value=mean(obj.Data(:));
        end

        % Compute max value
        function value=max(obj)
            % max
            % Computes the maximum value in the volumetric data.
            %
            % Returns:
            %   - value: Scalar maximum value.

            value=max(obj.Data(:));
        end

        % Compute min value
        function value=min(obj)
            % min
            % Computes the minimum value in the volumetric data.
            %
            % Returns:
            %   - value: Scalar minimum value.

            value=min(obj.Data(:));
        end

        % Compute gradients
        function [gx, gy, gz]=grad(obj, varargin)
            % grad
            % Computes the gradients of the volumetric data along all dimensions.
            %
            % Returns:
            %   - gx: Gradient along the x-axis as a Volume object.
            %   - gy: Gradient along the y-axis as a Volume object.
            %   - gz: Gradient along the z-axis as a Volume object.
            %
            % Arguments:
            %   - varargin: Optionally specifies a new size to resize the gradients.

            [data_x, data_y, data_z]=gradient(obj.Data);
            gx=Volume(data_x);
            gy=Volume(data_y);
            gz=Volume(data_z);


            if ( length(varargin) == 1 )
               newsize = varargin{1};
               gx.resize(newsize);
               gy.resize(newsize);
               gz.resize(newsize);
            end
        end

        % Normalize method
        function normalize(obj, newMin, newMax)
            % normalize
            % Linearly normalizes the data to the specified range.
            %
            % Arguments:
            %   - newMin: New minimum value after normalization.
            %   - newMax: New maximum value after normalization.

            max=obj.max();
            min=obj.min();
            obj.Data=(obj.Data-min) * (newMax - newMin)/ ...
                    (max-min) + newMin;
        end
    end
end % classdef

% Event handler for property changes
function propEventHandler(~,eventData)
    % propEventHandler
    % Handles events triggered by changes to the `Data` property.
    %
    % Updates the `TimeLastUpdate` property with the current timestamp.

    switch eventData.Source.Name % Get property name
        case 'Data'
            switch eventData.EventName % Get the event name
                case 'PostSet'
                    % set TimeLastUpdate to the current timestamp
                    eventData.AffectedObject.TimeLastUpdate = timestamp;
            end
    end
end