classdef Volume < handle
    % VOLUME Class for handling and manipulating volumetric data.
    %   This class provides methods to perform various operations on volumetric data,
    %   such as resizing, padding, normalization, and gradient calculation.
    %   It also supports Maximum Intensity Projection (MIP) and property observation
    %   to track updates to the volume data.
    %
    % Properties:
    %   Data               - Observable property containing the volumetric data as a 3D array.
    %                        The data is converted to single precision upon assignment.
    %   TimeLastUpdate     - Timestamp of the last update to the data, set whenever 'Data' changes.
    %
    % Methods:
    %   Volume             - Constructor to initialize the volume data and set up an observer on 'Data'.
    %   set.Data           - Setter method to enforce single-precision format for the data property.
    %   resize             - Resizes the volumetric data to a new size.
    %   size               - Returns the current size of the data array.
    %   pad                - Pads the volumetric data with a specified value around the edges.
    %   mip                - Computes a Maximum Intensity Projection (MIP) along the z-axis.
    %   mean               - Calculates the mean value of all elements in the data array.
    %   max                - Returns the maximum value within the data array.
    %   min                - Returns the minimum value within the data array.
    %   grad               - Computes the gradient of the volume in x, y, and z directions.
    %                        Optionally, the gradients can be resized to a specified size.
    %   normalize          - Performs linear normalization of the data to a specified range.
    %
    % Example:
    %   volData = rand(50, 50, 50);  % Generate random 3D data
    %   volume = Volume(volData);    % Create a Volume object
    %   volume.resize([100, 100, 100]); % Resize the volume
    %   mipImage = volume.mip();     % Compute a Maximum Intensity Projection (MIP)
    %   avgIntensity = volume.mean(); % Get the mean intensity of the volume
    %
    % Notes:
    %   - For more detailed information, refer to the accompanying documentation PDF.
    %   - The 'Data' property is observable; changes to it trigger an update to 'TimeLastUpdate'.
    
    properties(SetObservable)
        Data=[];
    end

    properties
        TimeLastUpdate = uint64(0);
    end
    
    methods
        function this=Volume(data)
            addlistener(this,'Data','PostSet',@propEventHandler);
            this.Data=single(data);
        end

        function set.Data(obj, data)
            obj.Data=single(data);
        end
        
        function resize(obj, newsize)
            if (ndims(obj.Data) == 3)
                obj.Data=imresize3(obj.Data, newsize);
            else
                obj.Data=imresize(obj.Data,newsize);
            end
        end

        function s=size(obj)
            s=size(obj.Data);
        end

        function pad(obj, padding, value)
            if (ndims(obj.Data) == 3)
                % newsize = size(obj.Data) + padding;
                obj.Data = padarray(obj.Data,[padding padding], value, 'both');
                obj.Data = permute(obj.Data,[1,3,2]);
                
                obj.Data = padarray(obj.Data,[0 padding], value, 'both');
                obj.Data = permute(obj.Data,[1,3,2]);
            end
        end
        
        function image=mip(obj)
          % maximum intensity projection
          image = max(permute(obj.Data,[2 1 3]), [], 3);
        end
        
        function value=mean(obj)
        % compute mean [scalar]
          value=mean(obj.Data(:));
        end

        function value=max(obj)
        % compute max [scalar]
          value=max(obj.Data(:));
        end

        function value=min(obj)
        % compute min [scalar]
          value=min(obj.Data(:));
        end

        function [gx, gy, gz]=grad(obj, varargin)
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

        function normalize(obj, newMin, newMax)
        % linear normalization
           max=obj.max();
           min=obj.min();
           obj.Data=(obj.Data-min) * (newMax - newMin)/ ...
                    (max-min) + newMin;
        end
    end
end % classdef

% called whenever data of a volume is set/changed
function propEventHandler(~,eventData)
   switch eventData.Source.Name % Get property name
      case 'Data'
         switch eventData.EventName % Get the event name
            case 'PostSet'
                % set TimeLastUpdate to the current timestamp
                eventData.AffectedObject.TimeLastUpdate = timestamp;
         end
   end
end