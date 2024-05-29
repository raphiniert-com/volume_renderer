classdef Volume < handle
    %For explanation see documentation (pdf)
    
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