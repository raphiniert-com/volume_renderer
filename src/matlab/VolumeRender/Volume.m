classdef Volume < handle
    %For explanation see documentation (pdf)
    
    properties(SetObservable)
        Data=[];
    end

    properties
        TimeLastUpdate=[];
    end
    
    methods
        function v=Volume(data)
            addlistener(v,'Data','PostSet',@propEventHandler);
            v.Data=single(data);
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

        function normalize(obj, newMin, newMax)
        % linear normalization
           oldMax=obj.max();
           oldMin=obj.min();
           obj.Data=(obj.Data-oldMin) * (newMax - newMin)/ ...
                    (oldMax-oldMin) + newMin;
        end
    end
end

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
