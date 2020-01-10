classdef Volume < handle
    %For explanation see documentation (pdf)
    
    properties
        Data=[];
    end
    
    methods
        function v=Volume(data)
            v.Data=single(data);
        end

        function set.Data(obj, data)
            obj.Data=single(data);
        end
        
        function resize(obj, newsize)
            newsize = floor(newsize);
            if (ndims(obj.Data) == 3)
                % author: Dr. Olaf Ronneberger
                tmp = imresize(obj.Data, [newsize(1) newsize(2)]);
                obj.Data = reshape(imresize(reshape(tmp, [newsize(1)*newsize(2), size(obj.Data,3)]), ...
					 [newsize(1)*newsize(2), newsize(3)]),...
                     newsize);
            else
                obj.Data=imresize(obj.Data,[newsize(1),newsize(2)]);
            end
        end
    end
end
