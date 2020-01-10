classdef LightSource
    %For explanation see documentation (pdf)
    properties(GetAccess='public')
        Color;
        Position;
    end
    methods
        % construct
        function obj=LightSource(pos, col)
            validate=size(pos) == [1,3];
            if (all(validate))
               obj.Position = single(pos);
            else
               error('dimensions of position must be [1,3]');
            end
            validate=size(col) == [1,3];
            if (all(validate))
               obj.Color = single(col);
            else
               error('dimensions of color must be [1,3]');
            end
        end
        function obj=set.Color(obj, val)
            if (all(size(val) == [1,3]))
                obj.Color = val;
            elseif (all(size(val) == [3,1]))
                obj.Color = val'; % transpose it
            else
                error('dimensions of ImageResolution must be [1,3]');
            end
        end
        function obj=set.Position(obj, val)
            if (all(size(val) == [1,3]))
                obj.Position = val;
            elseif (all(size(val) == [3,1]))
                obj.Position = val'; % transpose it
            else
                error('dimensions of ImageResolution must be [1,3]');
            end
        end
    end % methods 
end % classdef
