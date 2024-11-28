classdef LightSource
    % LightSource Class
    % 
    % This class represents a light source with a specified position and color.
    % It provides methods for constructing, setting, and validating the properties
    % of the light source. Each property must be a 1x3 numeric array.
    %
    % Properties:
    %   - Position: The position of the light source in 3D space. It must be a
    %               numeric array of size [1,3].
    %   - Color:    The color of the light source, represented as RGB values.
    %               It must also be a numeric array of size [1,3].
    %
    % Methods:
    %   - LightSource: Constructor method to initialize the position and color
    %                  properties.
    %   - set.Position: Setter method to validate and update the Position property.
    %   - set.Color:    Setter method to validate and update the Color property.
    %
    % Example:
    %   % Create a LightSource object with position [1, 0, 0] and color [1, 1, 1]
    %   light = LightSource([1, 0, 0], [1, 1, 1]);
    %
    %   % Update the Position property
    %   light.Position = [0, 1, 0];
    %
    %   % Update the Color property
    %   light.Color = [0.5, 0.5, 0.5];


    properties(GetAccess='public')
        % Position: Position of the light source in 3D space (1x3 numeric array)
        Position;

        % Color: Color of the light source in RGB format (1x3 numeric array)
        Color;
    end
    methods
        % Constructor for the LightSource class
        function obj=LightSource(pos, col)
            % LightSource Constructor
            % Validates and sets the initial position and color properties.
            %
            % Arguments:
            %   - pos: Numeric array of size [1,3], representing the position
            %          in 3D space.
            %   - col: Numeric array of size [1,3], representing the RGB color.
            %
            % Throws:
            %   - Error if the size of pos or col is not [1,3].

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

        % Setter for the Color property
        function obj=set.Color(obj, val)
            % set.Color
            % Validates and updates the Color property.
            %
            % Arguments:
            %   - val: Numeric array of size [1,3] or [3,1] representing RGB color.
            %
            % Throws:
            %   - Error if val is not a valid size.

            if (all(size(val) == [1,3]))
                obj.Color = val;
            elseif (all(size(val) == [3,1]))
                obj.Color = val'; % transpose it
            else
                error('dimensions of ImageResolution must be [1,3]');
            end
        end

        % Setter for the Position property
        function obj=set.Position(obj, val)
            % set.Position
            % Validates and updates the Position property.
            %
            % Arguments:
            %   - val: Numeric array of size [1,3] or [3,1] representing 3D position.
            %
            % Throws:
            %   - Error if val is not a valid size.
            
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