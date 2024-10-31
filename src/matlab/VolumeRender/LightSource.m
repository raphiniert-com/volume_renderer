classdef LightSource
    % LIGHTSOURCE Class for defining a light source in the rendering environment.
    %   The LightSource class specifies a light source's position, color, and intensity.
    %   It supports both spotlight and diffuse lighting, allowing for flexible lighting
    %   configurations in rendering.
    %
    % Properties:
    %   Position     - 3-element vector defining the x, y, and z position of the light source.
    %   Color        - RGB color of the light source as a 1x3 vector.
    %   Intensity    - Light intensity value. If set to -1, the light acts as diffuse lighting.
    %                  Otherwise, intensity is computed using perceptual weights based on color.
    %
    % Methods:
    %   LightSource       - Constructor to initialize the light source's position, color, and type.
    %                       Validates input dimensions and computes intensity based on LightType.
    %   set.Color         - Setter for the Color property with validation for size [1,3].
    %   set.Position      - Setter for the Position property with validation for size [1,3].
    %
    % Usage:
    %   To create a LightSource, specify the position, color, and type (Spotlight or Diffuse).
    %   - Spotlight: Point light with attenuation based on distance.
    %   - Diffuse: Constant intensity light that does not diminish with distance.
    %
    % Example:
    %   pos = [0, 10, 5];
    %   color = [1, 1, 1];
    %   light = LightSource(pos, color, LightType.Spotlight);
    %
    % Notes:
    %   - For a spotlight, Intensity is set based on color luminance using perceptual weights.
    %   - Ensure Position and Color are 3-element vectors, or an error will be raised.
    %
    % Supported Light Types (LightType enumeration):
    %   - Spotlight: Attenuated light based on distance.
    %   - Diffuse: Constant intensity light with no distance-based attenuation.
    
    properties(GetAccess='public')
        Position;
        Color;
        % if set to -1, the LightSource defines a diffuse light
        Intensity;
    end
    methods
        % construct
        function obj=LightSource(pos, col, type)
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
            % compute intensity based on perceptual weights
            if (type == LightType.Spotlight)
                obj.Intensity = -1;
            elseif (type == LightType.Diffuse)
                obj.Intensity = 0.2126 * obj.Color(1) + 0.7152 * obj.Color(2) + 0.0722 * obj.Color(3);
            else
               error('Please specify LightType');
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