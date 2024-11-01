classdef LightType
    %LIGHTTYPE Defines types of light sources in the shading model.
    %   This class provides an enumeration of different light types:
    %   - Spotlight: a light with attenuation based on distance.
    %   - Diffuse: a constant intensity light with no attenuation.
    
    enumeration
        Spotlight   % Light with inverse-square attenuation
        Diffuse     % Constant intensity light with no attenuation
    end
end
