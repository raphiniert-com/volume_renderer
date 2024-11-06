classdef LightType
  % LIGHTTYPE Defines types of light sources in the shading model.
  %   This class provides an enumeration of different light types:
  %   - Attenuated (Spotlight): A light with attenuation based on distance
  %       that follows the inverse-square law, creating a spotlight effect.
  %   - Diffuse: A constant intensity light that illuminates all surfaces
  %       evenly, regardless of distance.
  %   - Ambient: Provides a base level of illumination that ensures all
  %       objects are visible, without directional shading.

  enumeration
    Attenuated   % Light with inverse-square attenuation (e.g., Spotlight or Point Light)
    Diffuse      % Constant intensity light with no attenuation
    Ambient      % Non-directional light contributing base illumination
  end
end