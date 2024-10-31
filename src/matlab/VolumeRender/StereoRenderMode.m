classdef StereoRenderMode
    % STEREORENDERMODE Enumeration class for stereo rendering modes.
    %   This class defines the modes available for stereoscopic rendering
    %   output. It allows users to select between different rendering modes 
    %   for displaying 3D images.
    %
    % Enumeration Members:
    %   RedCyan               - Renders an anaglyph image with red and cyan channels for 3D viewing.
    %   LeftRightHorizontal   - Renders a side-by-side stereo image with left and right views 
    %                           horizontally aligned.
    %
    % Usage:
    %   The StereoRenderMode enumeration can be used to set the rendering
    %   output type for stereoscopic images in applications or classes
    %   that support 3D rendering modes.
    %
    % Example:
    %   % Set stereo mode to RedCyan for anaglyph rendering
    %   renderMode = StereoRenderMode.RedCyan;
    %
    %   % Set stereo mode to LeftRightHorizontal for side-by-side rendering
    %   renderMode = StereoRenderMode.LeftRightHorizontal;
    
    enumeration
        RedCyan
        LeftRightHorizontal
    end
end

