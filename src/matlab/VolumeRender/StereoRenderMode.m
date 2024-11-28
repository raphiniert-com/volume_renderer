classdef StereoRenderMode
    % StereoRenderMode
    %
    % This class defines an enumeration for different stereo rendering modes.
    % Stereo rendering modes are used in 3D graphics to create a stereoscopic 
    % visual effect by presenting slightly different images to each eye.
    %
    % Enumerations:
    %   - RedCyan: Represents a stereoscopic rendering mode using red-cyan 
    %              anaglyphs. This mode is compatible with red-cyan 3D glasses.
    %
    %   - LeftRightHorizontal: Represents a stereoscopic rendering mode where 
    %                          the left and right views are displayed side by 
    %                          side in a horizontal layout. This is commonly 
    %                          used for devices or viewers that support side-by-side 3D.
    %
    % Usage:
    %   You can use this enumeration to specify the stereo rendering mode
    %   in applications or functions that support stereoscopic rendering.
    %
    % Example:
    %   % Assign a stereo rendering mode
    %   renderMode = StereoRenderMode.RedCyan;
    %
    %   % Check the selected mode
    %   if renderMode == StereoRenderMode.LeftRightHorizontal
    %       disp('Using Left-Right Horizontal Stereo Mode');
    %   else
    %       disp('Using Red-Cyan Stereo Mode');
    %   end
    
    enumeration
        % Red-Cyan anaglyph rendering mode
        RedCyan

        % Left-right horizontal side-by-side rendering mode
        LeftRightHorizontal
    end
end

