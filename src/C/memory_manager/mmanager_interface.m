%EXAMPLE_INTERFACE Example MATLAB class wrapper to an underlying C++ class
classdef mmanager_interface < handle
    properties (SetAccess = private, Hidden = true)
        objectHandle; % Handle to the underlying C++ class instance
    end
    methods
        %% Constructor - Create a new C++ class instance 
        function this = mmanager_interface(varargin)
            this.objectHandle = mmanager('new', varargin{:});
        end
        
        %% Destructor - Destroy the C++ class instance
        function delete(this)
            mmanager('delete', this.objectHandle);
        end

        %% Train - an example class method call
        function varargout = train(this, varargin)
            [varargout{1:nargout}] = mmanager('train', this.objectHandle, varargin{:});
        end

        %% Test - another example class method call
        function varargout = test(this, varargin)
            [varargout{1:nargout}] = mmanager('test', this.objectHandle, varargin{:});
        end
    end
end