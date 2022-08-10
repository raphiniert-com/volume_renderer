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

        %% inc - an example class method call
        function varargout = inc(this, varargin)
            [varargout{1:nargout}] = mmanager('inc', this.objectHandle, varargin{:});
        end

        %% dec - another example class method call
        function varargout = dec(this, varargin)
            [varargout{1:nargout}] = mmanager('dec', this.objectHandle, varargin{:});
        end

        %% print - another example class method call
        function varargout = print(this, varargin)
            [varargout{1:nargout}] = mmanager('print', this.objectHandle, varargin{:});
        end

        %% set - another example class method call
        function varargout = set(this, value, varargin)
            [varargout{1:nargout}] = mmanager('set', this.objectHandle, value, varargin{:});
        end

        %% set - another example class method call
        function varargout = getAddress(this, varargin)
            [varargout{1:nargout}] = mmanager('getAddress', this.objectHandle, this, varargin{:});
        end
    end
end