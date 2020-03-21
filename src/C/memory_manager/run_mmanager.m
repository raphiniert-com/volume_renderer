function run_mmanager()
% Check if the mex exists
dir = fileparts(mfilename('fullpath'));
if ~isequal(fileparts(which('mmanager')), dir)
    % Compile the mex
    cwd = cd(dir);
    cleanup_obj = onCleanup(@() cd(cwd));
    fprintf('Compiling example_mex\n');
    mex mmanager.cpp
end

% Use the example interface
% This is the interface written specifically for the example class
fprintf('Using the mmanager interface\n');
obj = mmanager_interface();
train(obj);
test(obj);
clear obj % Clear calls the delete method

% Use the standard interface
% This interface can be used for any mex interface function using the
% pattern:
%   Construction -    obj = mexfun('new',         ...)
%   Destruction -           mexfun('delete', obj)
%   Other methods - [...] = mexfun('method', obj, ...)
% The standard interface avoids the need to write a specific interface
% class for each mex file.
fprintf('Using the standard interface\n');
obj = mex_interface(str2fun([dir '/mmanager'])); % str2fun allows us to use the full path, so the mex need not be on our path
obj.train();
obj.test();
clear obj % Clear calls the delete method
end