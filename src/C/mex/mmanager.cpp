#include "mex.h"
#include "class_handle.hpp"
#include <string>

// Copyright (c) 2020, Raphael Scheible
// Copyright (c) 2018, Oliver Woodford

// All rights reserved.

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:

//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in
//       the documentation and/or other materials provided with the distribution

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.


// The class that we are interfacing to
class MManager
{
public:
  MManager() { 
    
  };

  ~MManager() { 
    mexPrintf("Calling destructor\n");
  };

  void copy_to_device() { 
    mexPrintf("Copy to device\n");
  };

  void reset_device() { 
    mexPrintf("Reset device\n");
  };

  void delete_from_device() { 
    mexPrintf("Delete from device\n");
  };

private:
  uint64_t _ptr;

};

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{	
  // Get the command string
  char cmd[64];
	if (nrhs < 1 || mxGetString(prhs[0], cmd, sizeof(cmd)))
		mexErrMsgTxt("First input should be a command string less than 64 characters long.");
        
  // New
  if (!strcmp("new", cmd)) {
    // Check parameters
    if (nlhs != 1)
      mexErrMsgTxt("New: One output expected.");
    // Return a handle to a new C++ instance
    plhs[0] = convertPtr2Mat<MManager>(new MManager);
    return;
  }
  
  // Check there is a second input, which should be the class instance handle
  if (nrhs < 2)
    mexErrMsgTxt("Second input should be a class instance handle.");
  
  // Delete
  if (!strcmp("delete", cmd)) {
    // Destroy the C++ object
    destroyObject<MManager>(prhs[1]);
    // Warn if other commands were ignored
    if (nlhs != 0 || nrhs != 2)
      mexWarnMsgTxt("Delete: Unexpected arguments ignored.");
    return;
  }
  
  // Get the class instance pointer from the second input
  MManager* mmanager_instance = convertMat2Ptr<MManager>(prhs[1]);
  
  // Call the various class methods
  // Train
  if (!strcmp("delete_from_device", cmd)) {
    // Check parameters
    if (nlhs < 0 || nrhs < 2)
      mexErrMsgTxt("delete_from_device: Unexpected arguments.");
    // Call the method
    mmanager_instance->delete_from_device();
    return;
  }
  // Test
  if (!strcmp("copy_to_device", cmd)) {
    // Check parameters
    if (nlhs < 0 || nrhs < 2)
      mexErrMsgTxt("copy_to_device: Unexpected arguments.");
    // Call the method
    mmanager_instance->copy_to_device();
    return;
  }
  
  // Got here, so command not recognized
  mexErrMsgTxt("Command not recognized.");
}
