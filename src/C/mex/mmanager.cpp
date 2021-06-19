#include "mex.h"
#include <common/class_handle.hpp>
#include <string>
#include <common/common.h>
#include <helper_math.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <vr/volumeRender.h>

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

/*! \fn Volume make_volume(float *data, cudaExtent& size)
 * 	\brief get the most powerful GPU using CUDA Runtime API Version
 *  taken from https://docs.nvidia.com/cuda/optimus-developer-guide/index.html
 *  \return device id
 */
inline int cutGetMaxGflopsDeviceId() {
  int current_device = 0, sm_per_multiproc = 0;
  int max_compute_perf = 0, max_perf_device = 0;
  int device_count = 0, best_SM_arch = 0;
  int arch_cores_sm[4] = {1, 8, 32, 192};
  cudaDeviceProp deviceProp;

  cudaGetDeviceCount(&device_count);

  // Find the best major SM Architecture GPU device
  while (current_device < device_count) {
    cudaGetDeviceProperties(&deviceProp, current_device);
    if (deviceProp.major > 0 && deviceProp.major < 9999) {
      best_SM_arch = max(best_SM_arch, deviceProp.major);
    }
    current_device++;
  }

  // Find the best CUDA capable GPU device
  current_device = 0;
  while (current_device < device_count) {
    cudaGetDeviceProperties(&deviceProp, current_device);
    if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
      sm_per_multiproc = 1;
    } else if (deviceProp.major <= 3) {
      sm_per_multiproc = arch_cores_sm[deviceProp.major];
    } else { // Device has SM major > 3
      sm_per_multiproc = arch_cores_sm[3];
    }

    int compute_perf = deviceProp.multiProcessorCount * sm_per_multiproc *
                       deviceProp.clockRate;

    if (compute_perf > max_compute_perf) {
      // If we find GPU of SM major > 3, search only these
      if (best_SM_arch > 3) {
        // If device==best_SM_arch, choose this, or else pass
        if (deviceProp.major == best_SM_arch) {
          max_compute_perf = compute_perf;
          max_perf_device = current_device;
        }
      } else {
        max_compute_perf = compute_perf;
        max_perf_device = current_device;
      }
    }
    ++current_device;
  }

  cudaGetDeviceProperties(&deviceProp, max_perf_device);

#ifdef _DEBUG
  printf("\nDevice %d: \"%s\"\n", max_perf_device, deviceProp.name);
  printf("Compute Capability   : %d.%d\n", deviceProp.major, deviceProp.minor);
#endif

  return max_perf_device;
}

// pick the device with highest Gflops/s
void selectBestDevice() {
  // best device is selected automatically
  int devID = cutGetMaxGflopsDeviceId();
  HANDLE_ERROR(cudaSetDevice(devID));

  #ifdef _DEBUG
    cudaDeviceProp deviceProp;
    HANDLE_ERROR(cudaGetDeviceProperties(&deviceProp, devID));
    printf("> Using CUDA device [%d]: %s\n", devID, deviceProp.name);
  #endif
}


// The class that we are interfacing to
class MManager
{
public:
  MManager() {
    this->_ptr = rand() % 100;
  };

  ~MManager() { 
    mexPrintf("Calling destructor\n");
  };

  void copy_to_device() { 
    mexPrintf("Copy to device %d\n", this->_ptr);
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
        
  selectBestDevice();

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
