/*! \file volumeRender.cpp
 * 	\author Raphael Scheible <raphael.scheible@uniklinik-freiburg.de>
 * 	\version 1.0
 * 	\license This project is released under the GNU Affero General Public License, Version 3
 *
 * 	\brief implementation of the methods declared in volumeRender.h
 */

#include <stdint.h>
#include <cmath>
#include <cuda_runtime.h>
#include <helper_math.h>
#include <iostream>
#include <vector_functions.h>
#include <volumeRender.h>

namespace vr {

/*! \fn bool operator==( const Volume& a, const Volume& b )
 * 	\brief compares if the data pointer of two volumes are equal
 *  \param a first volume
 *  \param b another volume
 *  \return (b.data == b.data)
 */
bool operator==(const Volume &a, const Volume &b) { return (a.data == b.data); }

/*! \fn bool operator!=( const Volume& a, const Volume& b )
 * 	\brief compares if the data pointer of two volumes are unequal
 *  \param a first volume
 *  \param b another volume
 *  \return (b.data != b.data)
 */
bool operator!=(const Volume &a, const Volume &b) { return !(a == b); }

/*! \fn int iDivUp(int a, int b)
 *  \brief Round a / b to nearest higher integer value
 *  \param a dividend
 *  \param b divisor
 *  \return a/b rounded to nearest higher int
 */
int iDivUp(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }

/*! \fn LightSource make_lightSource(float3 pos, float3 color)
 * 	\brief constructing a LightSource structure
 *  \param pos light position
 *  \param color color intensity of the light
 *  \return structure of type LightSource
 */
LightSource make_lightSource(float3 pos, float3 color) {
  LightSource result;
  result.position = pos;
  result.color = color;
  return result;
}

/*! \fn Volume make_volume(float *data, cudaExtent& size)
 * 	\brief constructing a Volume structure
 *  \param data raw data of the volume
 *  \param last_update timestamp of last data change
 *  \param extent extent of the volume
 *  \return structure of type Volume
 */
Volume make_volume(float *data, uint64_t last_update, cudaExtent &extent) {
  Volume volume;
  volume.extent = extent;
  volume.last_update = last_update;
  volume.data = data;

  volume.memory_size =
      extent.width * extent.height * extent.depth * sizeof(VolumeType);

  return volume;
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

/*! \fn RenderOptions  initRender(	const uint aWidth,
                                    const uint aHeight,
                                    const float aScaleEmission,
                                    const float aScaleAbsorption,
                                    const float aScaleReflection,
                                    const float3& aElementSizeUm,
                                    const float4x3& aRotationMatrix,
                                    const float aOpacityThreshold,
                                    const cudaExtent& aVolumeSize)
 * 	\brief computes some properties and selects device on that the render computes
 *  \param aWidth width of the rendered image
 *  \param aHeight height of the rendered image
 *  \param aScaleEmission A value the emission samples are scaled by
 *  \param aScaleAbsorption A value the absorption samples are scaled by
 *  \param aScaleReflection A value the reflection samples are scaled by
 *  \param aElementSizeUm extent of the element size in um
 *  \param aRotationMatrix Rotation matrix that is applied to the scene
        last vector: [camera x-offset; focal length; object distance]
 *  \param aOpacityThreshold The opacity threshold of the raycasting
 *  \param aVolumeSize The size/extent of the volume
 *  \return RenderOptions struct
 */
RenderOptions
initRender(const uint aWidth, const uint aHeight, const float aScaleEmission,
           const float aScaleReflection, const float aScaleAbsorption,
           const float3 &aElementSizeUm, const float4x3 &aRotationMatrix,
           const float aOpacityThreshold, const cudaExtent &aVolumeSize) {
  RenderOptions result;
  result.image_width = aWidth;
  result.image_height = aHeight;
  result.element_size_um = aElementSizeUm;

  result.rotation_matrix = aRotationMatrix;
  result.opacity_threshold = aOpacityThreshold;

  result.boxmax =
      make_float3(1.f,
                  (aElementSizeUm.y * aVolumeSize.height) /
                      ((float)aVolumeSize.width * aElementSizeUm.x),
                  (aElementSizeUm.z * aVolumeSize.depth) /
                      ((float)aVolumeSize.width * aElementSizeUm.x));
  result.boxmin = -1 * result.boxmax;

  float3 diagonals = make_float3(sqrtf(aVolumeSize.width * aVolumeSize.width +
                                       aVolumeSize.height * aVolumeSize.height),
                                 sqrtf(aVolumeSize.height * aVolumeSize.height +
                                       aVolumeSize.depth * aVolumeSize.depth),
                                 sqrtf(aVolumeSize.width * aVolumeSize.width +
                                       aVolumeSize.depth * aVolumeSize.depth));

  // find out maximal diagonal (longest way of a ray)
  float maxDiagonal = fminf(diagonals.x, fminf(diagonals.y, diagonals.z));

  // nyquist-shannon
  float freq = 1.f / (2.2f * maxDiagonal);
  result.tstep = freq;

  result.scale_absorption = aScaleAbsorption;
  result.scale_emission = aScaleEmission;
  result.scale_reflection = aScaleReflection;

  result.opacity_threshold = aOpacityThreshold;

  selectBestDevice();

  return result;
}

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

/*! \fn render(const dim3& block_size, const dim3& grid_size,
               const RenderOptions& options, const Volume& volumeEmission,
               const Volume& volumeAbsorption, const Volume& volumeReflection,
               const float3& color)
 * 	\brief computes some properties and selects device on that the render  computes
 *  \param block_size CUDA block size
 * 	\param grid_size CUDA grid size
 *  \param aOptions Options of the rendering process
 *  \param aVolumeEmission emission volume
 *  \param aVolumeAbsorption absorption volume
 * 	\param aVolumeReflection reflection volume
 *  \param aColor the color the rendered volume absorbs
 *  \return pointer to the rendered 2D image
 */
float *render(const dim3 &block_size, const dim3 &grid_size,
              const RenderOptions &aOptions, const Volume &aVolumeEmission,
              const Volume &aVolumeAbsorption, const Volume &aVolumeReflection,
              const float3 &aColor) {
  initCuda(aVolumeEmission, aVolumeAbsorption, aVolumeReflection);
  HANDLE_ERROR(cudaDeviceSynchronize());

#ifdef _DEBUG
  printf("rendering scene..\n");

  // compute memory consumption
  size_t totalMemoryInBytes, curAvailMemoryInBytes;
  CUcontext context;
  CUdevice device;

  // get currently used device
  cudaGetDevice(&device);

  // Create context
  cuCtxCreate(&context, 0, device);

  cuMemGetInfo(&curAvailMemoryInBytes, &totalMemoryInBytes);

  printf("Memory after copying files:\n\tTotal Memory: %ld MB, Free Memory: "
         "%ld MB, Used Memory: %ld MB\n",
         totalMemoryInBytes / (1024 * 1024),
         curAvailMemoryInBytes / (1024 * 1024),
         (totalMemoryInBytes - curAvailMemoryInBytes) / (1024 * 1024));

  // Destroy context
  cuCtxDestroy(context);
#endif

  // allocate host memory
  size_t size(aOptions.image_width * aOptions.image_height *
              sizeof(VolumeType) * 3);
  float *readback = (float *)malloc(size);

  // allocate device memory
  float *d_output(NULL);
  HANDLE_ERROR(cudaMalloc((void **)&d_output, size));
  HANDLE_ERROR(cudaMemset(d_output, 0, size));

  HANDLE_ERROR(cudaDeviceSynchronize());

  const float3 gradientStep = make_float3(1.f / aVolumeEmission.extent.width,
                                          1.f / aVolumeEmission.extent.height,
                                          1.f / aVolumeEmission.extent.depth);

  render_kernel(d_output, block_size, grid_size, aOptions, aColor,
                gradientStep);
  // cutilCheckMsg("Error: render_kernel() execution FAILED");

  HANDLE_ERROR(cudaDeviceSynchronize());
  HANDLE_ERROR(cudaMemcpy(readback, d_output, size, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaDeviceSynchronize());

#ifdef _DEBUG
  for (int i = 0; i < aOptions.image_width * aOptions.image_height * 3;
       i += 1000) {
    if (readback[i] < 0)

      std::cerr << readback[i] << "!";
  }
#endif

  // free device memory
  HANDLE_ERROR(cudaFree(d_output));
  freeCudaBuffers();
  HANDLE_ERROR(cudaDeviceSynchronize());

  HANDLE_ERROR(cudaDeviceReset());

#ifdef _DEBUG
  printf("finished rendering..\n");
#endif

  return readback;
}
} // namespace vr
