/*! \file volumeRender.cpp
 * 	\author Raphael Scheible <raphael.scheible@uniklinik-freiburg.de>
 * 	\version 1.0
 * 	\license This project is released under the GNU Affero General Public License, Version 3
 *
 * 	\brief implementation of the methods declared in volumeRender.h
 */

#include <cmath>
#include <cuda_runtime.h>
#include <helper_math.h>
#include <iostream>
#include <vector_functions.h>
#include <volumeRender.h>

//! MatlabVolumeRenderer functions
namespace mvr {

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
 *  \param extent extent of the volume
 *  \return structure of type Volume
 */
Volume make_volume(float *data, cudaExtent &extent) {
  Volume volume;
  volume.extent = extent;
  volume.data = data;

  volume.memory_size =
      extent.width * extent.height * extent.depth * sizeof(VolumeType);

  return volume;
}

// pick the device with highest Gflops/s
void selectBestDevice() {
  // best device is selected automatically
  // int devID = cutGetMaxGflopsDeviceId();
  // HANDLE_ERROR(cudaSetDevice(devID));

// #ifdef _DEBUG
//   cudaDeviceProp deviceProp;
//   HANDLE_ERROR(cudaGetDeviceProperties(&deviceProp, devID));
//   printf("> Using CUDA device [%d]: %s\n", devID, deviceProp.name);
// #endif
}

/*! \fn RenderOptions  initRender(	const uint aWidth, const uint aHeight,
                                                        const float
 aScaleEmission, const float aScaleAbsorption, const float aScaleReflection,
                                                        const float3&
 aElementSizeUm, const float4x3& aRotationMatrix, const float aOpacityThreshold,
                                                        const cudaExtent&
 aVolumeSize)
 * 	\brief computes some properties and selects device on that the render
 computes
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

/*! \fn render(const dim3& block_size, const dim3& grid_size,
               const RenderOptions& options, const Volume& volumeEmission,
               const Volume& volumeAbsorption, const Volume& volumeReflection,
               const float3& color)
 * 	\brief computes some properties and selects device on that the render
 computes
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

#ifdef _DEBUG
  printf("rendering scene..\n");

  // compute memory consumption
  size_t totalMemoryInBytes, curAvailMemoryInBytes;
  CUcontext context;
  CUdevice device;

  cudaGetDevice(&device);
  cuCtxCreate(&context, 0, device); // Create context

  cuMemGetInfo(&curAvailMemoryInBytes, &totalMemoryInBytes);

  printf("Memory after copying files:\n\tTotal Memory: %ld MB, Free Memory: "
         "%ld MB, Used Memory: %ld MB\n",
         totalMemoryInBytes / (1024 * 1024),
         curAvailMemoryInBytes / (1024 * 1024),
         (totalMemoryInBytes - curAvailMemoryInBytes) / (1024 * 1024));

  cuCtxDetach(context); // Destroy context
#endif

  // allocate host memory
  size_t size(aOptions.image_width * aOptions.image_height *
              sizeof(VolumeType) * 3);
  float *readback = (float *)malloc(size);

  // allocate device memory
  float *d_output(NULL);
  HANDLE_ERROR(cudaMalloc((void **)&d_output, size));
  HANDLE_ERROR(cudaMemset(d_output, 0, size));

  const float3 gradientStep = make_float3(1.f / aVolumeEmission.extent.width,
                                          1.f / aVolumeEmission.extent.height,
                                          1.f / aVolumeEmission.extent.depth);

  render_kernel(d_output, block_size, grid_size, aOptions, aColor,
                gradientStep);
  // cutilCheckMsg("Error: render_kernel() execution FAILED");

  HANDLE_ERROR(cudaDeviceSynchronize());
  HANDLE_ERROR(cudaMemcpy(readback, d_output, size, cudaMemcpyDeviceToHost));

#ifdef _DEBUG
  for (int i = 0; i < aOptions.image_width * aOptions.image_height * 3;
       i += 1000) {
    if (readback[i] < 0)

      std::cerr << readback[i] << "!";
  }
#endif

  // free device memory
  cudaFree(d_output);
  freeCudaBuffers();
  cudaDeviceSynchronize();
  
  cudaDeviceReset();

#ifdef _DEBUG
  printf("finished rendering..\n");
#endif

  return readback;
}

// only for test, undocumented
#ifndef MATLAB_MEX_FILE
Volume readVolumeFile(const char *filename, cudaExtent &dim, float3 &dim_mm) {
  Volume data = make_volume(readRawFile(filename), dim);
  for (int i = 0; i < dim.width * dim.height * dim.depth; ++i) {
    data.data[i] = fabsf(data.data[i]);
  }

  return data;
}

float *readRawFile(const char *filename) {
  FILE *fp = fopen(filename, "rb");
  if (!fp) {
    fprintf(stderr, "Error opening file '%s'\n", filename);
    exit(EXIT_FAILURE);
  }

  // obtain file size:
  fseek(fp, 0, SEEK_END);
  long size = ftell(fp);
  rewind(fp);

  void *data = malloc(size);
  size_t read = fread(data, 4, size, fp);
  fclose(fp);

  return (float *)data;
}
#endif
} // namespace mvr
