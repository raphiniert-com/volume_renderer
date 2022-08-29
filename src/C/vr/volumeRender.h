/*! \file volumeRender.h
 * 	\author Raphael Scheible <mail@raphiniert.com>
 * 	\version 1.0
 * 	\license This project is released under the GNU Affero General Public License, Version 3
 *
 * 	\brief header file of all the functions of the volume renderer
 */

#include <assert.h>
#include <vr/common.h>
#include <cuda_runtime.h>
#include <utility>
#include <vector_types.h>

#ifndef _VOLUMERENDER_H_
#define _VOLUMERENDER_H_

/*! \namespace vr
 *  \brief namespace for volume renderer
 */
namespace vr {

/*! \var typedef float VolumeDataType
 * 	\brief Type of the volume data
 */
typedef float VolumeDataType;

/*! \struct float4x3 volumeRender.h
 * 	\brief struct of a 4x3 matrix, 4 rows, 3 cols
 */
typedef struct {
  /*! The cols of the matrix */
  float3 m[4];
} float4x3;

/*! \struct Ray volumeRender.h
 * 	\brief a simple ray
 */
struct Ray {
  /*! The origin of the ray */
  float3 origin;
  /*! The direction of the ray */
  float3 direction;
};

/*! \struct LightSource volumeRender.h
 * 	\brief a simple lightsource
 */
struct LightSource {
  /*! The light source position */
  float3 position;
  /*! The color intensity of the light */
  float3 color;
};

LightSource make_lightSource(float3 pos, float3 color);

/*! \struct Volume volumeRender.h
 * 	\brief a simple volume
 */
struct Volume {
  /*! The raw data of the volume */
  float *data;
  /*! The volume extent */
  cudaExtent extent;
  /*! The memory size the volume uses */
  size_t memory_size;
  /*! timestamp of last data change */
  uint64_t last_update;
};

Volume make_volume(float *data, uint64_t last_update, cudaExtent &extent);

bool operator==(const Volume &a, const Volume &b);
bool operator!=(const Volume &a, const Volume &b);

/*! \struct RenderOptions volumeRender.h
 * 	\brief The struct contains all rendering options for the raycasting on
 * the device
 */
struct RenderOptions {
  /*! The width of the rendered image */
  size_t image_width;
  /*! The height of the rendered image */
  size_t image_height;

  /*! A value the emission samples are scaled by */
  float scale_emission;
  /*! A value the absorption samples are scaled by */
  float scale_absorption;
  /*! A value the reflection samples are scaled by */
  float scale_reflection;

  /*! min extent of the intersection box */
  float3 boxmin;
  /*! max extent of the intersection box */
  float3 boxmax;

  /*! extent of the element size in um */
  float3 element_size_um;
  /*! Rotation matrix that is applied to the scene
  last vector: [camera x-offset; focal length; object distance] */
  float4x3 rotation_matrix; // last vector: [camera x-offset; focal length;
                            // object distance]
                            /*! The opacity threshold of the raycasting */
  float opacity_threshold;
  /*! The stepsize of the raycasting */
  float tstep;
};

RenderOptions
initRender(const size_t aWidth, const size_t aHeight, const float aScaleEmission,
           const float aScaleReflection, const float aScaleAbsorption,
           const float3 &aElementSizeUm, const float4x3 &aRotationMatrix,
           const float aOpacityThreshold, const cudaExtent &aVolumeSize);

float *render(const dim3 &block_size, const dim3 &grid_size,
              const RenderOptions &aOptions, const cudaExtent &aVolumeExtent,
              const float3 &aColor);

void render_kernel(float *d_output, const dim3 &block_size,
                   const dim3 &grid_size, const RenderOptions &options,
                   const float3 &volume_color, const float3 &aGradientStep);

void copyLightSources(const LightSource *lightSources, const size_t count);

cudaArray * setIlluminationTexture(const Volume &volume, cudaArray * ptr, const uint64_t timeLastMemSync);

void setGradientTextures(const Volume &dx, const Volume &dy, const Volume &dz, 
  cudaArray * &ptr_d_volumeDx, cudaArray * &ptr_d_volumeDy, cudaArray * &ptr_d_volumeDz, const uint64_t timeLastMemSync);

size_t iDivUp(size_t a, size_t b);

inline int cutGetMaxGflopsDeviceId();

Volume mxMake_volume(const mxArray *mxVolume);

/*! \enum GradientMethod
 * 	\brief possible gradient computation methods
 */
enum GradientMethod : int {
  gradientCompute = 0, /*!< gradient computation on the fly */
  gradientLookup = 1   /*!< use LUT to estimate gradient */
};

/*! \enum VolumeType
 * 	\brief possible volume types, which are used for memory management
 */
enum VolumeType : int {
    emission   = 0, 
    absorption = 1, 
    reflection = 2, 
    dx = 3, 
    dy = 4, 
    dz = 5,
    light = 6
};

void freeCudaGradientBuffers(
  cudaArray * d_aGradientXArray,
  cudaArray * d_aGradientYArray,
  cudaArray * d_aGradientZArray
);

void setGradientMethod(const GradientMethod aMethod);

void syncWithDevice(const Volume &aVolumeEmission, const Volume &aVolumeAbsorption,
                    const Volume &aVolumeReflection, const uint64_t timeLastMemSync,
                    cudaArray *  &d_aVolumeEmission, cudaArray *  &d_aVolumeAbsorption, 
                    cudaArray *  &d_aVolumeReflection);

}; // namespace vr

#endif // _VOLUMERENDER_H_
