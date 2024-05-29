/*! \file volumeRender_kernel.cu
 * 	\author Raphael Scheible <mail@raphiniert.com>
 * 	\version 1.0
 * 	\license This project is released under the GNU Affero General Public License, Version 3
 *
 * 	\brief CUDA C file with all the device functions
 */

#ifndef _VOLUMERENDER_KERNEL_CU_
#define _VOLUMERENDER_KERNEL_CU_

#include <float.h>
#include <helper_math.h>
#include <math.h>
#include <stdio.h>
#include <vector>
#include <vr/volumeRender.h>


#define ONE_OVER_2PI ((float)0.1591549430918953357688837633725143620344596457404564)
#define PI2 ((float)6.2831853071795864769252867665590057683943387987502116)

/*! \var typedef unsigned int uint
 * 	\brief defines abbrev for unsigned int: uint
 */
typedef unsigned int uint;

/*! \var typedef unsigned char uchar
 * 	\brief  defines abbrev for unsigned char: uchar
 */
typedef unsigned char uchar;

/*! \var typedef float3 (*gradientFunction)(const float3, const float3,
 * 				const float3, const float3, const float3, const float3) 
 *  \brief function pointer to gradientFunction that returns a gradient
 */
typedef float3 (*gradientFunction)(const float3, const float3, const float3,
                                   const float3, const float3, const float3);

// forward declaration
__device__ float3 computeGradient(const float3, const float3, const float3,
                                  const float3, const float3, const float3);

__device__ float3 lookupGradient(const float3, const float3, const float3,
                                 const float3, const float3, const float3);

/*! \var __device__ gradientFunction gradient_functions[2] = { computeGradient, lookupGradient }; 
 *  \brief Contains function pointer of possible lookup functions
 */
__device__ gradientFunction gradient_functions[2] = {computeGradient,
                                                     lookupGradient};

/*! \var __device__ __constant__ GradientMethod dc_activeGradientMethod
 * 	\brief current chosen gradient Method. Default value is gradientCompute.
 */
__device__ __constant__ vr::GradientMethod dc_activeGradientMethod = vr::gradientCompute;

/*! \var vr::LightSource *d_lightSources
 * 	\brief device array of lightsources
 */
vr::LightSource *d_lightSources = NULL;

/*! \var __device__ __constant__ size_t c_numLightSources
 * 	\brief device variable storing number of lightsources
 */
__device__ __constant__ size_t c_numLightSources;

/*! \var const cudaChannelFormatDesc channelDesc
 * 	\brief channel desc for textures
 */
const cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<vr::VolumeDataType>();

/*! \var __device__ vr::VolumeType d_idxEmmission
 * 	\brief id for the emission texture
 */
__device__ vr::VolumeType d_idxEmmission = vr::VolumeType::emission;

/*! \var __device__ vr::VolumeType d_idxAbsorption
 * 	\brief id for the absorption texture
 */
__device__ vr::VolumeType d_idxAbsorption = vr::VolumeType::emission;

/*! \var __device__ vr::VolumeType d_idxReflection
 * 	\brief id for the reflection texture
 */
__device__ vr::VolumeType d_idxReflection = vr::VolumeType::reflection;

/*! \var texture<vr::VolumeDataType, cudaTextureType3D, cudaReadModeElementType> tex_emission 
 * \brief 3D texture for emission lookup
 */
texture<vr::VolumeDataType, cudaTextureType3D, cudaReadModeElementType> tex_emission;

/*! \var texture<vr::VolumeDataType, cudaTextureType3D, cudaReadModeElementType> tex_gradientX 
 *  \brief 3D texture of gradient in x direction used in lookupGradient
 */
texture<vr::VolumeDataType, cudaTextureType3D, cudaReadModeElementType> tex_gradientX;

/*! \var texture<vr::VolumeDataType, cudaTextureType3D, cudaReadModeElementType> tex_gradientY 
 * \brief 3D texture of gradient in y direction used in lookupGradient
 */
texture<vr::VolumeDataType, cudaTextureType3D, cudaReadModeElementType> tex_gradientY;

/*! \var texture<vr::VolumeDataType, cudaTextureType3D, cudaReadModeElementType> tex_gradientZ 
 *  \brief 3D texture of gradient in z direction used in lookupGradient
 */
texture<vr::VolumeDataType, cudaTextureType3D, cudaReadModeElementType> tex_gradientZ;

/*! \var texture<vr::VolumeDataType, cudaTextureType3D, cudaReadModeElementType> tex_absorption 
 * \brief 3D texture for absorption lookup
 */
texture<vr::VolumeDataType, cudaTextureType3D, cudaReadModeElementType> tex_absorption;

/*! \var texture<vr::VolumeDataType, cudaTextureType3D, cudaReadModeElementType> tex_reflection 
 *  \brief 3D texture for reflection lookup
 */
texture<vr::VolumeDataType, cudaTextureType3D, cudaReadModeElementType> tex_reflection;

/*! \var texture<vr::VolumeDataType, cudaTextureType3D, cudaReadModeElementType> tex_illumination 
 *  \brief 3D texture for illumination lookup
 */
texture<vr::VolumeDataType, cudaTextureType3D, cudaReadModeElementType> tex_illumination;

/*! \var __device__ texture<vr::VolumeDataType, cudaTextureType3D, cudaReadModeElementType> getTexture(vr::VolumeType aType)
 *  \brief 3D texture for illumination lookup
 *  \param aType volume type id from type vr::VolumeType
 *  \return texture given an id
 */
__device__ texture<vr::VolumeDataType, cudaTextureType3D, cudaReadModeElementType> getTexture(vr::VolumeType aType) {
  switch (aType) {
    case vr::VolumeType::emission:
      return tex_emission;
    case vr::VolumeType::absorption:
      return tex_absorption;
    case vr::VolumeType::reflection:
      return tex_reflection;
    case vr::VolumeType::dx:
      return tex_gradientX;
    case vr::VolumeType::dy:
      return tex_gradientY;
    case vr::VolumeType::dz:
      return tex_gradientZ;
    default:
      return tex_emission;
  }
}

/*! \fn int intersectBox(Ray aRay, float3 aBoxmin, float3 aBoxmax, float *aTnear, float *aTfar) 
 * \brief Intersect ray with a box. (see https://doi.org/10.1080/2151237X.2005.10129188) 
 * \param aRay ray tested for intersection. 
 * \param aBoxmin min box coordinates. 
 * \param aBoxmax max box coordinates. 
 * \param aTnear tnear plane. 
 * \param aTfar tfar plane. 
 * \return 1 if the intersects the box, 0 if not
 */
__forceinline__ __device__ int intersectBox(vr::Ray aRay, float3 aBoxmin,
                                            float3 aBoxmax, float *aTnear,
                                            float *aTfar) {
  int sign[3];
  float3 parameters[2] = {aBoxmin, aBoxmax};
  float3 inv_direction = 1.f / aRay.direction;
  float3 origin = aRay.origin;

  sign[0] = (inv_direction.x < 0);
  sign[1] = (inv_direction.y < 0);
  sign[2] = (inv_direction.z < 0);

  // intersection computation
  float tmin, tmax, tymin, tymax, tzmin, tzmax;

  tmin = (parameters[sign[0]].x - origin.x) * inv_direction.x;
  tmax = (parameters[1 - sign[0]].x - origin.x) * inv_direction.x;
  tymin = (parameters[sign[1]].y - origin.y) * inv_direction.y;
  tymax = (parameters[1 - sign[1]].y - origin.y) * inv_direction.y;

  if ((tmin > tymax) || (tymin > tmax))
    return false;

  if (tymin > tmin)
    tmin = tymin;

  if (tymax < tmax)
    tmax = tymax;

  tzmin = (parameters[sign[2]].z - origin.z) * inv_direction.z;
  tzmax = (parameters[1 - sign[2]].z - origin.z) * inv_direction.z;
  if ((tmin > tzmax) || (tzmin > tmax))
    return false;

  if (tzmin > tmin)
    tmin = tzmin;

  if (tzmax < tmax)
    tmax = tzmax;

  *aTnear = tmin;
  *aTfar = tmax;

  return true;
}

/*! \fn float3 computeGradient(const float3, const float3, const float3,
                                const float3, const float3, const float3)
 *  \brief computes gradient using central differences
 *  \param aSamplePosition position for the texture lookup [unused]
 *  \param aPosition ray position in world coordinate
 * 	\param aStep step size to a neighbor voxel
 * 	\param aBoxmin min extents of the intersection box
 * 	\param aBoxmin max extents of the intersection box
 * 	\param aBoxScale 1 devided by size of the box
 * 	\return gradient
 */
__device__ float3 computeGradient(const float3 aSamplePosition,
                                  const float3 aPosition, const float3 aStep,
                                  const float3 aBoxmin, const float3 aBoxmax,
                                  const float3 aBoxScale) {
  float3 gradient;
  float3 samplePosition1, samplePosition2;

  // normal computation using central differences
  samplePosition1 =
      (aPosition + make_float3(aStep.x, 0, 0) - aBoxmin) * (aBoxScale);
  samplePosition2 =
      (aPosition - make_float3(aStep.x, 0, 0) - aBoxmin) * (aBoxScale);

  gradient.x = tex3D(tex_emission, samplePosition1.x, samplePosition1.y,
                     samplePosition1.z) -
               tex3D(tex_emission, samplePosition2.x, samplePosition2.y,
                     samplePosition2.z);

  samplePosition1 =
      (aPosition + make_float3(0, aStep.y, 0) - aBoxmin) * (aBoxScale);
  samplePosition2 =
      (aPosition - make_float3(0, aStep.y, 0) - aBoxmin) * (aBoxScale);

  gradient.y = tex3D(tex_emission, samplePosition1.x, samplePosition1.y,
                     samplePosition1.z) -
               tex3D(tex_emission, samplePosition2.x, samplePosition2.y,
                     samplePosition2.z);

  samplePosition1 =
      (aPosition + make_float3(0, 0, aStep.z) - aBoxmin) * (aBoxScale);
  samplePosition2 =
      (aPosition - make_float3(0, 0, aStep.z) - aBoxmin) * (aBoxScale);

  gradient.z = tex3D(tex_emission, samplePosition1.x, samplePosition1.y,
                     samplePosition1.z) -
               tex3D(tex_emission, samplePosition2.x, samplePosition2.y,
                     samplePosition2.z);

  gradient = gradient * make_float3(0.5f);

  return gradient;
}

/*! \fn float3 lookupGradient (const float3, const float3, const float3,
                               const float3, const float3, const float3)
 *  \brief determines the gradient via texture lookup in gradient textures
 *  \param aSamplePosition position for the texture lookup
 *  \param aPosition ray position in world coordinate [unused]
 * 	\param aStep step size to a neighbor voxel [unused]
 * 	\param aBoxmin min extents of the intersection box [unused]
 * 	\param aBoxmin max extents of the intersection box [unused]
 * 	\param aBoxScale 1 devided by size of the box [unused]
 * 	\return gradient
 */
__device__ float3 lookupGradient(const float3 aSamplePosition,
                                 const float3 aPosition, const float3 aStep,
                                 const float3 aBoxmin, const float3 aBoxmax,
                                 const float3 aBoxScale) {
  return make_float3(tex3D(tex_gradientX, aSamplePosition.x, aSamplePosition.y,
                           aSamplePosition.z),
                     tex3D(tex_gradientY, aSamplePosition.x, aSamplePosition.y,
                           aSamplePosition.z),
                     tex3D(tex_gradientZ, aSamplePosition.x, aSamplePosition.y,
                           aSamplePosition.z));
}

/*! \fn float angle(const float3& a, const float3& b)
 *  \brief computes the angle of two vectors
 *  \param a vector a
 *  \param b vector b
 * 	\return angle between a and b
 */
__forceinline__ __device__ float angle(const float3 &a, const float3 &b) {
  // radian to degree
  return acos(dot(a, b) / (length(a) + length(b)));
}

/*! \fn float3 shade(const float3& aSamplePosition, const float3 aPosition,
                     const float3 aGradientStep, const float3 aViewPosition,
                     const float3 aColor, vr::LightSource * aLightSources, 
                     const float aFactorReflection, const float3 aBoxmin, 
                     const float3 aBoxmax, const float3 aBoxScale)
 *  \brief determines the light performed at a voxelposition of all defined lightsources
 * 			   depending on the undelying illumination texture/model
 *  \param aSamplePosition
 *  \param aPosition
 * 	\param aGradientStep step size to a neighbor voxel
 * 	\param aViewPosition position of the viewer
 * 	\param aColor the color the rendered volume absorbs
 * 	\param aLightSources pointer to all light sources
 * 	\param aFactorReflection factors the sampled value of reflection
 * 	\param aBoxmin min extents of the intersection box
 * 	\param aBoxmin max extents of the intersection box
 * 	\param aBoxScale 1 devided by size of the box
 *  \return voxel value
 */
__device__ float3 shade(const float3 &aSamplePosition, const float3 aPosition,
                        const float3 aGradientStep, const float3 aViewPosition,
                        const float3 aColor, vr::LightSource *aLightSources,
                        const float aFactorReflection, const float3 aBoxmin,
                        const float3 aBoxmax, const float3 aBoxScale) {
  const float factorReflection = aFactorReflection;

  // negativ gradient approx surface normal
  const float3 surfaceNormal =
      -1 * normalize((gradient_functions[dc_activeGradientMethod])(
               aSamplePosition, aPosition, aGradientStep, aBoxmin, aBoxmax,
               aBoxScale));

  float3 result = make_float3(0.f);

  for (size_t i = 0; i < c_numLightSources; ++i) {
    vr::LightSource lightSource = aLightSources[i];

    // calculation of angles
    float3 lightPosition = (lightSource.position);
    float alpha = angle(surfaceNormal, lightPosition) / PI2 *
                  ONE_OVER_2PI; // normalizing to [0,1]
    float beta = angle(surfaceNormal, aViewPosition) / PI2 * ONE_OVER_2PI;

    float3 lightOut = (lightPosition - aPosition);
    float3 lightIn = (aViewPosition - aPosition);

    // dot( , ) here: scalar projection of lightOut/lightIn onto surfaceNormal
    float3 lightOutProj =
        lightPosition - (dot(lightOut, surfaceNormal) * surfaceNormal);
    float3 lightInProj =
        aViewPosition - (dot(lightIn, surfaceNormal) * surfaceNormal);
    float gamma = angle(lightInProj, lightOutProj) * ONE_OVER_2PI;

    // lookup in tex_reflection
    float reflection =
        factorReflection * tex3D(getTexture(d_idxReflection), aSamplePosition.x,
                                aSamplePosition.y, aSamplePosition.z);

    float light = tex3D(tex_illumination, alpha, beta, gamma);

    // consider light absorption (color of material)
    result += reflection * light * lightSource.color * aColor;
  }

  return result;
}

/*! \fn void d_render(float *d_aOutput,  const vr::RenderOptions aOptions,
         const float3 aColor, const vr::LightSource * aLightSources,
         const float3 aGradientStep)
 *  \brief performs raycasting on the device
 *  \param d_aOutput device pointer of the computed 2D output
 *  \param aOptions options of the rendering process
 * 	\param aColor the color the rendered volume absorbs
 * 	\param aLightSources pointer to all light sources
 * 	\param aGradientStep step size to a neighbor voxel
 */
__global__ void d_render(float *d_aOutput, const vr::RenderOptions aOptions,
                         const float3 aColor,
                         const vr::LightSource *aLightSources,
                         const float3 aGradientStep) {
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  // return if out of image bound
  if ((x >= aOptions.image_width) || (y >= aOptions.image_height))
    return;

  vr::LightSource *lightSources =
      const_cast<vr::LightSource *>(aLightSources);
  const float tstep = aOptions.tstep;
  const float opacityThreshold = aOptions.opacity_threshold;
  const float factorAbsorption = aOptions.factor_absorption;
  const float factorEmission = aOptions.factor_emission;
  const float factorReflection = aOptions.factor_reflection;

  const float3 boxMin = aOptions.boxmin;
  const float3 boxMax = aOptions.boxmax;

  // 2D image plane in [-1,1]
  float u = (x / (float)aOptions.image_width) * 2.0f - 1.0f;
  float ratio = aOptions.image_height / (float)aOptions.image_width;
  float v = (y / (float)aOptions.image_height) * 2.0f * ratio - 1.0f * ratio;

  // calculate eye ray in world space
  vr::Ray eyeRay;

  // box factor
  const float3 boxScale = 1.f / (boxMax - boxMin);

  // translate into factor
  const float cameraXOffset = aOptions.rotation_matrix.m[3].x;
  const float focalLength = aOptions.rotation_matrix.m[3].y;
  const float objectDistance = aOptions.rotation_matrix.m[3].z;

  // in case of 3D rendering we have an x offset [Off-axis]
  const float3 xVector = aOptions.rotation_matrix.m[0];
  const float3 yVector = aOptions.rotation_matrix.m[1];
  const float3 zVector = aOptions.rotation_matrix.m[2];
  const float3 vCameraOffset = (cameraXOffset * xVector);

  // Ray properties
  eyeRay.origin = vCameraOffset + (-1 * objectDistance * zVector);

  eyeRay.direction =
      normalize(u * normalize(xVector) + v * yVector + focalLength * zVector);

  // find intersection with box
  float tnear(0), tfar(0);
  int hit = intersectBox(eyeRay, boxMin, boxMax, &tnear, &tfar);

  // if (hit)
  //    printf("near: %f, far: %f\n", tnear, tfar);

  if (!hit)
    return;
  if (tnear < 0.0f)
    tnear = 0.0f; // clamp to near plane

  // march along ray from front to back, accumulating color
  float4 sum = make_float4(0.0f);
  float t = tnear;
  float3 pos = eyeRay.origin + eyeRay.direction * tnear;
  float3 step = eyeRay.direction * tstep;

  // map step to [0, 1] coordinates
  float3 pos_sample_old = make_float3(0.f);
  while (true) {
    // map position to [0, 1] coordinates
    float3 pos_sample = (pos - boxMin) * boxScale;

    // ################
    // ### sampling ###
    // ################

    // read from 3D texture and apply several factor factor
    float emission = factorEmission * tex3D(getTexture(d_idxEmmission), pos_sample.x,
                                           pos_sample.y, pos_sample.z);
    float absorption = factorAbsorption * tex3D(getTexture(d_idxAbsorption), pos_sample.x,
                                               pos_sample.y, pos_sample.z);

    float3 sample = make_float3(emission);

    // ###############################
    // ### illumination & Coloring ###
    // ###############################

    float dx = tstep;
    float alpha = 1 - __expf(-absorption * dx);

    // apply color
    float ds = tstep;
    float3 colored = sample * ds * aColor;

    float3 illumination =
        shade(pos_sample, pos, aGradientStep, eyeRay.origin, aColor,
              lightSources, factorReflection, boxMin, boxMax, boxScale);

    float3 illuminated = colored + illumination;

    float4 shaded =
        make_float4(illuminated.x, illuminated.y, illuminated.z, alpha);

    // ###################
    // ### compositing ###
    // ###################

    // alpha-blending
    // pre-multiply alpha
    shaded.x *= shaded.w;
    shaded.y *= shaded.w;
    shaded.z *= shaded.w;

    // "under" operator for front-to-back blending
    sum = (1 - sum.w) * (shaded) + sum;

    // exit early if opaque
    if (sum.w > opacityThreshold)
      break;

    t += tstep;
    if (t > tfar)
      break;

    pos += step;
  }

  // write in image structure
  uint size = aOptions.image_width * aOptions.image_height;

  // linear matlab conform memory layout (column-major)
  // descibed on:
  // https://eli.thegreenplace.net/2015/memory-layout-of-multi-dimensional-arrays/
  uint k = x * aOptions.image_height + y;

  // write output in RBG
  d_aOutput[k] = sum.x;
  d_aOutput[k + size] = sum.y;
  d_aOutput[k + size * 2] = sum.z;
}


namespace vr {

/*! \fn cudaArray * createTextureFromVolume(
                    texture<VolumeDataType, cudaTextureType3D, cudaReadModeElementType> &aTex,
                    const Volume &aVolume, cudaArray *d_aArray, const bool aAllocateMemory) 
 *  \brief creates a texture based on a Volume
 *  \param aTex texture that is used to perform lookups
 *  \param aVolume volume that is copied to the device
 * 	\param d_aArray device array where the data are stored in/copied to
 *  \param aAllocateMemory if set to true, memory will be allocated and data copied from host to device
 * 	\return device pointer of the device array
 */
cudaArray * createTextureFromVolume(
    texture<VolumeDataType, cudaTextureType3D, cudaReadModeElementType> &aTex,
    const Volume &aVolume, cudaArray *d_aArray, const bool aAllocateMemory) {
  // if volume was refreshed or first render
  // bool allocateMemory = (aVolume.last_update > aTimeLastMemSync) || (aTimeLastMemSync == 0);

  // only allocate memory and copy data to GPU if required
  if (d_aArray == 0 || aAllocateMemory) {
    HANDLE_ERROR(cudaMalloc3DArray(&d_aArray, &channelDesc, aVolume.extent));

    // copy data to 3D d_array
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr = make_cudaPitchedPtr(
      aVolume.data, aVolume.extent.width * sizeof(VolumeDataType),
      aVolume.extent.width, aVolume.extent.height);
    copyParams.dstArray = d_aArray;
    copyParams.extent = aVolume.extent;
    copyParams.kind = cudaMemcpyHostToDevice;
    HANDLE_ERROR(cudaMemcpy3D(&copyParams));
  }

  // set texture parameters
  aTex.normalized = true; // access with normalized texture coordinates
  aTex.filterMode = cudaFilterModeLinear;     // linear interpolation
  aTex.addressMode[0] = cudaAddressModeClamp; // clamp texture coordinates
  aTex.addressMode[1] = cudaAddressModeClamp;
  aTex.addressMode[2] = cudaAddressModeClamp;

  // bind d_aArray to 3D texture
  HANDLE_ERROR(cudaBindTextureToArray(aTex, d_aArray, channelDesc));

  return d_aArray;
}

/*! \fn void freeCudaGradientBuffers(
              cudaArray * d_aGradientXArray,
              cudaArray * d_aGradientYArray,
              cudaArray * d_aGradientZArray
            )
 *  \param d_aGradientXArray device memory address to the x-gradient
 *  \param d_aGradientYArray device memory address to the y-gradient
 *  \param d_aGradientZArray device memory address to the z-gradient
 *  \brief removes the gradient volumes from the device memory
 */
void freeCudaGradientBuffers(
  cudaArray * d_aGradientXArray,
  cudaArray * d_aGradientYArray,
  cudaArray * d_aGradientZArray
) {
  HANDLE_ERROR(cudaFreeArray(d_aGradientXArray));
  HANDLE_ERROR(cudaFreeArray(d_aGradientYArray));
  HANDLE_ERROR(cudaFreeArray(d_aGradientZArray));
}

/*! \fn void render_kernel(float* d_aOutput, const dim3& block_size,
                           const dim3& grid_size, const RenderOptions& aOptions,
                           const float3& aColor, const float3& aGradientStep)
 *  \brief starts the ray casting on the device
 * 	\param d_aOutput device pointer of the computed 2D output
 *  \param block_size CUDA block size
 * 	\param grid_size CUDA grid size
 *  \param aOptions options of the rendering process
 * 	\param aColor the color the rendered volume absorbs
 * 	\param aGradientStep step size to a neighbor voxel
 */
void render_kernel(float *d_aOutput, const dim3 &block_size,
                   const dim3 &grid_size, const RenderOptions &aOptions,
                   const float3 &aColor, const float3 &aGradientStep) {
  d_render<<<grid_size, block_size>>>(d_aOutput, aOptions, aColor,
                                      d_lightSources, aGradientStep);
}

/*! \fn void copyLightSources(const LightSource *aLightSources,
                      const size_t aNumOfLightSources)
 *  \brief copy light sources to device
 * 	\param aLightSources pointer to all light sources
 * 	\param aNumOfLightSources number of light sources
 */
void copyLightSources(const LightSource *aLightSources,
                      const size_t aNumOfLightSources) {
  size_t size(aNumOfLightSources * sizeof(LightSource));
  HANDLE_ERROR(cudaMalloc((void **)&d_lightSources, size));
  HANDLE_ERROR(
      cudaMemcpy(d_lightSources, aLightSources, size, cudaMemcpyHostToDevice));

  HANDLE_ERROR(cudaMemcpyToSymbol(c_numLightSources, &aNumOfLightSources,
                                  sizeof(size_t)));
}

/*! \fn void setGradientMethod(const vr::GradientMethod)
 *  \brief set gradientMethod
 * 	\param aMethod kind of gradient method used while rendering
 */
void setGradientMethod(const vr::GradientMethod aMethod) {
  HANDLE_ERROR(
    cudaMemcpyToSymbol(dc_activeGradientMethod, &aMethod, sizeof(enum vr::GradientMethod))
  );
}

/*! \fn cudaArray * referenceTexture(
            texture<vr::VolumeDataType, cudaTextureType3D, cudaReadModeElementType>& aTexture, 
            cudaArray* d_aArray, const vr::VolumeType& d_aIdx, const vr::VolumeType aTypeAssigned)
 *  \brief reference a texture, using the mechanism undelying getTexture
 *  \param aTexture texture which should be re-referenced
 * 	\param d_aArray array pointing to the device memory undelying the texture
 *  \param d_aIdx volume type id the texture has
 * 	\param aTypeAssigned the volume type id the texture should refer to
 *  \return device address of texture
 */
cudaArray * referenceTexture(
    texture<vr::VolumeDataType, cudaTextureType3D, cudaReadModeElementType>& aTexture, 
    cudaArray* d_aArray, const vr::VolumeType& d_aIdx, const vr::VolumeType aTypeAssigned) {
  
  // clear memory
  if (d_aArray != 0) {
    HANDLE_ERROR(cudaUnbindTexture(aTexture));
    d_aArray = 0;
  }
  
  // just reference to the same device variable
  // HANDLE_ERROR(cudaBindTextureToArray(aTexture, d_aArrayRef, channelDesc));
  HANDLE_ERROR(
    cudaMemcpyToSymbol(d_aIdx, &aTypeAssigned, sizeof(enum vr::VolumeType))
  );

  return d_aArray;
}

/*! \fn cudaArray * syncVolume(
          texture<vr::VolumeDataType, cudaTextureType3D, cudaReadModeElementType>& aTexture,
          cudaArray* &d_aArray, const Volume& aVolume, const bool aAllocateMemory)
 *  \brief sync volume with device (if required) and setup texture
 * 	\param aTexture texture which should be synced
 * 	\param d_aArray array pointing to the device memory undelying the texture
 * 	\param aVolume volume data which should be synched onto the device
 *  \param aAllocateMemory if set to true, memory will be allocated and data copied from host to device
 */
cudaArray * syncVolume(
    texture<vr::VolumeDataType, cudaTextureType3D, cudaReadModeElementType>& aTexture,
    cudaArray* d_aArray, const Volume& aVolume, const bool aAllocateMemory) {
  // clear memory
  if (d_aArray != 0 || aAllocateMemory) {
    HANDLE_ERROR(cudaUnbindTexture(aTexture));
    HANDLE_ERROR(cudaFreeArray(d_aArray));
    d_aArray = 0;
  }

  d_aArray = createTextureFromVolume(aTexture, aVolume, d_aArray, aAllocateMemory);

  return d_aArray;
}

/*! \fn cudaArray* setIlluminationTexture(const Volume &aVolume, 
                                          cudaArray * d_aIllumination, 
                                          const uint64_t aTimeLastMemSync)
 *  \brief copies illumination volume from host to device
 *  \param aVolume illumination volume
 *  \param d_aIllumination array pointing to the device memory
 *  \param aTimeLastMemSync timestamp on which the last rendering took place
 */
cudaArray * setIlluminationTexture(const Volume &aVolume, 
                                   cudaArray * d_aIllumination, 
                                   const uint64_t aTimeLastMemSync) {

  bool allocateMemory = (aVolume.last_update > aTimeLastMemSync) || (aTimeLastMemSync == 0);
  
  return syncVolume(tex_illumination, d_aIllumination, aVolume, allocateMemory);
}

/*! \fn setGradientTextures(const Volume &aDx, 
                         const Volume &aDy,
                         const Volume &aDz, 
                         cudaArray * &ptr_d_volumeDx,
                         cudaArray * &ptr_d_volumeDy,
                         cudaArray * &ptr_d_volumeDz,
                         const uint64_t aTimeLastMemSync)
 *  \brief copies gradient volumes from host to device 
 *  \param aDx volume of gradient in x direction 
 *  \param aDy volume of gradient in y direction 
 *  \param aDz volume of gradient in z direction
 */
void setGradientTextures(const Volume &aDx, 
                         const Volume &aDy,
                         const Volume &aDz, 
                         cudaArray * &ptr_d_volumeDx,
                         cudaArray * &ptr_d_volumeDy,
                         cudaArray * &ptr_d_volumeDz,
                         const uint64_t aTimeLastMemSync) {
  bool allocateMemoryDx = (aDx.last_update > aTimeLastMemSync) || (aTimeLastMemSync == 0);
  bool allocateMemoryDy = (aDy.last_update > aTimeLastMemSync) || (aTimeLastMemSync == 0);
  bool allocateMemoryDz = (aDz.last_update > aTimeLastMemSync) || (aTimeLastMemSync == 0);

  ptr_d_volumeDx = syncVolume(tex_gradientX, ptr_d_volumeDx, aDx, allocateMemoryDx);
  ptr_d_volumeDy = syncVolume(tex_gradientY, ptr_d_volumeDy, aDy, allocateMemoryDy);
  ptr_d_volumeDz = syncVolume(tex_gradientZ, ptr_d_volumeDz, aDz, allocateMemoryDz);

  // assign gradient_function
  vr::GradientMethod tmp = gradientLookup;
  HANDLE_ERROR(cudaMemcpyToSymbol(dc_activeGradientMethod, &tmp,
                                  sizeof(enum vr::GradientMethod)));
}

/*! \fn void syncWithDevice(const Volume &aVolumeEmission, const Volume &aVolumeAbsorption,
                    const Volume &aVolumeReflection, const uint64_t aTimeLastMemSync,
                    cudaArray * &d_aVolumeEmission, cudaArray * &d_aVolumeAbsorption, 
                    cudaArray * &d_aVolumeReflection)
 * \brief Copies volume data to device and binds textures to the appropriate data.
 *         Data of one volume can be assigned to multiple textures and won't be copied to device, 
 *         if nothing had been changed since the last rendering.
 * \param aVolumeEmission volume for emission
 * \param aVolumeAbsorption volume for absorption
 * \param aVolumeReflection volume for reflection
 * \param aTimeLastMemSync timestamp on which the last rendering took place
 * \param d_aVolumeEmission array pointing to the device memory of the emission volume
 * \param d_aVolumeAbsorption array pointing to the device memory of the absorption volume
 * \param d_aVolumeReflection array pointing to the device memory of the reflection volume
 */
void syncWithDevice(const Volume &aVolumeEmission, const Volume &aVolumeAbsorption,
                    const Volume &aVolumeReflection, const uint64_t aTimeLastMemSync,
                    cudaArray * &d_aVolumeEmission, cudaArray * &d_aVolumeAbsorption, 
                    cudaArray * &d_aVolumeReflection) {
  // similarities of volumes
  const bool simEmAb = (aVolumeEmission == aVolumeAbsorption);
  const bool simEmRe = (aVolumeEmission == aVolumeReflection);
  const bool simAbRe = (aVolumeAbsorption == aVolumeReflection);

  // update required
  const bool reqUpdateEm = (aVolumeEmission.last_update > aTimeLastMemSync) || (aTimeLastMemSync == 0);
  const bool reqUpdateAb = (aVolumeAbsorption.last_update > aTimeLastMemSync) || (aTimeLastMemSync == 0);
  const bool reqUpdateRe = (aVolumeReflection.last_update > aTimeLastMemSync) || (aTimeLastMemSync == 0);

  // save status
  bool updatedEm = false;
  bool updatedAb = false;
  bool updatedRe = false;

#ifdef DEBUG
  mexPrintf("Emission %d\n", reqUpdateEm);
  mexPrintf("Absorption %d\n", reqUpdateAb);
  mexPrintf("Reflection %d\n", reqUpdateRe);
#endif

  // conditionally update GPU memory and textures in order to save bandwidth
  if (reqUpdateEm) {
    if (!updatedEm) {
      d_aVolumeEmission = syncVolume(tex_emission, d_aVolumeEmission, aVolumeEmission, updatedEm);
      updatedEm = true;
    }

    if (simEmRe && !updatedRe) {
      d_aVolumeReflection = referenceTexture(tex_reflection, d_aVolumeReflection,
                   d_idxReflection, vr::VolumeType::emission);
      updatedRe = true;

#ifdef DEBUG
  mexPrintf("Emission = Reflection\n");
  mexPrintf("setup Reflection\n");
#endif
    }

    if (simEmAb && !updatedAb) {
      d_aVolumeAbsorption = referenceTexture(tex_absorption, d_aVolumeAbsorption,
                   d_idxAbsorption, vr::VolumeType::emission);
      updatedAb = true;

#ifdef DEBUG
  mexPrintf("Emission = Absorption\n");
  mexPrintf("setup Reflection: %d\n", updatedAb);
#endif
    }
  }

  if (reqUpdateAb) {
    if (!updatedAb) {
      d_aVolumeAbsorption =
          syncVolume(tex_absorption, d_aVolumeAbsorption, aVolumeAbsorption, updatedAb);
      updatedAb = true;

#ifdef DEBUG
  mexPrintf("Synced Volume Absorption\n");
#endif
    }

    if (simAbRe && !updatedRe) {
      d_aVolumeReflection = referenceTexture(tex_reflection, d_aVolumeReflection,
                   d_idxReflection, vr::VolumeType::absorption);
      updatedRe = true;

#ifdef DEBUG
  mexPrintf("Absorption = Reflection\n");
  mexPrintf("setup Reflection: %d\n", updatedRe);
#endif

    }

    if (simEmAb && !updatedEm) {
      d_aVolumeAbsorption = referenceTexture(tex_emission, d_aVolumeAbsorption,
                   d_idxEmmission, vr::VolumeType::absorption);
      updatedEm = true;

#ifdef DEBUG
  mexPrintf("Absorption = Emission\n");
  mexPrintf("setup Emission: %d\n", updatedEm);
#endif

    }
  }

  if (reqUpdateRe) {
    if (!updatedRe) {
      d_aVolumeReflection = 
          syncVolume(tex_reflection, d_aVolumeReflection, aVolumeReflection, reqUpdateRe);
      updatedRe = true;

#ifdef DEBUG
  mexPrintf("Synced Volume Reflection\n");
#endif
    }

    if (simAbRe && !updatedAb) {
      d_aVolumeAbsorption = referenceTexture(tex_absorption, d_aVolumeAbsorption,
                   d_idxAbsorption, vr::VolumeType::reflection);
      updatedAb = true;

#ifdef DEBUG
  mexPrintf("Reflection = Absorption\n");
  mexPrintf("setup Absorption: %d\n", updatedAb);
#endif

    }

    if (simEmAb && !updatedEm) {
      d_aVolumeEmission = referenceTexture(tex_emission, d_aVolumeEmission,
                   d_idxEmmission, vr::VolumeType::reflection);
      updatedEm = true;

#ifdef DEBUG
  mexPrintf("Reflection = Emission\n");
  mexPrintf("setup Emission: %d\n", updatedEm);
#endif
    }
  }

  // no further check is necessary
  return;
};
} // namespace vr
#endif // #ifndef _VOLUMERENDER_KERNEL_CU_
