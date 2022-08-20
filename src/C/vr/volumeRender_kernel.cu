/*! \file volumeRender_kernel.cu
 * 	\author Raphael Scheible <raphael.scheible@uniklinik-freiburg.de>
 * 	\version 1.0
 * 	\license This project is released under the GNU Affero General Public License, Version 3
 *
 * 	\brief CUDA C file with all the device functions
 *
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

/*! \var cudaArray *d_emissionArray
 * 	\brief device array of the emission volume data
 */
cudaArray *d_emissionArray = 0;

/*! \var cudaArray *d_gradientXArray
 * 	\brief device array of the gradient volume data in x direction
 */
cudaArray *d_gradientXArray = 0;

/*! \var cudaArray *d_gradientYArray
 * 	\brief device array of the gradient volume data in y direction
 */
cudaArray *d_gradientYArray = 0;

/*! \var cudaArray *d_gradientZArray
 * 	\brief device array of the gradient volume data in z direction
 */
cudaArray *d_gradientZArray = 0;

/*! \var cudaArray *d_absorptionArray
 * 	\brief device array of the absorption volume data
 */
cudaArray *d_absorptionArray = 0;

/*! \var cudaArray *d_reflectionArray
 * 	\brief device array of the reflection volume data
 */
cudaArray *d_reflectionArray = 0;

/*! \var cudaArray *d_illuminationArray
 * 	\brief device array of the illumination volume data
 */
cudaArray *d_illuminationArray = 0;

/*! \var const cudaChannelFormatDesc channelDesc
 * 	\brief channel desc for textures
 */
const cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<vr::VolumeDataType>();

__device__ vr::VolumeType d_idxEmmission = vr::VolumeType::emission;
__device__ vr::VolumeType d_idxAbsorption = vr::VolumeType::emission;
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
                     const float aScaleReflection, const float3 aBoxmin, 
                     const float3 aBoxmax, const float3 aBoxScale)
 *  \brief determines the light performed at a voxelposition of all defined lightsources
 * 			   depending on the undelying illumination texture/model
 *  \param aSamplePosition
 *  \param aPosition
 * 	\param aGradientStep step size to a neighbor voxel
 * 	\param aViewPosition position of the viewer
 * 	\param aColor the color the rendered volume absorbs
 * 	\param aLightSources pointer to all light sources
 * 	\param aScaleReflection scales the sampled value of reflection
 * 	\param aBoxmin min extents of the intersection box
 * 	\param aBoxmin max extents of the intersection box
 * 	\param aBoxScale 1 devided by size of the box
 */
__device__ float3 shade(const float3 &aSamplePosition, const float3 aPosition,
                        const float3 aGradientStep, const float3 aViewPosition,
                        const float3 aColor, vr::LightSource *aLightSources,
                        const float aScaleReflection, const float3 aBoxmin,
                        const float3 aBoxmax, const float3 aBoxScale) {
  const float scaleReflection = aScaleReflection;

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

    // lookup in d_reflectionArray/tex_reflection
    float reflection =
        scaleReflection * tex3D(getTexture(d_idxReflection), aSamplePosition.x,
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
  const float scaleAbsorption = aOptions.scale_absorption;
  const float scaleEmission = aOptions.scale_emission;
  const float scaleReflection = aOptions.scale_reflection;

  const float3 boxMin = aOptions.boxmin;
  const float3 boxMax = aOptions.boxmax;

  // 2D image plane in [-1,1]
  float u = (x / (float)aOptions.image_width) * 2.0f - 1.0f;
  float ratio = aOptions.image_height / (float)aOptions.image_width;
  float v = (y / (float)aOptions.image_height) * 2.0f * ratio - 1.0f * ratio;

  // calculate eye ray in world space
  vr::Ray eyeRay;

  // box scale
  const float3 boxScale = 1.f / (boxMax - boxMin);

  // translate into scale
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

    // read from 3D texture and apply several scale factor
    float emission = scaleEmission * tex3D(getTexture(d_idxEmmission), pos_sample.x,
                                           pos_sample.y, pos_sample.z);
    float absorption = scaleAbsorption * tex3D(getTexture(d_idxAbsorption), pos_sample.x,
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
              lightSources, scaleReflection, boxMin, boxMax, boxScale);

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

/*! \fn void createTextureFromVolume(texture<VolumeDataType,
 *      cudaTextureType3D, cudaReadModeElementType>& aTex, const Volume& aVolume,
 * 			cudaArray* d_aArray, const bool aNormalized=true)
 *  \brief copies Volume from host to device
 *  \param aTex texture that is used to perform lookups
 *  \param aVolume volume that is copied to the device
 * 	\param d_aArray device array where the data are stored in/copied to
 * 	\return device pointer of the device array
 */
void createTextureFromVolume(
    texture<VolumeDataType, cudaTextureType3D, cudaReadModeElementType> &aTex,
    const Volume &aVolume, cudaArray *d_aArray) {
  
  if (d_aArray == 0)
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

  // set texture parameters
  aTex.normalized = true; // access with normalized texture coordinates
  aTex.filterMode = cudaFilterModeLinear;     // linear interpolation
  aTex.addressMode[0] = cudaAddressModeClamp; // clamp texture coordinates
  aTex.addressMode[1] = cudaAddressModeClamp;
  aTex.addressMode[2] = cudaAddressModeClamp;

  // bind d_aArray to 3D texture
  HANDLE_ERROR(cudaBindTextureToArray(aTex, d_aArray, channelDesc));
}

/*! \fn cudaArray* setIlluminationTexture(const Volume& aVolume)
 *  \brief copies illumination volume from host to device
 *  \param aVolume illumination volume
 */
void setIlluminationTexture(const Volume &aVolume) {
  createTextureFromVolume(tex_illumination, aVolume, d_illuminationArray);
}

/*! \fn cudaArray* setGradientTextures(const Volume& aDx, const Volume& aDy, const Volume& aDz) 
 *  \brief copies gradient volumes from host to device 
 *  \param aDx volume of gradient in x direction 
 *  \param aDy volume of gradient in y direction 
 *  \param aDz volume of gradient in z direction
 */
void setGradientTextures(const Volume &aDx, 
                         const Volume &aDy,
                         const Volume &aDz) {
  createTextureFromVolume(tex_gradientX, aDx, d_gradientXArray);
  createTextureFromVolume(tex_gradientY, aDy, d_gradientYArray);
  createTextureFromVolume(tex_gradientZ, aDz, d_gradientZArray);

  // assign gradient_function
  vr::GradientMethod tmp = gradientLookup;
  HANDLE_ERROR(cudaMemcpyToSymbol(dc_activeGradientMethod, &tmp,
                                  sizeof(enum vr::GradientMethod)));
}

/*! \fn void freeCudaBuffersGradients()
 *  \brief removes the gradients from the device memory
 */
void freeCudaGradientBuffers() {
  // get value of dc_activeGradientMethod from device to host
  vr::GradientMethod h_activeGradientMethod;
  cudaMemcpyFromSymbol(&h_activeGradientMethod, dc_activeGradientMethod,
                       sizeof(vr::GradientMethod), 0);

  if (h_activeGradientMethod == gradientLookup) {
    HANDLE_ERROR(cudaFreeArray(d_gradientXArray));
    HANDLE_ERROR(cudaFreeArray(d_gradientYArray));
    HANDLE_ERROR(cudaFreeArray(d_gradientZArray));
  }
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

/*! \fn void copyLightSources(const LightSource *aLightSources, const size_t aNumOfLightSources)
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

/*! \fn void syncTexture(const textureReference& aTexture, const cudaArray* d_array, 
                         const cudaArray* d_arrayRef, bool& aUpdated)
 *  \brief sync texture
 * 	\param d_array 
 *  \param d_arrayRef 
 * 	\param aTexture 
 * 	\param aUpdated 
 */
void setupTexture(
    texture<vr::VolumeDataType, cudaTextureType3D, cudaReadModeElementType>& aTexture, 
    cudaArray* d_aArray, vr::VolumeType& d_aIdx, vr::VolumeType aTypeAssigned, bool& aUpdated) {
  
  // clear memory
  if (d_aArray != 0) {
    HANDLE_ERROR(cudaUnbindTexture(aTexture));
    HANDLE_ERROR(cudaFreeArray(d_aArray));
    d_aArray = 0;
  }
  
  // just reference to the same device variable
  // HANDLE_ERROR(cudaBindTextureToArray(aTexture, d_aArrayRef, channelDesc));
  aUpdated = true;
  HANDLE_ERROR(
    cudaMemcpyToSymbol(d_aIdx, &aTypeAssigned, sizeof(enum vr::VolumeType))
  );
}

/*! \fn void syncVolume(cudaArray* d_array, const Volume& aVolume, 
                               const textureReference& aTexture, bool& aUpdated)
 *  \brief sync texture
 * 	\param d_array 
 * 	\param aVolume 
 * 	\param aTexture 
 * 	\param aUpdated 
 */
void syncVolume(
    texture<vr::VolumeDataType, cudaTextureType3D, cudaReadModeElementType>& aTexture,
    cudaArray* d_aArray, const Volume& aVolume, bool& aUpdated) {
  // clear memory
  if (d_aArray != 0) {
    HANDLE_ERROR(cudaUnbindTexture(aTexture));
    HANDLE_ERROR(cudaFreeArray(d_aArray));
    d_aArray = 0;
  }

  createTextureFromVolume(aTexture, aVolume, d_aArray);
  aUpdated = true;
}


/*! \fn void syncWithDevice(const Volume &aVolumeEmission, const Volume &aVolumeAbsorption,
                            const Volume &aVolumeReflection, const uint64_t &timeLastMemSync)
 * \brief Copies volume data to device and binds textures to the appropriate data.
 *         Data of one volume can be assigned to multiple textures and won't be copied to device, 
 *         if nothing had been changed since the last rendering.
 * \param aVolumeEmission
 * \param aVolumeAbsorption
 * \param aVolumeReflection
 * \param timeLastMemSync
 */
void syncWithDevice(const Volume &aVolumeEmission, const Volume &aVolumeAbsorption,
                    const Volume &aVolumeReflection, const uint64_t &timeLastMemSync) {

  // similarities of volumes
  const bool simEmAb = (aVolumeEmission == aVolumeAbsorption);
  const bool simEmRe = (aVolumeEmission == aVolumeReflection);
  const bool simAbRe = (aVolumeAbsorption == aVolumeReflection);

  // update required
  const bool reqUpdateEm = (aVolumeEmission.last_update > timeLastMemSync) || (timeLastMemSync == 0);
  const bool reqUpdateAb = (aVolumeAbsorption.last_update > timeLastMemSync) || (timeLastMemSync == 0);
  const bool reqUpdateRe = (aVolumeReflection.last_update > timeLastMemSync) || (timeLastMemSync == 0);

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
      syncVolume(tex_emission, d_emissionArray, aVolumeEmission, updatedEm);
    }

    if (simEmRe && !updatedRe) {
      setupTexture(tex_reflection, d_reflectionArray,
                   d_idxReflection, vr::VolumeType::reflection, updatedRe);

#ifdef DEBUG
  mexPrintf("Emission = Reflection\n");
  mexPrintf("setup Reflection\n");
#endif
    }

    if (simEmAb && !updatedAb) {
      setupTexture(tex_absorption, d_absorptionArray,
                   d_idxAbsorption, vr::VolumeType::emission, updatedAb);
#ifdef DEBUG
  mexPrintf("Emission = Absorption\n");
  mexPrintf("setup Reflection: %d\n", updatedAb);
#endif
    }
  }

  if (reqUpdateAb) {
    if (!updatedAb) {
      syncVolume(tex_absorption, d_absorptionArray, aVolumeAbsorption, updatedAb);

#ifdef DEBUG
  mexPrintf("Synced Volume Absorption\n");
#endif
    }

    if (simAbRe && !updatedRe) {
      setupTexture(tex_reflection, d_reflectionArray,
                   d_idxReflection, vr::VolumeType::reflection, updatedRe);
#ifdef DEBUG
  mexPrintf("Absorption = Reflection\n");
  mexPrintf("setup Reflection: %d\n", updatedRe);
#endif

    }

    if (simEmAb && !updatedEm) {
      setupTexture(tex_emission, d_emissionArray, 
                   d_idxEmmission, vr::VolumeType::emission, updatedEm);

#ifdef DEBUG
  mexPrintf("Absorption = Emission\n");
  mexPrintf("setup Emission: %d\n", updatedEm);
#endif

    }
  }

  if (reqUpdateRe) {
    if (!updatedRe) {
      syncVolume(tex_reflection, d_reflectionArray, aVolumeReflection, updatedRe);

#ifdef DEBUG
  mexPrintf("Synced Volume Reflection\n");
#endif
    }

    if (simAbRe && !updatedAb) {
      setupTexture(tex_absorption, d_absorptionArray, 
                   d_idxAbsorption, vr::VolumeType::absorption, updatedAb);

#ifdef DEBUG
  mexPrintf("Reflection = Absorption\n");
  mexPrintf("setup Absorption: %d\n", updatedAb);
#endif

    }

    if (simEmAb && !updatedEm) {
      setupTexture(tex_emission, d_emissionArray, 
                   d_idxEmmission, vr::VolumeType::emission, updatedEm);
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