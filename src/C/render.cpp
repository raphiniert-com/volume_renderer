/*! \file render.cpp
 * 	\author Raphael Scheible <raphael.scheible@uniklinik-freiburg.de>
 * 	\version 1.0
 * 	\license This project is released under the GNU Affero General Public
 * License, Version 3
 *
 * 	\brief interface to matlab
 */

#include <stdio.h>
#include <string.h>

#include <common.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <mex.h>
#include <volumeRender.h>

using namespace vr;

/*! \fn Volume make_volume(const mxArray* prhs)
 * 	\brief constructing a Volume structure
 *  \param prhs pointer to the matlab volume
 *  \return structure of type Volume
 */
Volume make_volume(const mxArray *prhs) {
  const mxArray *arrData = prhs;
  const size_t *dimArray = mxGetDimensions(arrData);

  float *data = (float *)mxGetPr(arrData);

  size_t depth(1);

#ifdef DEBUG
  mexPrintf("Volume:\n\t#dimensions: %d\n", mxGetNumberOfDimensions(arrData));
#endif

  // since mxGetNumberOfDimensions allways returns 2 or greater this works
  if (mxGetNumberOfDimensions(arrData) == 3)
    depth = dimArray[2];

#ifdef DEBUG
  if (mxGetNumberOfDimensions(arrData) > 2)
    mexPrintf("\tresolution: %dx%dx%d\n", dimArray[0], dimArray[1],
              dimArray[2]);
  else
    mexPrintf("\tresolution: %dx%d\n", dimArray[0], dimArray[1]);
#endif

  cudaExtent extent =
      make_cudaExtent(dimArray[0], dimArray[1], depth);

  return make_volume((float *)data, extent);
}

/*! \fn float3 make_float3(float * aPointer)
 * 	\brief constructing a float3 structure [x,y,z]
 *  \param aPointer pointer the first element
 *  \return structure of type float3
 */
float3 make_float3(float *aPointer) {
  return make_float3(aPointer[0], aPointer[1], aPointer[2]);
}

/*! \fn float3 make_float3(float * aPointer)
 * 	\brief constructing a float3 structure [lev, row, col]
 *  \param aPointer pointer the first element
 *  \return structure of type float3
 */
float3 make_float3Inv(float *aPointer) {
  return make_float3(aPointer[2], aPointer[1], aPointer[0]);
}

/*! \fn void checkFreeDeviceMemory(size_t aRequiredRAMInBytes)
 * 	\brief checks if there is enough free device memory available
 *  \param aRequiredRAMInBytes required memory in bytes
 *
 * 	If there is not enough free device memory available the program will be
 * stopped and an error message will be displayed in the matlab interface. The
 * user will be informed how much memory he wanted to allocate and how much
 * 	(free) memory the device offers.
 */
void checkFreeDeviceMemory(size_t aRequiredRAMInBytes) {
  size_t totalMemoryInBytes, curAvailMemoryInBytes;

  bool isEnough = false;
  // CUcontext context;
  // CUdevice device;

  // cudaGetDevice(&device);
  // cuCtxCreate(&context, 0, device); // Create context

  // cuMemGetInfo(&curAvailMemoryInBytes, &totalMemoryInBytes);
  cudaMemGetInfo(&curAvailMemoryInBytes, &totalMemoryInBytes);
#ifdef DEBUG

  mexPrintf(
      "\ttotal memory: %ld MB, free memory: %ld MB, required memory: %ld MB\n",
      totalMemoryInBytes / (1024 * 1024), curAvailMemoryInBytes / (1024 * 1024),
      aRequiredRAMInBytes / (1024 * 1024));

#endif

  isEnough = (curAvailMemoryInBytes >= aRequiredRAMInBytes);
  // cuCtxDetach(context); // Destroy context

  if (!isEnough) {
    std::ostringstream os;
    os << "insufficient free VRAM!\n"
       << "\tTotal Memory (MB): \t" << totalMemoryInBytes / (1024 * 1024)
       << "\n"
       << "\tFree Memory (MB): \t" << curAvailMemoryInBytes / (1024 * 1024)
       << "\n"
       << "\tRequired memory (MB): \t" << aRequiredRAMInBytes / (1024 * 1024)
       << "\n";

    mexErrMsgTxt(os.str().c_str());
  }
}

#define MIN_ARGS 12

/*! \fn void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray
 * *prhs[] ) \brief connects matlab with the renderer \param nlhs number of
 * left-sided arguments (results) \param plhs pointer that points to the
 * left-sided arguments \param nrhs number of right arguments (parameters)
 * 	\param prhs pointer that points to the right arguments
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs == 0)
    mexErrMsgTxt("no parameter!");

  if (nlhs > 1)
    mexErrMsgTxt("Too many output arguments.");

  if (nrhs < MIN_ARGS)
    mexErrMsgTxt("insufficient parameter!");

  const mxArray *mxLightSources = prhs[0];
  const mxArray *mxVolumeLight = prhs[4];

  // compute the size of data copied to the GPU
  size_t requiredRAM(0);

  if (!(mxIsClass(mxLightSources, "logical") ||
        mxIsClass(mxVolumeLight, "logical"))) {
    // get size num lights
    const size_t numLightSources = mxGetN(mxLightSources);
    const Volume volumeLight = make_volume(mxVolumeLight);

#ifdef DEBUG
    mexPrintf("Setting up %d Lightsources\n", numLightSources);
#endif

    // setup lightsources
    LightSource* lightSources = new LightSource[numLightSources];

    // read in lightSources
    for (int l = 0; l < numLightSources; ++l) {
      // get light source data from the matlab class LightSource
      float *mxLightColor =
          (float *)mxGetPr(mxGetProperty(mxLightSources, l, "Color"));
      float *mxLightPosition =
          (float *)mxGetPr(mxGetProperty(mxLightSources, l, "Position"));

      lightSources[l] = make_lightSource(make_float3Inv(mxLightPosition),
                                         make_float3(mxLightColor));

#ifdef DEBUG
      mexPrintf("\t#%d:\tPosition: %f %f %f, \n\t\tColor: %f %f %f\n", l + 1,
                lightSources[l].position.x, lightSources[l].position.y,
                lightSources[l].position.z, lightSources[l].color.x,
                lightSources[l].color.y, lightSources[l].color.z);
#endif
    }

    // copy to GPU
    copyLightSources(lightSources, numLightSources);
    setIlluminationTexture(volumeLight);

    // compute needed RAM
    requiredRAM += (volumeLight.extent.width * volumeLight.extent.depth *
                      volumeLight.extent.height * sizeof(VolumeType) +
                      numLightSources * sizeof(LightSource));
  }

  // reading all volume data from the matlab class Volume
  Volume volumeEmission = make_volume(prhs[1]);
  const Volume volumeReflection = make_volume(prhs[2]);
  const Volume volumeAbsorption = make_volume(prhs[3]);

  // 0: emission
  // 1: reflection
  // 2: absorption
  const float *scales = reinterpret_cast<float *>(mxGetPr(prhs[5]));
  const float3 elementSizeUm = make_float3Inv((float *)mxGetPr(prhs[6]));
  const size_t *imageResolution = reinterpret_cast<size_t *>(mxGetPr(prhs[7]));
  const float *ptrRotationMatrix = reinterpret_cast<float *>(mxGetPr(prhs[8]));
  const float *properties = (float *)mxGetPr(prhs[9]);

#ifdef DEBUG
  mexPrintf("Resolution: %dx%d\n", imageResolution[1], imageResolution[0]);
#endif

  requiredRAM += imageResolution[0] * imageResolution[1] * sizeof(VolumeType) * 3;

  // lev, row, col -> x,y,z
  float4x3 rotationMatrix;
  rotationMatrix.m[0] = make_float3(ptrRotationMatrix[2], ptrRotationMatrix[1],
                                    ptrRotationMatrix[0]);

  rotationMatrix.m[1] = make_float3(ptrRotationMatrix[5], ptrRotationMatrix[4],
                                    ptrRotationMatrix[3]);

  rotationMatrix.m[2] = make_float3(ptrRotationMatrix[8], ptrRotationMatrix[7],
                                    ptrRotationMatrix[6]);
  rotationMatrix.m[3] =
      make_float3(properties[0], properties[1], properties[2]);

#ifdef DEBUG
  mexPrintf("Matrix:\n"
            "\t%f %f %f\n"
            "\t%f %f %f\n"
            "\t%f %f %f\n",
            rotationMatrix.m[0].x, rotationMatrix.m[0].y, rotationMatrix.m[0].z,
            rotationMatrix.m[1].x, rotationMatrix.m[1].y, rotationMatrix.m[1].z,
            rotationMatrix.m[2].x, rotationMatrix.m[2].y,
            rotationMatrix.m[2].z);
#endif

  const float opacityThreshold = (float)mxGetScalar(prhs[10]);

  dim3 block_size = dim3(16, 16);
  dim3 grid_size = dim3(vr::iDivUp(imageResolution[1], block_size.x),
                        vr::iDivUp(imageResolution[0], block_size.y));

  // FIXME: BGR -> RGB
  const float3 color = make_float3((float *)mxGetPr(prhs[11]));

  RenderOptions options =
      initRender(imageResolution[1], imageResolution[0], scales[0], scales[1],
                 scales[2], elementSizeUm, rotationMatrix, opacityThreshold,
                 volumeEmission.extent); // TODO: analyze Volume Size
  // compute needed ram
  // emission is needed in anycase
  requiredRAM += volumeEmission.extent.width * volumeEmission.extent.depth *
               volumeEmission.extent.height * sizeof(VolumeType);

  // check if absorption is unique
  if (volumeEmission != volumeAbsorption &&
      volumeReflection != volumeAbsorption) {
    requiredRAM += volumeAbsorption.extent.width * volumeAbsorption.extent.depth *
                 volumeAbsorption.extent.height * sizeof(VolumeType);
  }

  // check if reflection is unique
  if (volumeEmission != volumeReflection &&
      volumeReflection != volumeAbsorption) {
    requiredRAM += volumeReflection.extent.width * volumeReflection.extent.depth *
                 volumeReflection.extent.height * sizeof(VolumeType);
  }

  // if gradients are passed through
  if (nrhs == MIN_ARGS + 3) {
    Volume dx = make_volume(prhs[MIN_ARGS]);
    Volume dy = make_volume(prhs[MIN_ARGS + 1]);
    Volume dz = make_volume(prhs[MIN_ARGS + 2]);

    setGradientTextures(dx, dy, dz);

    requiredRAM += (dx.extent.width * dx.extent.depth * dx.extent.height *
                      sizeof(VolumeType) +

                  dy.extent.width * dy.extent.depth * dy.extent.height *
                      sizeof(VolumeType) +

                  dz.extent.width * dz.extent.depth * dz.extent.height *
                      sizeof(VolumeType));
  }

  // check if there is enough free VRam
  // if not program will stop with an error msg
  checkFreeDeviceMemory(requiredRAM);

  // switch
  mwSize dim[3] = {imageResolution[0], imageResolution[1], 3};

  float *result = render(block_size, grid_size, options, volumeEmission,
                         volumeAbsorption, volumeReflection, color);

  mxArray *resultArray = mxCreateNumericArray(3, dim, mxSINGLE_CLASS, mxREAL);

  float *outData = (float *)mxGetPr(resultArray);
  size_t size(imageResolution[0] * imageResolution[1] * 3);

  // write result to matlab
  for (size_t i = 0; i < size; ++i) {
    outData[i] = result[i];
  }

  plhs[0] = resultArray;

  // free host memory
  free(result);

  return;
}
