/*! \file render.cpp
 * 	\author Raphael Scheible <mail@raphiniert.com>
 * 	\version 1.0
 * 	\license This project is released under the GNU Affero General Public License, Version 3
 *
 * 	\brief interface to matlab
 */

#include <stdio.h>
#include <string.h>
#include <stddef.h>
#include <vr/mm/mmanager.hxx>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <mex.h>
#include <vr/volumeRender.h>

using namespace vr;

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



#define MIN_ARGS 14

/*! \fn void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray prhs[] ) 
 *  \brief connects matlab with the renderer
 *  \param nlhs number of left-sided arguments (results)
 *  \param plhs pointer that points to the left-sided arguments
 *  \param nrhs number of right arguments (parameters)
 * 	\param prhs pointer that points to the right arguments
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs == 0)
    mexErrMsgTxt("no parameter!");

  char cmd[64];
  if (nrhs < 1 || mxGetString(prhs[0], cmd, sizeof(cmd)))
		mexErrMsgTxt("First input should be a command string less than 64 characters long.");

  if (!strcmp("new", cmd)) {
    // Check parameters
    if (nlhs != 1)
        mexErrMsgTxt("New: One output expected.");
    // Return a handle to a new C++ instance
    plhs[0] = mm::convertPtr2Mat<mm::MManager>(new mm::MManager);
    return;
  }

  // Check there is a second input, which should be the class instance handle
  if (nrhs < 2)
    mexErrMsgTxt("Second input should be a class instance handle.");

  // Delete
  if (!strcmp("delete", cmd)) {
      // Destroy the C++ object
      mm::destroyObject<mm::MManager>(prhs[1]);
      // Warn if other commands were ignored
      if (nlhs != 0 || nrhs != 2)
          mexWarnMsgTxt("Delete: Unexpected arguments ignored.");
      return;
  }

  // Get the class instance pointer from the second input
  mm::MManager* mmanager_instance = mm::convertMat2Ptr<mm::MManager>(prhs[1]);

  // mem_info
  if (!strcmp("mem_info", cmd)) {
    mexPrintf(mmanager_instance->memInfo().c_str());
  }

  // compute the size of data copied to the GPU
  size_t requiredRAM = mmanager_instance->getRequiredMemory();

  // sync_volumes
  if (!strcmp("sync_volumes", cmd)) {
    mmanager_instance->timeLastMemSync = (uint64_t)mxGetScalar(prhs[2]);

    mmanager_instance->volumeEmission = mxMake_volume(prhs[3]);
    mmanager_instance->volumeReflection = mxMake_volume(prhs[4]);
    mmanager_instance->volumeAbsorption = mxMake_volume(prhs[5]);


    // assign gradient_function
    vr::GradientMethod gm = gradientCompute;

    // gradient volume is given
    if (nrhs == 9) {
      mmanager_instance->volumeDx = mxMake_volume(prhs[6]);
      mmanager_instance->volumeDy = mxMake_volume(prhs[7]);
      mmanager_instance->volumeDz = mxMake_volume(prhs[8]);

      gm = gradientLookup;
    } else if(nrhs == 6) {
      mmanager_instance->resetGradients();
    }

    // compute required RAM
    requiredRAM += mmanager_instance->getRequiredMemory();

    // check if GPU has enough space
    mm::MManager::checkFreeDeviceMemory(requiredRAM);

    setGradientMethod(gm);
    mmanager_instance->sync();

    // Warn if other commands were ignored
    if (nlhs != 0 || nrhs > 9)
        mexWarnMsgTxt("SyncVolumes: Unexpected arguments ignored.");

    return;
  }

  // ------
  // render
  // ------
  if (!strcmp("render", cmd)) {

    if (nlhs > 1)
      mexErrMsgTxt("Too many output arguments.");

    if (nrhs < MIN_ARGS) {
      mexErrMsgTxt("insufficient parameter!");
      mexPrintf("   %s of %s\n", nrhs, MIN_ARGS);
    }

    const mxArray *mxLightSources = prhs[2];

    if (!mxIsClass(mxLightSources, "logical")) {
      // get size num lights
      const size_t numLightSources = mxGetN(mxLightSources);

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
        float lightIntensity = (float)mxGetScalar(mxGetProperty(mxLightSources, l, "Intensity"));

        lightSources[l] = make_lightSource(make_float3Inv(mxLightPosition),
                                          make_float3(mxLightColor), lightIntensity);

  #ifdef DEBUG
        mexPrintf("\t#%d:\tPosition: %f %f %f, \n\t\tColor: %f %f %f\n", l + 1,
                  lightSources[l].position.x, lightSources[l].position.y,
                  lightSources[l].position.z, lightSources[l].color.x,
                  lightSources[l].color.y, lightSources[l].color.z);
  #endif
      }

      // compute required RAM
      requiredRAM += (numLightSources * sizeof(LightSource));

      // check if GPU has enough space for the LightVolume
      mm::MManager::checkFreeDeviceMemory(requiredRAM);

      // copy to GPU
      copyLightSources(lightSources, numLightSources);
    }

    // assign phase_function
    vr::PhaseMethod sm = phaseCompute;

    const mxArray *mxVolumePhase = prhs[3];
    if (!mxIsClass(mxVolumePhase, "logical")) {
      const Volume volumePhase = mxMake_volume(mxVolumePhase);
      mmanager_instance->ptr_d_volumePhase = setPhaseTexture(volumePhase, 
                                                mmanager_instance->ptr_d_volumePhase,
                                                mmanager_instance->timeLastMemSync);

      sm = phaseLookup;
    }
    setPhaseMethod(sm);

    const float *factors = reinterpret_cast<float *>(mxGetPr(prhs[4]));
    const float3 elementSizeUm = make_float3Inv((float *)mxGetPr(prhs[5]));
    const size_t *imageResolution = reinterpret_cast<size_t *>(mxGetPr(prhs[6]));
    const float *ptrRotationMatrix = reinterpret_cast<float *>(mxGetPr(prhs[7]));
    const float *properties = (float *)mxGetPr(prhs[8]);

  #ifdef DEBUG
    mexPrintf("Resolution: %dx%d\n", imageResolution[1], imageResolution[0]);
  #endif

    // add the rendered image to the required memory
    requiredRAM += imageResolution[0] * imageResolution[1] * sizeof(VolumeDataType) * 3;

    // check if GPU still has enough free space for the rendered image
    mm::MManager::checkFreeDeviceMemory(requiredRAM);

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

    const float opacityThreshold = (float)mxGetScalar(prhs[9]);

    dim3 block_size = dim3(16, 16);
    dim3 grid_size = dim3(vr::iDivUp(imageResolution[1], block_size.x),
                          vr::iDivUp(imageResolution[0], block_size.y));

    const float3 color = make_float3((float *)mxGetPr(prhs[10]));
    const float shininess = (float)mxGetScalar(prhs[11]);
    const float scattering_weight = (float)mxGetScalar(prhs[12]);
    const float hg_asymmetry = (float)mxGetScalar(prhs[13]);

    RenderOptions options =
        initRender(imageResolution[1], imageResolution[0], factors[0], factors[1],
                  factors[2], elementSizeUm, rotationMatrix, opacityThreshold,
                  mmanager_instance->volumeEmission.extent, shininess, scattering_weight,
                  hg_asymmetry);

    // switch
    mwSize dim[3] = {imageResolution[0], imageResolution[1], 3};

#ifdef DEBUG
    mexPrintf("factor ab: %f\n", options.factor_absorption);
    mexPrintf("factor em: %f\n", options.factor_emission);
    mexPrintf("factor re: %f\n", options.factor_reflection);
    mexPrintf("height: %lu\n", options.image_height);
    mexPrintf("width: %lu\n", options.image_width);
    mexPrintf("opacity: %f\n", options.opacity_threshold);
    mexPrintf("shininess: %f\n", options.shininess);
    mexPrintf("scattering weight: %f\n", options.scattering_weight);
    mexPrintf("hg asymmetry: %f\n", options.hg_asymmetry);
#endif

    float *result = render(block_size, grid_size, options, mmanager_instance->volumeEmission.extent, color);

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
}
