/*! \file HenyeyGreenstein.cc
 * 	\author Raphael Scheible <mail@raphiniert.com>
 * 	\version 1.0
 *  \license This project is released under the GNU Affero General Public License, Version 3
 *
 * 	\brief matlab command to generate a LUT with Henyey-Greenstein phase function
 *
 */

#include <vr/illumination/float3.h>
#include <math.h>
#include <mex.h>
#include <sstream>
#include <stdio.h>
#include <string.h>

#define MIN_ARGS 1
#define PI ((float)3.141592653589793238462643383279502884197169399375105820)
#define PI2 ((float)6.2831853071795864769252867665590057683943387987502116)

using namespace vr::illumination;

/*! \fn void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray
 * *prhs[] ) \brief computes Heneyey-Greenstein LUT provides an interface to
 * matlab \param nlhs number of left-sided arguments (results) \param plhs
 * pointer that points to the left-sided arguments \param nrhs number of right
 * arguments (parameters) \param prhs pointer that points to the right arguments
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs == 0)
    mexErrMsgTxt("no parameter!");

  if (nlhs > 1)
    mexErrMsgTxt("Too many output arguments.");

  if (nrhs < MIN_ARGS)
    mexErrMsgTxt("insufficient parameter!");

  float aG = 0.8;

  if (nrhs == 2)
    aG = (float)mxGetScalar(prhs[1]);

  if (aG > 1 || aG < -1)
    mexErrMsgTxt("g must be in interval [-1,1]");

  unsigned int volumeSize = (unsigned int)mxGetScalar(prhs[0]);

  mwSize dim[3] = {volumeSize, volumeSize, volumeSize};
  mxArray *resultArray = mxCreateNumericArray(3, dim, mxSINGLE_CLASS, mxREAL);

  float frac_half = PI / volumeSize;

  float *outData = (float *)mxGetPr(resultArray);

  float alpha = 0, beta = 0, gamma = 0;
  float3 lightIn, lightOut;
  float3x3 rotMatrix;

  int pageSize = volumeSize * volumeSize;

  for (int c = 0; c < volumeSize; ++c) {
    gamma = c * frac_half;

    for (int a = 0; a < volumeSize; ++a) {
      alpha = a * frac_half;
      lightOut = make_float3(sinf(alpha), 0, cosf(alpha));

      for (int b = 0; b < volumeSize; ++b) {
        beta = b * frac_half;
        lightIn = make_float3(sinf(beta), 0, cosf(beta));

        // rotate around X because of the properties of the unit circle
        // x denotes the normal in the later application
        float3x3 rotMatrix = rotateAroundX(gamma);
        float3 lightOutRotated = rotMatrix * lightOut;

        float cosTheta = dot(lightOutRotated, lightIn);

        float numerator = (1.f - powf(aG, 2.f));
        float denominator = sqrtf(
            powf((1.f + powf(aG, 2.f) - (2.f * aG * cosTheta)), 3.f));

        // memory layout described on:
        // http://www.mathworks.de/help/techdoc/matlab_external/f21585.html
        int k = c * pageSize + a * volumeSize + b;

        outData[k] = 1.f / (4.f * PI) * (numerator / denominator);
      }
    }
  }

  plhs[0] = resultArray;

  return;
}
