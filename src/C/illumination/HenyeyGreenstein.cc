/*! \file volumeRender.h
 * 	\author Raphael Scheible <raphael.scheible@uniklinik-freiburg.de>
 * 	\version 1.0
 *  \license This project is released under the GNU Affero General Public License, Version 3
 *
 * 	\brief matlab command to generate a LUT with Henyey-Greenstein phase
 * function
 *
 */

#include <math.h>
#include <mex.h>
#include <sstream>
#include <stdio.h>
#include <string.h>

/*! \struct float3
 * 	\brief struct of a 3D float
 */
struct float3 {
  /*! The x component */
  float x;
  /*! The y component */
  float y;
  /*! The z component */
  float z;
};

/*! \fn float3 make_float3(float x, float y, float z)
 * 	\brief constructing a float3 structure [x,y,z]
 *  \param x the x component
 *  \param x the y component
 *  \param x the z component
 *  \return structure of type float3
 */
inline float3 make_float3(float x, float y, float z) {
  float3 result;
  result.x = x;
  result.y = y;
  result.z = z;

  return result;
}

/*! \fn float dot(const float3& a,const float3& b)
 * 	\brief computes the dot-product of two vectors
 *  \param a first vector
 *  \param b second vector
 *  \return dot product (scalar)
 */
inline float dot(const float3 &a, const float3 &b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

/*! \fn float3 make_float3(float * aPointer)
 * 	\brief constructing a float3 structure [x,y,z]
 *  \param aPointer pointer the first element
 *  \return structure of type float3
 */
inline float3 make_float3(float *aPointer) {
  return make_float3(aPointer[0], aPointer[1], aPointer[2]);
}

/*! \struct float3x3
 * 	\brief struct of a 3x3 matrix of floats
 */
struct float3x3 {
  /*! The cols of the matrix */
  float3 col[3];
};

/*! \fn float3 operator*( const float3x3& m, const float3& v )
 * 	\brief multiplication operand of matrix times vector
 *  \param m a 3x3 matrix
 *  \param v a vector
 *  \return the resulting vector
 */
float3 operator*(const float3x3 &m, const float3 &v) {
  float3 result;

  result.x = m.col[0].x * v.x + m.col[1].x * v.y + m.col[2].x * v.z;
  result.y = m.col[0].y * v.x + m.col[1].y * v.y + m.col[2].y * v.z;
  result.z = m.col[0].z * v.x + m.col[1].z * v.y + m.col[2].z * v.z;

  return result;
}

/*! float3x3 rotateAroundX( float angle )
 * 	\brief rotatation around the x-component [1,0,0]
 *  \param angle rotation angle
 *  \return rotation matrix
 */
float3x3 rotateAroundX(float angle) {
  float s = sinf(angle);
  float c = cosf(angle);
  float3x3 m;
  m.col[0].x = 1;
  m.col[1].x = 0;
  m.col[2].x = 0;
  m.col[0].y = 0;
  m.col[1].y = c;
  m.col[2].y = s;
  m.col[0].z = 0;
  m.col[1].z = -s;
  m.col[2].z = c;
  return m;
}

#define MIN_ARGS 1
#define PI ((float)3.141592653589793238462643383279502884197169399375105820)
#define PI2 ((float)6.2831853071795864769252867665590057683943387987502116)

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

  float aG(0.8);

  if (nrhs == 2)
    aG = (float)mxGetScalar(prhs[1]);

  if (aG > 1 || aG < -1)
    mexErrMsgTxt("g must be in interval [-1,1]");

  unsigned int volumeSize = (unsigned int)mxGetScalar(prhs[0]);

  mwSize dim[3] = {volumeSize, volumeSize, volumeSize};
  mxArray *resultArray = mxCreateNumericArray(3, dim, mxSINGLE_CLASS, mxREAL);

  float frac_full(PI2 / volumeSize);
  float frac_half(PI / volumeSize);

  float *outData = (float *)mxGetPr(resultArray);

  float alpha(0), beta(0), gamma(0);
  float3 lightIn, lightOut;
  float3x3 rotMatrix;

  int pageSize(volumeSize * volumeSize);

  for (int c = 0; c < volumeSize; ++c) {
    gamma = c * frac_full;

    for (int a = 0; a < volumeSize; ++a) {
      alpha = a * frac_half;
      lightOut = make_float3(sinf(alpha), cosf(alpha), 0);

      for (int b = 0; b < volumeSize; ++b) {
        beta = b * frac_half;
        lightIn = make_float3(sinf(beta), cosf(beta), 0);

        // rotate around X because of the properties of the unit circle
        // x denotes the normal in the later application
        float3x3 rotMatrix = rotateAroundX(gamma);
        float3 lightOutRotated = rotMatrix * lightOut;

        float scalar = dot(lightOutRotated, lightIn);

        float angleRad = acosf(scalar);

        float numerator = (1.f - powf(aG, 2.f));
        float denominator = sqrtf(
            powf((1.f + powf(aG, 2.f) - (2.f * aG * cosf(angleRad))), 3.f));

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
