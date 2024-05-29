/*! \file float3.h
 * 	\author Raphael Scheible <mail@raphiniert.com>
 * 	\version 1.0
 *  \license This project is released under the GNU Affero General Public License, Version 3
 *
 * 	\brief implementation of float3 struct intended for the computation of illumination models
 */

#include <math.h>

#ifndef _FLOAT3_H_
#define _FLOAT3_H_

namespace vr {
namespace illumination {
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
} // namespace illumination
} // namespace vr

#endif // _FLOAT3_H_