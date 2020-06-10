/*! \file render.cpp
 * 	\author Raphael Scheible <raphael.scheible@uniklinik-freiburg.de>
 * 	\version 1.0
 * 	\license This project is released under the GNU Affero General Public License, Version 3 
 *
 * 	\brief interface to matlab for generating a unix timestamp
 */

#include <stdint.h>
#include <chrono>
#include "mex.h"

using namespace std::chrono;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  /* Function accepts only one input argument */
  if(nrhs > 0)
  {
      mexErrMsgTxt("No one input argument accepted\n\n");
  }

  // create matlab return type into
  plhs[0] = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
  int* data = (int*) mxGetData(plhs[0]);

  // get timestamp
  milliseconds ms = duration_cast<milliseconds>(
    system_clock::now().time_since_epoch()
  );

  // assign it as return value
  data[0]=(int64_t)ms.count();
}