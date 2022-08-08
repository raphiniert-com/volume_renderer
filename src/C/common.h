/*! \file common.h
 * 	\author Raphael Scheible <raphael.scheible@uniklinik-freiburg.de>
 * 	\version 1.0
 *  \license This project is released under the GNU Affero General Public License, Version 3
 *
 * 	\brief include common header files and provides a common CUDA debug
 * function
 */

#ifndef COMMON_H_
#define COMMON_H_

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <sstream>
#include <stdio.h>

#ifdef MATLAB_MEX_FILE
#include <mex.h>
#endif

#ifdef DEBUG

/*! \fn static void HandleError(cudaError_t err, const char *file, int line)
 *  \brief prints an error message with the codeline where the error occured
 *  \param err cudaError_t
 *  \param file the file in which the error occured
 * 	\param line	the line in which the error occured
 */
static void HandleError(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {

    std::ostringstream stringStream;
    stringStream << cudaGetErrorString(err) << " in " << file << " at line "
                 << line << "\n";
    std::string error = stringStream.str();

#ifdef MATLAB_MEX_FILE
    mexErrMsgTxt(error.c_str());
#else
    printf(error.c_str());
    exit(EXIT_SUCCESS);
#endif
  }
}
#endif

// release does not need CUDA code check to accelerate the code
#ifdef DEBUG
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))
#else
#define HANDLE_ERROR(err) (err)
#endif

#endif /* COMMON_H_ */
