//----------------------------------------------------------------------------------
//! Dynamic library and runtime type system initialization.
/*!
// \file    
// \author  Alvarez
// \date    2020-03-17
*/
//----------------------------------------------------------------------------------

#include "ZESSDRR_CUDASystem.h"

// Include definition of ML_INIT_LIBRARY.
#include <mlLibraryInitMacros.h>

// Include all module headers ...
#include "mlDRR_CUDA.h"


ML_START_NAMESPACE

//----------------------------------------------------------------------------------
//! Calls init functions of all modules to add their types to the runtime type
//! system of the ML.
//----------------------------------------------------------------------------------
int ZESSDRR_CUDAInit()
{
  // Add initClass calls from modules here.
  DRR_CUDA::initClass();

  return 1;
}

ML_END_NAMESPACE


//! Calls the init method implemented above during load of shared library.
ML_INIT_LIBRARY(ZESSDRR_CUDAInit)