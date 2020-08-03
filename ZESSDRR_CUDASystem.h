//----------------------------------------------------------------------------------
//! Project global and OS specific declarations.
/*!
// \file    
// \author  Alvarez
// \date    2020-03-17
*/
//----------------------------------------------------------------------------------


#pragma once


// DLL export macro definition.
#ifdef ZESSDRR_CUDA_EXPORTS
  // Use the ZESSDRR_CUDA_EXPORT macro to export classes and functions.
  #define ZESSDRR_CUDA_EXPORT ML_LIBRARY_EXPORT_ATTRIBUTE
#else
  // If included by external modules, exported symbols are declared as import symbols.
  #define ZESSDRR_CUDA_EXPORT ML_LIBRARY_IMPORT_ATTRIBUTE
#endif
