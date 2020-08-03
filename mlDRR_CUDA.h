//----------------------------------------------------------------------------------
//! The ML module class DRR_CUDA.
/*!
// \file    
// \author  Alvarez
// \date    2020-03-17
//
// 
*/
//----------------------------------------------------------------------------------


#pragma once



#include "ZESSDRR_CUDASystem.h"

#include <mlModuleIncludes.h>

#define		PROJECT_CLASS_NAME			DRR_CUDA
#include <Eigen\Dense>
#include <unsupported\Eigen\CXX11\Tensor>

#include <iostream>
#include <sstream>
#include <mlModuleIncludes.h>

#include <ctime>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#include <thread>
#include <future>

#include <cuda_runtime_api.h>
#include "CUDAdrr.cuh"

#include <math.h>
# define M_PI           3.14159265358979323846  /* pi */

ML_START_NAMESPACE

using namespace std;
using namespace Eigen;

//! 
class ZESSDRR_CUDA_EXPORT DRR_CUDA : public Module
{
public:

  //! Constructor.
  DRR_CUDA();

  //! Handles field changes of the field \p field.
  virtual void handleNotification (Field* field);

  // ----------------------------------------------------------
  //! \name Image processing methods.
  //@{
  // ----------------------------------------------------------

  //! Sets properties of the output image at output \p outputIndex.
  virtual void calculateOutputImageProperties(int outputIndex, PagedImage* outputImage);

  //! Returns the input image region required to calculate a region of an output image.
  //! \param inputIndex        The input of which the regions shall be calculated.
  //! \param outputSubImageBox The region of the output image for which the required input region
  //!                          shall be calculated.
  //! \param outputIndex       The index of the output image for which the required input region
  //!                          shall be calculated.
  //! \return Region of input image needed to compute the region \p outputSubImageBox on output \p outputIndex.
  virtual SubImageBox calculateInputSubImageBox(int inputIndex, const SubImageBox& outputSubImageBox, int outputIndex);

  //! Calculates page \p outputSubImage of output image with index \p outputIndex by using \p inputSubImages.
  //! \param outputSubImage The sub-image of output image \p outputIndex calculated from \p inputSubImges.
  //! \param outputIndex    The index of the output the sub-image is calculated for.
  //! \param inputSubImages Array of sub-image(s) of the input(s) whose extents were specified
  //!                       by calculateInputSubImageBox. Array size is given by getNumInputImages().
  virtual void calculateOutputSubImage(SubImage* outputSubImage, int outputIndex, SubImage* inputSubImages);

  //! Method template for type-specific page calculation. Called by calculateOutputSubImage().
  //! \param outputSubImage The typed sub-image of output image \p outputIndex calculated from \p inputSubImages.
  //! \param outputIndex    The index of the output the sub-image is calculated for.
  //! \param inSubImg0 Temporary typed sub-image of input 0.
  template <typename T>
  void calculateOutputSubImage (TSubImage<T>* outputSubImage, int outputIndex
                               , TSubImage<T>* inputSubImage0
                               );
  //@}

  void prepareDRROutputVariables();

  //! Function to transfer the 3D input object to an Eigen tensor and to a flat object (for CUDA) 
  bool inputImageToEigenMatrix(unsigned inputNumber, float *_o2D, Image3D &CT_Scan_structure, Tensor<float, 3>& inputImage, VectorXf &object3D_eigen);

  //! Function to load the variables that go in the DRR functions
  static void loadVariablesForExecuteDRR(DRRParameters &DRR_parameters, Image2D &DRRoutput, vector<float> _translation, vector<float> _rotation, float _scd, int imageDim[2], float imagePixelDim[2], float _isocenter[3], float _o2D[2], bool _useROI, Vector2 _startROI, Vector2 _endROI, bool _useQuat);

  //! Generic Function to calculate the DRR 
  void getDRR(MatrixXf &_imageOut, vector<float> _translation, vector<float> _rotation, float _scd, Image3D _CT_Scan, bool _useQuat, float imagePixelDim[2], float _o2D[2], bool _useROI, Vector2 _startROI, Vector2 _endROI, bool calculateUsingCUDA  = false);

  //! Function to calculate the DRR using the serial design
  void getDRRSerial(DRRParameters DRR_parameters, Image2D DRRoutput, MatrixXf &imageOut);

  //! Function to calculate the DRR using the parallel design
  void getDRRwithCUDA(DRRParameters DRR_parameters, Image2D DRRoutput, MatrixXf &imageOut);

  //! Function to rotate the points using Euler
  static Vector3f getTranslationAndRotation(Vector3f vectorToTransform, vector<float> translation, vector<float> rotation, vector<float> isocenter);

  //! Function to rotate the points using Quaternions
  static Vector3f getTransAndRotQuaternions(Vector3 vectorToTransform, vector<float> translation, vector<float> rotation, vector<float> isocenter/*, Vector3 isocenter, Vector3 rotationVector, float rotationAngle, float tx, float ty, float tz*/);

  //! Method to to transfer the produced image to the visualization module
  template <typename T>
  static void EigenMatricesToOutputImage(TSubImage<T>* _outputSubImage, int maxValue, MatrixXf imageOut);


  /* Defined by Julio*/
  void readDICOMTagFromInputImage(unsigned inputNumber, Vector3d &ctPixelSpacing);
  bool readDICOMTagFromInputImage(PagedImage* pagedImage, const std::string& tagName, std::string& tagValue, std::string& tagVR, std::vector < std::string >& statusMessages);
  void updateCircleMask();
  void updateROIMask();
  void updateMask();
  
  static void CreateCircleMaskForOutputs(int dx, int dz, bool useCircleMask, cv::Mat &CircleMask, cv::Mat &CircleMask_drr);
  static void CreateCircleMaskForOutputs(int dx, int dz, bool useCircleMask, double centerX, double centerY, double radius, cv::Mat &CircleMask, cv::Mat &CircleMask_drr);
  static void ProcessOutputBeforeRendering(int minDRR, int maxDRR, bool isPosiveOutput, bool useMask, cv::Mat TotalMask, MatrixXf &imageOut);
  static Vector3 validateZeroVector(Vector3 in_vector);
  static cv::Mat translateImg(cv::Mat img, double offsetx, double offsety);
  static void CreateROIForOutputs(int dx, int dz, bool useROI, Vector2 startROI, Vector2 endROI, cv::Mat &ROIMask, cv::Mat &ROIMask_drr);


private:
	/*Defined by Julio*/
	cv::Mat TotalMask;
	cv::Mat TotalMask_drr;
	cv::Mat CircleMask;
	cv::Mat CircleMask_drr;
	cv::Mat ROIMask;
	cv::Mat ROIMask_drr;	

	//flags
	bool isAutoApply = false;
	bool isModuleInitialized = false;
	bool isPositiveOutput = true;
	bool useROI;
	bool useCircleMask;
	bool useMask;
	bool newMaskAvailable;

	//Field variables
	BoolField*		_supported;
	Vector3Field*	PositionFld;
	Vector3Field*	OrientationFld;
	FloatField*		SourceDistFld;
	BoolField*		isMaskedOutputfld;
	BoolField*		MaskAutoSizeFld;
	Vector2Field*	MaskCenterFld;
	IntField*		MaskRadiusFld;
	BoolField*		isPositiveOutputfld;
	BoolField*		useCUDAFld;
	Vector3Field*	ResolutionFld;
	Vector3Field*	PixelDensityFld;
	BoolField*		QuaternionsFld;
	IntField*		DRRMinPixelValueFld;
	IntField*		DRRMaxPixelValueFld;

	BoolField*		isAutoApplyFld;
	TriggerField*	ApplyFld;
	MLDataType      _outType;
	BoolField*		useROIFld;
	Vector2Field*	startROIFld;
	Vector2Field*	endROIFld;

	// Region of interest variables
	Vector2	startROI;
	Vector2	endROI;

	//Eigen variable-size matrixes
	MatrixXf imageOut;

	//Eigen tensors
	Tensor<float, 3> imageIn;
	Image3D CT_Scan;

	//Eigen variable-size vectors
	VectorXf object3D_eigen;

	// Default pixel spacing in the iso-center plane in mm
	float im_sx;
	float im_sz;
	float res_sx;
	float res_sz;

	// Size of the output image in number of pixels
	//the dx and dz variables must have an initial value of the size of the CT in X and Y axes, respectively. (Yes, Y instead of Z)
	int dx;
	int dz;

	//The central axis positions of the 2D images in continuous indices
	float o2D[2] = { 0,0 };

	//pointer variables for the CUDA calculations
	float* object3D;

  // Implements interface for the runtime type system of the ML.
  ML_MODULE_CLASS_HEADER(DRR_CUDA)
};


ML_END_NAMESPACE