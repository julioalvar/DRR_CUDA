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

#include "mlDRR_CUDA.h"


ML_START_NAMESPACE

//! Implements code for the runtime type system of the ML
ML_MODULE_CLASS_SOURCE(DRR_CUDA, Module);

//----------------------------------------------------------------------------------

DRR_CUDA::DRR_CUDA() : Module(1, 1)
{
  // Suppress calls of handleNotification on field changes to
  // avoid side effects during initialization phase.
  handleNotificationOff();

  //Fields from the GUI
  PositionFld = addVector3("Position", Vector3());
  OrientationFld = addVector3("Orientation", Vector3());
  SourceDistFld = addFloat("Source_Detector_Distance", 970);
  ResolutionFld = addVector3("Resolution", Vector3(568, 568, 0));
  PixelDensityFld = addVector3("Resolution_in_mm", Vector3(207.152, 207.152, 0));
  isMaskedOutputfld = addBool("MaskedOutput",true);
  MaskAutoSizeFld = addBool("MaskAutoSize", true);
  MaskCenterFld = addVector2("MaskCenter", Vector2(284, 284));
  MaskRadiusFld = addInt("MaskRadius", 284);
  isPositiveOutputfld = addBool("PositiveOutput", true);
  useCUDAFld = addBool("UseCUDA",true);

  useROIFld = addBool("UseROI", false);
  startROIFld = addVector2("startROI", Vector2());
  endROIFld = addVector2("endROI", Vector2(568, 568));

  //Quaternion controls
  QuaternionsFld = addBool("Enable_Quaternions");

  _supported = addBool("supported");

  DRRMinPixelValueFld = addInt("DRRMinPixelValue", 0);
  DRRMaxPixelValueFld = addInt("DRRMaxPixelValue", 255);


  isAutoApplyFld = addBool("AutoApply", false);
  ApplyFld = addTrigger("Apply");
  _outType = MLuint16Type;

  dx = 568;
  dz = 568;
  loadOuputVariablesInGPUMemory(dx,dz);
  prepareDRROutputVariables();
  updateROIMask();
  
  useROI = false;
  useCircleMask = true;
  useMask = true;
  newMaskAvailable = true;
  // Reactivate calls of handleNotification on field changes.
  handleNotificationOn();


}

//----------------------------------------------------------------------------------

void DRR_CUDA::handleNotification(Field* field)
{
	/*
	The information that comes from the GUI uses X as [0], Y as [1] and Z as [2]
	while this algorithm uses X as [2], Y as [1] and Z as [0]
	Therefore, a conversion must be performed when reading the values
	*/

	// Handle changes of module parameters and input image fields here.
	bool touchOutputs = false;

	// If there is something connected to the input field, read it once
	if (isInputImageField(field))
	{		
		
	}

	if (field == getInputImageField(0))
	{
		if (inputImageToEigenMatrix(0, o2D, CT_Scan, imageIn, object3D_eigen))
		{
			freeDICOMFromGPUMemory();
			loadDICOMInGPUMemory(object3D_eigen.data(), CT_Scan.SizeCT, CT_Scan.PixelSpacingCT);// sizeCT[0] * sizeCT[1] * sizeCT[2]);
		}
	}


	if (field == ResolutionFld || field == PixelDensityFld)
	{
		prepareDRROutputVariables();
	}

	if (field == isAutoApplyFld)
	{
		isAutoApply = isAutoApplyFld->getBoolValue();
	}

	if (field == isMaskedOutputfld || field == MaskAutoSizeFld || field == MaskRadiusFld || field == MaskCenterFld)
	{
		updateCircleMask();
	}

	if (field == useROIFld || field == startROIFld || field == endROIFld)
	{
		updateROIMask();
	}

	if (field == isMaskedOutputfld || field == useROIFld)
	{
		useMask = (useCircleMask || useROI);
		updateMaskFlagInGPUMemory(useMask);
	}

	if (newMaskAvailable)
	{
		newMaskAvailable = false;
		if (useMask) 
			updateMask();
		
	}


	if (field == PositionFld || field == OrientationFld || field == SourceDistFld ||  field == QuaternionsFld || field == ApplyFld)
	{
		if (isAutoApply || (field == ApplyFld))
		{
			//auto t1 = std::chrono::high_resolution_clock::now();			
			float source_detector = SourceDistFld->getValue();
			float scd = source_detector / 2; //distance from source to isocenter

			Vector3 Pos = PositionFld->getValue();
			Vector3 Ori = OrientationFld->getValue();

			bool useQuat = QuaternionsFld->getValue();
			
			/*
			If CUDA is supported, use the GPU for the siddon-jacobs algorithm
			If CUDA is not supported, use the CPU for all the calculations
			*/
			float imagePixelDim[2] = { im_sx ,im_sz };
			vector<float> _translation	= { static_cast<float>(Pos[0]),static_cast<float>(Pos[1]),static_cast<float>(Pos[2])};
			vector<float> _rotation		= { static_cast<float>(Ori[0]),static_cast<float>(Ori[1]),static_cast<float>(Ori[2]) };
			bool flag_using_CUDA = _supported && useCUDAFld->getBoolValue();
 
			getDRR(imageOut, _translation, _rotation, scd, CT_Scan, useQuat, imagePixelDim, o2D, useROI, startROI, endROI, flag_using_CUDA);			 

			isModuleInitialized = true; // The initialization ends when the user can create a DRR
			touchOutputs = true;

			/*auto t2 = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
			cout << "drr calculation time[us] = " << duration << endl;*/
		}
	}


	if (touchOutputs)
	{
		// Touch all output image fields to notify connected modules.
		touchOutputImageFields();
		
	}
}


void PROJECT_CLASS_NAME::prepareDRROutputVariables()
{
	Vector3 Res = ResolutionFld->getValue();
	Vector3 Den = PixelDensityFld->getValue();

	res_sx = Den[0];
	res_sz = Den[1];

	if (!((dx == Res[0]) && (dz == Res[1])))
	{
		dx = Res[0];
		dz = Res[1];

		freeAuxiliaryVariablesInGPUMemory();
		loadOuputVariablesInGPUMemory(dx,dz);
	}

	//initialize the input/output size
	//if ((dx == 0) && (dz == 0) && initSize)
	if ((dx == 0) && (dz == 0))
	{
		dx = CT_Scan.SizeCT[2];
		dz = CT_Scan.SizeCT[1];
	}

	im_sx = res_sx / dx;
	im_sz = res_sz / dz;

	
	//resizing all the matrixes to generate the output with the right size
	imageOut.conservativeResize(dx, dz);

	updateCircleMask();	
	updateROIMask();

	// Changing the output type
	if (DRRMaxPixelValueFld->getIntValue() > 255)
		_outType = MLuint16Type;
	else
		_outType = MLuint8Type;

	//------------------------------------
}

void PROJECT_CLASS_NAME::updateCircleMask()
{
	useCircleMask = isMaskedOutputfld->getValue();
	if (!MaskAutoSizeFld->getValue())	
		CreateCircleMaskForOutputs(dx, dz, useCircleMask, MaskCenterFld->getValue()[1], MaskCenterFld->getValue()[0], MaskRadiusFld->getValue(), CircleMask, CircleMask_drr);
	else			
		CreateCircleMaskForOutputs(dx, dz, useCircleMask, CircleMask, CircleMask_drr);

	newMaskAvailable = true;
}

void PROJECT_CLASS_NAME::updateROIMask()
{
	useROI = useROIFld->getBoolValue();
	startROI = startROIFld->getValue();
	endROI = endROIFld->getValue();
	CreateROIForOutputs(dx, dz, useROI, startROI, endROI, ROIMask, ROIMask_drr);
	newMaskAvailable = true;
}

void PROJECT_CLASS_NAME::updateMask()
{
	// TotalMask goes to the output that continues with openCV
	bitwise_and(CircleMask, ROIMask, TotalMask);

	// TotalMask_drr is the variable that is loaded to the constant memory in CUDA
	bitwise_and(CircleMask_drr, ROIMask_drr, TotalMask_drr);

	// There is half pixel that must be compensated because of two differen mixes of openCV, eigen and cuda
	cv::Mat shifteddMask = translateImg(TotalMask_drr, -0.5, 0);

	// OpenCV and Eigen manipulate matrices in different ways (rows as cols), so a transposition is required
	cv::Mat maskTranspose = shifteddMask.t();

	// Loading value to constant memory in the GPU
	loadMaskInGPUMemory(maskTranspose.data, dx, dz,useMask);
}

//----------------------------------------------------------------------------------

void DRR_CUDA::calculateOutputImageProperties(int /*outputIndex*/, PagedImage* outputImage)
{
  // Change properties of output image outputImage here whose
  // defaults are inherited from the input image 0 (if there is one).

  // Since the input is 3D and the output 2D, the image extent must
  // be declared to have only two dimensions. 
	//outputImage->setImageExtent(ImageVector(dx, dz, 1, 1, 1, 1)); //output image (2D)
	
	Vector3 _res = validateZeroVector(ResolutionFld->getValue());
	Vector3 _res_mm = validateZeroVector(PixelDensityFld->getValue());
	Vector3 _density = Vector3(_res_mm.x/_res.x, _res_mm.y / _res.y, 1);
	Matrix4 _worldMatrix = Matrix4();
	_worldMatrix[0][0] = _density[0]; 
	_worldMatrix[1][1] = _density[1];


	outputImage->setImageExtent(ImageVector(_res[0], _res[1], 1, 1, 1, 1)); //output image (2D)
	outputImage->setPageExtent(ImageVector(dx, dz, CT_Scan.SizeCT[0], 1, 1, 1)); //input object (3D)

	outputImage->setVoxelSize(_density);
	outputImage->setVoxelToWorldMatrix(_worldMatrix);
	outputImage->setDataType(_outType);
	outputImage->setMinVoxelValue(DRRMinPixelValueFld->getIntValue());
	outputImage->setMaxVoxelValue(DRRMaxPixelValueFld->getIntValue());
}

//----------------------------------------------------------------------------------

SubImageBox DRR_CUDA::calculateInputSubImageBox(int inputIndex, const SubImageBox& outputSubImageBox, int outputIndex)
{
  // Return region of input image inputIndex needed to compute region
  // outSubImgBox of output image outputIndex.
	inputIndex;
	outputIndex;
  return outputSubImageBox;
}


//----------------------------------------------------------------------------------

ML_CALCULATEOUTPUTSUBIMAGE_NUM_INPUTS_1_SCALAR_TYPES_CPP(DRR_CUDA);

template <typename T>
void DRR_CUDA::calculateOutputSubImage(TSubImage<T>* outputSubImage, int outputIndex
                                     , TSubImage<T>* inputSubImage0
                                     )
{
	// Compute sub-image of output image outputIndex from input sub-images.
	outputIndex; // included line to remove warning during compilation time
	inputSubImage0;
				 //if there is a new input object, call the function to read the dicom object once, which is transformed into a 3D Tensor 

	if (isModuleInitialized)
	{
		int minDRR = DRRMinPixelValueFld->getIntValue();
		int maxDRR = DRRMaxPixelValueFld->getIntValue();

		// This function converts the imageOut to an openCV image, it process the image, and it returns the results to imageOut
		ProcessOutputBeforeRendering(minDRR, maxDRR, isPositiveOutputfld->getValue(), useMask, TotalMask, imageOut);

		//rendering the 2D image output provided by the DRR function	
		EigenMatricesToOutputImage(outputSubImage, maxDRR, imageOut);
	}
}


//Generation of the module's output
template <typename T>
void PROJECT_CLASS_NAME::EigenMatricesToOutputImage(TSubImage<T>* _outputSubImage, int maxValue, MatrixXf imageOut)
{
	// Process all voxels of the valid region of the output page.
	ImageVector p;
	T*  outVoxel = _outputSubImage->getImagePointer(p);

	if (maxValue > 255)
	{
		for (int i = 0; i < imageOut.size(); i++, ++outVoxel)
			*outVoxel = static_cast<MLuint16> (imageOut(i));
	}
	else
	{
		for (int i = 0; i < imageOut.size(); i++, ++outVoxel)
			*outVoxel = static_cast<MLuint8> (imageOut(i));
	}
}



Vector3 PROJECT_CLASS_NAME::validateZeroVector(Vector3 in_vector)
{
	if (in_vector[0] == 0)
		in_vector[0] = 1;

	if (in_vector[1] == 0)
		in_vector[1] = 1;

	if (in_vector[2] == 0)
		in_vector[2] = 1;

	return in_vector;
}

ML_END_NAMESPACE