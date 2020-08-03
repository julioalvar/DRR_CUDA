#include "mlDRR_CUDA.h"


ML_START_NAMESPACE

//Reading input 3D object, both for serial and parallel design
//template <typename T>
//Tensor<float, 3> DRR_CUDA::inputImageToEigenMatrix(TSubImage<T>* inputSubImage)
bool PROJECT_CLASS_NAME::inputImageToEigenMatrix(unsigned inputNumber, float *_o2D, Image3D &CT_Scan_structure, Tensor<float, 3>& inputImage3D, VectorXf &inputImage3D_eigen)
{
	bool reply = false;

	if ((getUpdatedInputImage(inputNumber) != NULL) && getUpdatedInputImage(inputNumber)->isValid())
	{
		const ml::PagedImage* pi = getUpdatedInputImage(inputNumber);

		// always requires a valid input image
		ML_CHECK(pi);

		ml::SubImageBox sib = pi->getBoxFromImageExtent();

		const ImageVector iSize = pi->getImageExtent();

		//cout << iSize.x << " " << iSize.y << " " << iSize.z << endl;

		if (!(iSize == Vector(0, 0, 0, 0, 0, 0)))
		{
			// Init the slice data pointer here since we want to reuse the allocated slice memory later.
			void *sliceData = NULL;

			// get sub image box for current slice
			ml::SubImageBox sliceSib = SubImageBox(Vector(0, 0, 0, 0, 0, 0), Vector(iSize.x - 1, iSize.y - 1, iSize.z - 1, 0, 0, 0));

			int depth_image = static_cast<int> (iSize.z);
			int height_image = static_cast<int> (iSize.y);
			int width_image = static_cast<int> (iSize.x);

			//dimensions in pixels
			float dimensionZ = depth_image;
			float dimensionY = height_image;
			float dimensionX = width_image;

			//cout << dimensionZ << " " << dimensionY << " " << dimensionX << endl;

			inputImage3D.resize(depth_image, height_image, width_image);//z,y,x

			MLDataType imageDT = pi->getDataType();
			// Get the current slice from the buffer.  
			MLErrorCode err = getTile(getInputImage(inputNumber), sliceSib, imageDT, &sliceData);

			TSubImage<MLuint16> slice(sliceSib.getExtent(), pi->getDataType(), sliceData);

			float minBrightness = static_cast<float>(pi->getMinVoxelValue());

			// use overall max brightness maximum instead of slice-maximum
			float maxBrightness = static_cast<float>(pi->getMaxVoxelValue());

			err;
			minBrightness;
			maxBrightness;
			//uchar val;

			const unsigned int numPixels = static_cast<int>(iSize.x * iSize.y);
			cv::Mat aux = cv::Mat::zeros(static_cast<int>(iSize.x), static_cast<int>(iSize.y), CV_8UC1);

			for (unsigned int z = 0; z < iSize.z; ++z)
			{
				for (unsigned int y = 0; y < iSize.y; ++y)
				{
					for (unsigned int x = 0; x < iSize.x; ++x)
					{
						inputImage3D(z, y, x) = slice.getImageValue(x, y, z);
					}
				}
			}

			reply = true;
			// Release the allocated memory.
			if (sliceData)
			{
				freeTile(sliceData);
				sliceData = NULL;
			}

			Vector3d ctPixelSpacing;
			readDICOMTagFromInputImage(inputNumber, ctPixelSpacing);
			
			//Moving the whole 3D object so that the imOrigin is at (0,0,0)
			Vector3f imOrigin = Vector3f();

			Vector3d isocenter;

			//Setting the center of the CT object as the isocenter *commented by Julio
			/*isocenter[0] = imOrigin[0] + (ctPixelSpacing[0] * (dimensionZ / 2));
			isocenter[1] = imOrigin[1] + (ctPixelSpacing[1] * (dimensionY / 2));
			isocenter[2] = imOrigin[2] + (ctPixelSpacing[2] * (dimensionX / 2));*/			

			//Setting the center of the CT object as the isocenter
			isocenter[0] = ctPixelSpacing[0] * ((dimensionZ + 1) / 2);
			isocenter[1] = ctPixelSpacing[1] * ((dimensionY + 1) / 2);
			isocenter[2] = ctPixelSpacing[2] * ((dimensionX + 1) / 2);


			//cout << isocenter[0] << " " << isocenter[1] << " " << isocenter[2] << endl;

			//Initial value of the center of the 2D image
			_o2D[0] = isocenter[2]; //o2Dx
			_o2D[1] = isocenter[0]; //o2Dz

			//Check if CUDA is supported
			int deviceCount;
			cudaError_t errorId = cudaGetDeviceCount(&deviceCount);
			if (errorId == cudaSuccess) {
				_supported->setValue(true);
			}
			else {
				_supported->setValue(false);
				mlInfo("CUDAdrr") << "cudaGetDeviceCount returned " << static_cast<int>(errorId) << " (" << cudaGetErrorString(errorId) << ")" << std::endl;
			}

			//If CUDA is supported, transfer the 3D object into a flatten object (Eigen tensor)
			if (_supported)
			{
				inputImage3D_eigen.resize(width_image*height_image*depth_image);
				for (int i = 0; i < depth_image; i++) {
					for (int j = 0; j < height_image; j++) {
						for (int k = 0; k < width_image; k++) {
							inputImage3D_eigen[k + (j*width_image) + (i*width_image*height_image)] = inputImage3D(i, j, k);
						}
					}
				}

				CT_Scan_structure.image = inputImage3D_eigen.data();
				CT_Scan_structure.PixelSpacingCT[0] = ctPixelSpacing[0];
				CT_Scan_structure.PixelSpacingCT[1] = ctPixelSpacing[1];
				CT_Scan_structure.PixelSpacingCT[2] = ctPixelSpacing[2];
				CT_Scan_structure.SizeCT[0] = dimensionZ;
				CT_Scan_structure.SizeCT[1] = dimensionY;
				CT_Scan_structure.SizeCT[2] = dimensionX;
				CT_Scan_structure.isoCenter[0] = isocenter[0];
				CT_Scan_structure.isoCenter[1] = isocenter[1];
				CT_Scan_structure.isoCenter[2] = isocenter[2];

			}
		}
	}

	return reply;
}


void PROJECT_CLASS_NAME::CreateROIForOutputs(int dx, int dz, bool useROI, Vector2 startROI, Vector2 endROI, cv::Mat &ROIMask, cv::Mat &ROIMask_drr)
{

	ROIMask = cv::Mat(dx, dz, CV_8UC1);
	ROIMask_drr = cv::Mat(dx, dz, CV_8UC1);
	ROIMask = 0; ROIMask_drr = 0;

	if (useROI)
	{
		cv::Point Pt1 = cv::Point(startROI.y, startROI.x);
		cv::Point Pt2 = cv::Point(endROI.y, endROI.x );
		cv::Point Pt1drr = cv::Point(dz - startROI.y, startROI.x);
		cv::Point Pt2drr = cv::Point(dz - endROI.y, endROI.x);
		rectangle(ROIMask, Pt1, Pt2, 255, CV_FILLED);
		rectangle(ROIMask_drr, Pt1drr, Pt2drr, 255, CV_FILLED);
	}
	else
	{
		ROIMask = 255; 
		ROIMask_drr = 255;
	}

}


void PROJECT_CLASS_NAME::CreateCircleMaskForOutputs(int dx, int dz, bool useCircleMask, cv::Mat &CircleMask, cv::Mat &CircleMask_drr)
{

	if ((dx == 0) && (dz == 0))
	{
		CircleMask = cv::Mat(512, 512, CV_8UC1, 255);
		CircleMask_drr = cv::Mat(512, 512, CV_8UC1, 255);
	}
	else
	{
		CircleMask = cv::Mat(dx, dz, CV_8UC1, 255);
		CircleMask_drr = cv::Mat(dx, dz, CV_8UC1, 255);
	}

	if (useCircleMask)
	{
		CircleMask = 0;  CircleMask_drr = 0;
		cv::Point MaskCircleCenter = CircleMask.size() / 2;
		int MaskCircleRadius = (CircleMask.size().height / 2);// -10;
		circle(CircleMask, MaskCircleCenter, MaskCircleRadius, 255, CV_FILLED);
		circle(CircleMask_drr, MaskCircleCenter, MaskCircleRadius, 255, CV_FILLED);
	}


}

void PROJECT_CLASS_NAME::CreateCircleMaskForOutputs(int dx, int dz, bool useCircleMask, double centerX, double centerY, double radius, cv::Mat &CircleMask, cv::Mat &CircleMask_drr)
{
	CircleMask = cv::Mat(dx, dz, CV_8UC1,255);
	CircleMask_drr = cv::Mat(dx, dz, CV_8UC1,255);

	if (useCircleMask)
	{
		CircleMask = 0; CircleMask_drr = 0;
		cv::Point MaskCircleCenter = cv::Point(centerX, centerY);
		cv::Point MaskCircleCenter_drr = cv::Point(dz - centerX, centerY);
		int MaskCircleRadius = static_cast<int>(radius);
		circle(CircleMask, MaskCircleCenter, MaskCircleRadius, 255, CV_FILLED);
		circle(CircleMask_drr, MaskCircleCenter_drr, MaskCircleRadius, 255, CV_FILLED);
	}

}

void PROJECT_CLASS_NAME::ProcessOutputBeforeRendering(int minDRR, int maxDRR, bool isPosiveOutput, bool useMask, cv::Mat TotalMask, MatrixXf &imageOut)
{
	//auto t1 = std::chrono::high_resolution_clock::now();
	
		// Converting image with DRR from Eigen to OpenCV
	cv::Mat imageOutCV;
	cv::eigen2cv(imageOut, imageOutCV);

	// This type convertion is required for creating a positive image
	if (maxDRR > 255)
		imageOutCV.convertTo(imageOutCV, CV_16UC1);
	else
		imageOutCV.convertTo(imageOutCV, CV_8UC1);


	//Making output positive or negative color
	if (isPosiveOutput)
		imageOutCV = ~imageOutCV;

	// Normalizing image grayscale
	double min, max;
	cv::minMaxLoc(imageOutCV, &min, &max, NULL, NULL, TotalMask);


	double alpha = (double(maxDRR - minDRR)) / (max - min);
	double beta = double(minDRR) - (double(maxDRR - minDRR))*min / (max - min);

	cv::Mat NormalizedimageOutCV;
	if (maxDRR > 255)
		imageOutCV.convertTo(NormalizedimageOutCV, CV_16UC1, alpha, beta);
	else
		imageOutCV.convertTo(NormalizedimageOutCV, CV_8UC1, alpha, beta);




	// Applying mask to the OpenCV image output
	cv::Mat MaskedNormalizedimageOutCV;
	if (useMask)
		NormalizedimageOutCV.copyTo(MaskedNormalizedimageOutCV, TotalMask);
	else
		NormalizedimageOutCV.copyTo(MaskedNormalizedimageOutCV);


	// Converting back the result from OpenCV to Eigen
	cv::cv2eigen(MaskedNormalizedimageOutCV, imageOut);


	//auto t2 = std::chrono::high_resolution_clock::now();
	//auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
	//cout << "Output preprocessing[ms] = " << duration << endl;
}

cv::Mat PROJECT_CLASS_NAME::translateImg(cv::Mat img, double offsetx, double offsety)
{
	cv::Mat trans_mat = (cv::Mat_<double>(2, 3) << 1, 0, offsetx, 0, 1, offsety);
	cv::warpAffine(img, img, trans_mat, img.size());
	return img;
}

ML_END_NAMESPACE