#include "mlDRR_CUDA.h"

ML_START_NAMESPACE

void PROJECT_CLASS_NAME::loadVariablesForExecuteDRR(DRRParameters &DRR_parameters, Image2D &DRRoutput, vector<float> _translation, vector<float> _rotation, float _scd, int imageDim[2], float imagePixelDim[2], float _isocenter[3], float _o2D[2], bool _useROI, Vector2 _startROI, Vector2 _endROI, bool _useQuat)
{
	/*
	INSIDE THIS FUNCTION:
	[0] is Z
	[1] is Y
	[2] is X
	*/

	float _dx = imageDim[0];
	float _dz = imageDim[1];
	float _im_sx = imagePixelDim[0];
	float _im_sz = imagePixelDim[1];

	Vector3f origin;
	Vector3f SourceWorldOriginal;
	Vector3f SourceWorld;
	Vector3f corner00;
	Vector3f corner01;
	Vector3f corner10;
	Vector3f corner11;
	Vector3f cornerOriginal00;
	Vector3f cornerOriginal01;
	Vector3f cornerOriginal10;
	Vector3f cornerOriginal11;

	Vector3f vectorInX;
	Vector3f vectorInZ;
	Vector3f stepInX;
	Vector3f stepInZ;
	
	// commented by Julio
	//origin vector is the corner00 of the 2D image
	origin[0] = _o2D[1] - (_im_sz * ((_dz - 1) / 2));
	origin[1] = _isocenter[1] - _scd;
	origin[2] = _o2D[0] - (_im_sx * ((_dx - 1) / 2));


	//Four-points method
	//------------------------------------------------------------------------
	//Corner 0,0
	cornerOriginal00[0] = origin[0];
	cornerOriginal00[1] = origin[1];
	cornerOriginal00[2] = origin[2];

	//Corner 0,1
	cornerOriginal01[0] = cornerOriginal00[0];
	cornerOriginal01[1] = cornerOriginal00[1];
	cornerOriginal01[2] = cornerOriginal00[2] + (_im_sx * (_dx - 1));

	//Corner 1,0
	cornerOriginal10[0] = cornerOriginal00[0] + (_im_sz * (_dz - 1));
	cornerOriginal10[1] = cornerOriginal00[1];
	cornerOriginal10[2] = cornerOriginal00[2];

	//Source original location
	SourceWorldOriginal[0] = _isocenter[0];
	SourceWorldOriginal[1] = _isocenter[1] + _scd;
	SourceWorldOriginal[2] = _isocenter[2];

	vector<float> isoC = { static_cast<float>(_isocenter[0]) ,static_cast<float>(_isocenter[1]) ,static_cast<float>(_isocenter[2]) };
	if (_useQuat == 1) {
		Vector3 cornerOrig00, cornerOrig01, cornerOrig10, SourceP;

		//Convesion from Vector3f to Vector3 (required for the quaternion calculations)
		cornerOrig00[0] = cornerOriginal00[0];
		cornerOrig00[1] = cornerOriginal00[1];
		cornerOrig00[2] = cornerOriginal00[2];
		cornerOrig01[0] = cornerOriginal01[0];
		cornerOrig01[1] = cornerOriginal01[1];
		cornerOrig01[2] = cornerOriginal01[2];
		cornerOrig10[0] = cornerOriginal10[0];
		cornerOrig10[1] = cornerOriginal10[1];
		cornerOrig10[2] = cornerOriginal10[2];
		SourceP[0] = SourceWorldOriginal[0];
		SourceP[1] = SourceWorldOriginal[1];
		SourceP[2] = SourceWorldOriginal[2];

		corner00 = getTransAndRotQuaternions(cornerOrig00, _translation, _rotation, isoC);
		corner01 = getTransAndRotQuaternions(cornerOrig01, _translation, _rotation, isoC);
		corner10 = getTransAndRotQuaternions(cornerOrig10, _translation, _rotation, isoC);
		SourceWorld = getTransAndRotQuaternions(SourceP, _translation, _rotation, isoC);

	}
	if (_useQuat == 0) {
		corner00 = getTranslationAndRotation(cornerOriginal00, _translation, _rotation, isoC);
		corner01 = getTranslationAndRotation(cornerOriginal01, _translation, _rotation, isoC);
		corner10 = getTranslationAndRotation(cornerOriginal10, _translation, _rotation, isoC);
		SourceWorld = getTranslationAndRotation(SourceWorldOriginal, _translation, _rotation, isoC);
	}
	//------------------------------------------------------------------------

	//Pixel-step method
	//------------------------------------------------------------------------
	vectorInX[0] = corner01[0] - corner00[0];
	vectorInX[1] = corner01[1] - corner00[1];
	vectorInX[2] = corner01[2] - corner00[2];
	stepInX[0] = vectorInX[0] / ((float)_dx);
	stepInX[1] = vectorInX[1] / ((float)_dx);
	stepInX[2] = vectorInX[2] / ((float)_dx);
	vectorInZ[0] = corner10[0] - corner00[0];
	vectorInZ[1] = corner10[1] - corner00[1];
	vectorInZ[2] = corner10[2] - corner00[2];
	stepInZ[0] = vectorInZ[0] / ((float)_dz);
	stepInZ[1] = vectorInZ[1] / ((float)_dz);
	stepInZ[2] = vectorInZ[2] / ((float)_dz);
	//------------------------------------------------------------------------

	//Image2D DRRoutput;
	DRRoutput.size[0] = _dx;
	DRRoutput.size[1] = _dz;

	int _ROI[4] = { static_cast<int>(_startROI.x), static_cast<int>(_startROI.y), static_cast<int>(_endROI.x), static_cast<int>(_endROI.y) };

	//DRRParameters DRR_parameters;
	DRR_parameters.corner00[0] = corner00[0]; DRR_parameters.corner00[1] = corner00[1]; DRR_parameters.corner00[2] = corner00[2];
	DRR_parameters.SourceWorld[0] = SourceWorld[0]; DRR_parameters.SourceWorld[1] = SourceWorld[1]; DRR_parameters.SourceWorld[2] = SourceWorld[2];
	DRR_parameters.stepInX[0] = stepInX[0]; DRR_parameters.stepInX[1] = stepInX[1]; DRR_parameters.stepInX[2] = stepInX[2];
	DRR_parameters.stepInY[0] = stepInZ[0]; DRR_parameters.stepInY[1] = stepInZ[1]; DRR_parameters.stepInY[2] = stepInZ[2];
	DRR_parameters.ROI[4] = static_cast<int>(_useROI);
	DRR_parameters.ROI[0] = _ROI[0]; DRR_parameters.ROI[1] = _ROI[1]; DRR_parameters.ROI[2] = _ROI[2]; DRR_parameters.ROI[3] = _ROI[3];
}

void PROJECT_CLASS_NAME::getDRR(MatrixXf &_imageOut, vector<float> _translation, vector<float> _rotation , float _scd, Image3D _CT_Scan, bool _useQuat, float imagePixelDim[2], float _o2D[2], bool _useROI, Vector2 _startROI, Vector2 _endROI, bool calculateUsingCUDA)
{
	DRRParameters DRR_parameters;
	Image2D	DRRoutput;

	int imageDim[2] = { static_cast<int>(_imageOut.rows()) , static_cast<int>(_imageOut.cols()) };

	loadVariablesForExecuteDRR(DRR_parameters, DRRoutput, _translation, _rotation, _scd, imageDim, imagePixelDim, _CT_Scan.isoCenter, _o2D, _useROI, _startROI, _endROI, _useQuat);

	if (calculateUsingCUDA)
		getDRRwithCUDA(DRR_parameters, DRRoutput, _imageOut);
	else
		getDRRSerial(DRR_parameters, DRRoutput, _imageOut);
}



//DRR parallel design
void PROJECT_CLASS_NAME::getDRRwithCUDA(DRRParameters DRR_parameters, Image2D DRRoutput, MatrixXf &_imageOut)
{
	/*
	INSIDE THIS FUNCTION:
	[0] is Z
	[1] is Y
	[2] is X
	*/

	//Creating the vector to store the result of the calculations after returning from the CUDA host function
	long int vectorSize = DRRoutput.size[0]* DRRoutput.size[1];
	std::vector<float> image2D_flat;
	image2D_flat.resize(vectorSize);
	DRRoutput.image = image2D_flat.data();	

	int numThreads = 1024;
	int numBlocks = (int)ceil((float)vectorSize / numThreads);

	CUDAParamerters CUDA_parameters;
	CUDA_parameters.numThreads = numThreads;
	CUDA_parameters.numBlocks = numBlocks;

	//Calling the CUDA host function
	calculateDRRwithCUDA(CT_Scan, DRRoutput, CUDA_parameters, DRR_parameters);

	//Mapping the resulting vector into a matrix
	MatrixXf _rad2Daux = Map<MatrixXf>(DRRoutput.image, DRRoutput.size[0], DRRoutput.size[1]);

	//Mirroring the matrix in Z-axis
	_imageOut = _rad2Daux.rowwise().reverse();

}

//DRR series design
void PROJECT_CLASS_NAME::getDRRSerial(DRRParameters DRR_parameters, Image2D DRRoutput, MatrixXf &_imageOut)
{
	/*
	INSIDE THIS FUNCTION:
	[0] is Z
	[1] is Y
	[2] is X
	*/

	MatrixXf _rad2Daux(DRRoutput.size[0], DRRoutput.size[1]);
	//_imageOut = Map<MatrixXf>(DRRoutput.image, DRRoutput.size[0], DRRoutput.size[1]);

	//variables for siddon-jacobs algorithm
	float alphaX1, alphaXN, alphaXmin, alphaXmax;
	float alphaY1, alphaYN, alphaYmin, alphaYmax;
	float alphaZ1, alphaZN, alphaZmin, alphaZmax;
	float alphaMin, alphaMax;
	float alphaX, alphaY, alphaZ, alphaCmin, alphaCminPrev;
	float alphaUx, alphaUy, alphaUz;
	float d12, value;
	float pixval;
	int iU, jU, kU;
	int cIndexX, cIndexY, cIndexZ;

	Vector3f drrPixelWorld;
	Vector3f rayVector;
	Vector3f firstIntersection;
	Vector3f firstIntersectionIndex;
	Vector3f firstIntersectionIndexUp;
	Vector3f firstIntersectionIndexDown;
	Vector3f alphaIntersectionUp;
	Vector3f alphaIntersectionDown;
	Vector3f _ctPixelSpacing = Vector3f(CT_Scan.PixelSpacingCT[0], CT_Scan.PixelSpacingCT[1], CT_Scan.PixelSpacingCT[2]);
	Vector3f sizeCT = Vector3f(CT_Scan.SizeCT[0], CT_Scan.SizeCT[1], CT_Scan.SizeCT[2]);
	Vector3f cIndex;


	//Nested for-loop for the calculation of every pixel in the series design
	for (int j = 0; j < DRRoutput.size[1]; ++j) {

		if (DRR_parameters.ROI[4])
		{
			if ((j < DRRoutput.size[1] - DRR_parameters.ROI[3] - 1) || (j > DRRoutput.size[1] - DRR_parameters.ROI[2] - 1))
				continue;
		}

		for (int i = 0; i < DRRoutput.size[0]; ++i) {

			if (DRR_parameters.ROI[4])
			{
				if ((i < DRR_parameters.ROI[0]) || (i > DRR_parameters.ROI[1]))
					continue;
			}

			//Calculate the spatial position of the pixel
			drrPixelWorld[0] = DRR_parameters.corner00[0] + DRR_parameters.stepInX[0] * i + DRR_parameters.stepInY[0] * j;
			drrPixelWorld[1] = DRR_parameters.corner00[1] + DRR_parameters.stepInX[1] * i + DRR_parameters.stepInY[1] * j;
			drrPixelWorld[2] = DRR_parameters.corner00[2] + DRR_parameters.stepInX[2] * i + DRR_parameters.stepInY[2] * j;

			//Here starts the implementation of the Siddon-Jacobs algorithm based on ITK
			//Calculate the ray vector
			rayVector[0] = drrPixelWorld[0] - DRR_parameters.SourceWorld[0];
			rayVector[1] = drrPixelWorld[1] - DRR_parameters.SourceWorld[1];
			rayVector[2] = drrPixelWorld[2] - DRR_parameters.SourceWorld[2];

			//Calculate alphaMin and alphaMax 
			alphaXmin = -2; alphaXmax = 2;
			if (rayVector[2] != 0)
			{
				alphaX1 = (0.0 - DRR_parameters.SourceWorld[2]) / rayVector[2];
				alphaXN = (sizeCT[2] * _ctPixelSpacing[2] - DRR_parameters.SourceWorld[2]) / rayVector[2];
				alphaXmin = std::min(alphaX1, alphaXN);
				alphaXmax = std::max(alphaX1, alphaXN);
			}

			alphaYmin = -2;	alphaYmax = 2;
			if (rayVector[1] != 0)
			{
				alphaY1 = (0.0 - DRR_parameters.SourceWorld[1]) / rayVector[1];
				alphaYN = (sizeCT[1] * _ctPixelSpacing[1] - DRR_parameters.SourceWorld[1]) / rayVector[1];
				alphaYmin = std::min(alphaY1, alphaYN);
				alphaYmax = std::max(alphaY1, alphaYN);
			}

			alphaZmin = -2; alphaZmax = 2;
			if (rayVector[0] != 0)
			{
				alphaZ1 = (0.0 - DRR_parameters.SourceWorld[0]) / rayVector[0];
				alphaZN = (sizeCT[0] * _ctPixelSpacing[0] - DRR_parameters.SourceWorld[0]) / rayVector[0];
				alphaZmin = std::min(alphaZ1, alphaZN);
				alphaZmax = std::max(alphaZ1, alphaZN);
			}

			alphaMin = std::max(std::max(alphaXmin, alphaYmin), alphaZmin); //alphaMin: ray enters the volume
			alphaMax = std::min(std::min(alphaXmax, alphaYmax), alphaZmax); //alphaMax: ray leaves the volume

																			//Calculate the first intersection of the ray with the planes (alphaX, alphaY and alphaZ)

			firstIntersection[0] = DRR_parameters.SourceWorld[0] + alphaMin * rayVector[0];
			firstIntersection[1] = DRR_parameters.SourceWorld[1] + alphaMin * rayVector[1];
			firstIntersection[2] = DRR_parameters.SourceWorld[2] + alphaMin * rayVector[2];

			firstIntersectionIndex[0] = firstIntersection[0] / _ctPixelSpacing[0];
			firstIntersectionIndex[1] = firstIntersection[1] / _ctPixelSpacing[1];
			firstIntersectionIndex[2] = firstIntersection[2] / _ctPixelSpacing[2];

			firstIntersectionIndexUp[0] = (int)ceil(firstIntersectionIndex[0]);
			firstIntersectionIndexUp[1] = (int)ceil(firstIntersectionIndex[1]);
			firstIntersectionIndexUp[2] = (int)ceil(firstIntersectionIndex[2]);

			firstIntersectionIndexDown[0] = (int)floor(firstIntersectionIndex[0]);
			firstIntersectionIndexDown[1] = (int)floor(firstIntersectionIndex[1]);
			firstIntersectionIndexDown[2] = (int)floor(firstIntersectionIndex[2]);

			alphaX = 2;
			if (rayVector[2] != 0)
			{
				alphaIntersectionUp[2] = (firstIntersectionIndexUp[2] * _ctPixelSpacing[2] - DRR_parameters.SourceWorld[2]) / rayVector[2];
				alphaIntersectionDown[2] = (firstIntersectionIndexDown[2] * _ctPixelSpacing[2] - DRR_parameters.SourceWorld[2]) / rayVector[2];
				alphaX = std::max(alphaIntersectionUp[2], alphaIntersectionDown[2]);
			}

			alphaY = 2;
			if (rayVector[1] != 0)
			{
				alphaIntersectionUp[1] = (firstIntersectionIndexUp[1] * _ctPixelSpacing[1] - DRR_parameters.SourceWorld[1]) / rayVector[1];
				alphaIntersectionDown[1] = (firstIntersectionIndexDown[1] * _ctPixelSpacing[1] - DRR_parameters.SourceWorld[1]) / rayVector[1];
				alphaY = std::max(alphaIntersectionUp[1], alphaIntersectionDown[1]);
			}

			alphaZ = 2;
			if (rayVector[0] != 0)
			{
				alphaIntersectionUp[0] = (firstIntersectionIndexUp[0] * _ctPixelSpacing[0] - DRR_parameters.SourceWorld[0]) / rayVector[0];
				alphaIntersectionDown[0] = (firstIntersectionIndexDown[0] * _ctPixelSpacing[0] - DRR_parameters.SourceWorld[0]) / rayVector[0];
				alphaZ = std::max(alphaIntersectionUp[0], alphaIntersectionDown[0]);
			}

			//Calculate incremental values (alphaUx, alphaUx, alphaUz) when the ray intercepts the planes
			alphaUx = 999;
			if (rayVector[2] != 0)
				alphaUx = _ctPixelSpacing[2] / fabs(rayVector[2]);

			alphaUy = 999;
			if (rayVector[1] != 0)
				alphaUy = _ctPixelSpacing[1] / fabs(rayVector[1]);

			alphaUz = 999;
			if (rayVector[0] != 0)
				alphaUz = _ctPixelSpacing[0] / fabs(rayVector[0]);


			// Calculate voxel index incremental values along the ray path
			iU = -1;
			if (DRR_parameters.SourceWorld[2] < drrPixelWorld[2])
				iU = 1;

			jU = -1;
			if (DRR_parameters.SourceWorld[1] < drrPixelWorld[1])
				jU = 1;

			kU = -1;
			if (DRR_parameters.SourceWorld[0] < drrPixelWorld[0])
				kU = 1;

			//Initialize the weighted sum to zero
			d12 = 0.0;

			//Initialize current ray position 
			alphaCmin = std::min(std::min(alphaX, alphaY), alphaZ);

			//Initialize the current voxel index
			cIndex[0] = firstIntersectionIndexDown[0];
			cIndex[1] = firstIntersectionIndexDown[1];
			cIndex[2] = firstIntersectionIndexDown[2];

			//The while loop represents when the ray is inside the volume
			while (alphaCmin < alphaMax)
			{
				// Store the current ray position
				alphaCminPrev = alphaCmin;

				if ((alphaX <= alphaY) && (alphaX <= alphaZ))
				{
					//Ray front intercepts with x-plane. Update alphaX
					alphaCmin = alphaX;
					cIndex[2] = cIndex[2] + iU;
					alphaX = alphaX + alphaUx;
				}
				else if ((alphaY <= alphaX) && (alphaY <= alphaZ))
				{
					//Ray front intercepts with y-plane. Update alphaY
					alphaCmin = alphaY;
					cIndex[1] = cIndex[1] + jU;
					alphaY = alphaY + alphaUy;
				}
				else
				{
					//Ray front intercepts with z-plane. Update alphaZ
					alphaCmin = alphaZ;
					cIndex[0] = cIndex[0] + kU;
					alphaZ = alphaZ + alphaUz;
				}


				if ((cIndex[0] >= 0) && (cIndex[0] < (sizeCT[0])) &&
					(cIndex[1] >= 0) && (cIndex[1] < (sizeCT[1])) &&
					(cIndex[2] >= 0) && (cIndex[2] < (sizeCT[2])))
				{
					//If it is a valid index, get the voxel intensity
					cIndexZ = static_cast<int> (cIndex[0]);
					cIndexY = static_cast<int> (cIndex[1]);
					cIndexX = static_cast<int> (cIndex[2]);

					value = imageIn(cIndexZ, cIndexY, cIndexX);
					

					if (value > 0)
						d12 += value*(alphaCmin - alphaCminPrev);//weighted sum					
				}
			}

			pixval = d12;
			if (d12 < 0)
				pixval = 0;

			if (d12 > 255)
				pixval = 255;


			//Mirroring in Z-axis 
			_rad2Daux(i, j) = pixval;
			_imageOut(i, DRRoutput.size[1] - j - 1) = _rad2Daux(i, j);
		}
	}
	//_imageOut = rad2Daux.rowwise().reverse();
	//return _rad2D;
}



//Translation and rotation using Euler
Vector3f PROJECT_CLASS_NAME::getTranslationAndRotation(Vector3f vectorToTransform, vector<float> _translation, vector<float> _rotation, vector<float> isocenter)
{
	/*
	INSIDE THIS FUNCTION:
	[0] is X
	[1] is Y
	[2] is Z
	*/

	//Rotation variables
	double sinRx, sinRy, sinRz, cosRx, cosRy, cosRz;

	//Eigen fixed-size matrixes
	Eigen::Matrix4d translation1;
	Eigen::Matrix4d translation2;
	Eigen::Matrix3d rotation0;
	Eigen::Matrix4d rotation1;
	Eigen::Matrix3d rotX;
	Eigen::Matrix3d rotY;
	Eigen::Matrix3d rotZ;

	Eigen::Vector4d v1Eigen, v2Eigen;

	v1Eigen[0] = vectorToTransform[2]; //reversing X and Z - in this function 0 is X
	v1Eigen[1] = vectorToTransform[1];
	v1Eigen[2] = vectorToTransform[0]; //reversing X and Z - in this function 2 is Z
	v1Eigen[3] = 1;

	

	if ((_rotation[0] >= 89.8) && (_rotation[0] <= 90.2)) {
		sinRz = sin(_rotation[2] *M_PI / 180);
		sinRx = sin((89.8)*M_PI / 180);
		sinRy = sin(_rotation[1] *M_PI / 180);
		cosRz = cos(_rotation[2] *M_PI / 180);
		cosRx = cos((89.8)*M_PI / 180);
		cosRy = cos(_rotation[1] *M_PI / 180);
	}
	else {
		sinRz = sin(_rotation[2] *M_PI / 180);
		sinRx = sin(_rotation[0] *M_PI / 180);
		sinRy = sin(_rotation[1] *M_PI / 180);
		cosRz = cos(_rotation[2] *M_PI / 180);
		cosRx = cos(_rotation[0] *M_PI / 180);
		cosRy = cos(_rotation[1] *M_PI / 180);
	}


	translation1 << 1, 0, 0, -isocenter[2],  //1,  0,  0, -iso
		0, 1, 0, -isocenter[1],              //0,  1,  0, -iso
		0, 0, 1, -isocenter[0],			     //0,  0,  1, -iso
		0, 0, 0, 1;                          //0,  0,  0,  1

	translation2 << 1, 0, 0, +isocenter[2] + _translation[0], //1,  0,  0, +iso+tx
		0, 1, 0, +isocenter[1] + _translation[1],			 //0,  1,  0, +iso+ty
		0, 0, 1, +isocenter[0] + _translation[2],			 //0,  0,  1, +iso+tz
		0, 0, 0, 1;								 //0,  0,  0,    1


	rotX << 1, 0, 0,
		0, cosRx, -sinRx,
		0, sinRx, cosRx;

	rotY << cosRy, 0, sinRy,
		0, 1, 0,
		-sinRy, 0, cosRy;

	rotZ << cosRz, -sinRz, 0,
		sinRz, cosRz, 0,
		0, 0, 1;

	rotation0 = rotZ*rotX*rotY;

	rotation1 << rotation0(0, 0), rotation0(0, 1), rotation0(0, 2), 0,
		rotation0(1, 0), rotation0(1, 1), rotation0(1, 2), 0,
		rotation0(2, 0), rotation0(2, 1), rotation0(2, 2), 0,
		0, 0, 0, 1;

	v2Eigen = translation2*(rotation1*(translation1*v1Eigen));

	Vector3f vectorTransformed;
	vectorTransformed[2] = static_cast <float> (v2Eigen[0]);
	vectorTransformed[1] = static_cast <float> (v2Eigen[1]);
	vectorTransformed[0] = static_cast <float> (v2Eigen[2]);

	return vectorTransformed;
}


//Translation and rotation using Quaternions
Vector3f PROJECT_CLASS_NAME::getTransAndRotQuaternions(Vector3 vectorToTransform, vector<float> translation_, vector<float> _rotation, vector<float> isocenter/*, Vector3 rotationVector, float rotationAngle, float tx, float ty, float tz*/)
{
	/*
	INSIDE THIS FUNCTION:
	[0] is X
	[1] is Y
	[2] is Z
	*/
	Vector3 pointSpaceIso;
	Vector3 pointSpaceOriginal;
	pointSpaceOriginal[0] = vectorToTransform[2];//reversing X and Z - in this function 0 is X
	pointSpaceOriginal[1] = vectorToTransform[1];
	pointSpaceOriginal[2] = vectorToTransform[0];//reversing X and Z - in this function 2 is Z

	Vector3 isoc;
	isoc[0] = isocenter[2];//reversing X and Z - in this function 0 is X
	isoc[1] = isocenter[1];
	isoc[2] = isocenter[0];//reversing X and Z - in this function 2 is Z

	pointSpaceIso = pointSpaceOriginal - isoc;

	//These two variables are necessary because the equations for the quaternions have different axes as the used here.
	//For a better understanding, look at the Figure 3.4 of my written work and compare it to the Figure 1 of the referenced
	//paper of NASA with the title "Euler Angles, Quaternions and Transformation matrices"
	float yr, zr;
	zr = -(_rotation[2]);
	yr = -(_rotation[1] + 180);

	float sinTheta1 = sin((zr / 2)*M_PI / 180);
	float sinTheta2 = sin((_rotation[0] / 2)*M_PI / 180);
	float sinTheta3 = sin((yr / 2)*M_PI / 180);
	float cosTheta1 = cos((zr / 2)*M_PI / 180);
	float cosTheta2 = cos((_rotation[0] / 2)*M_PI / 180);
	float cosTheta3 = cos((yr / 2)*M_PI / 180);

	float q1 = -(sinTheta1*sinTheta2*sinTheta3) + (cosTheta1*cosTheta2*cosTheta3);
	float q2 = -(sinTheta1*cosTheta2*sinTheta3) + (cosTheta1*sinTheta2*cosTheta3);
	float q3 = +(sinTheta1*sinTheta2*cosTheta3) + (cosTheta1*cosTheta2*sinTheta3);
	float q4 = +(sinTheta1*cosTheta2*cosTheta3) + (cosTheta1*sinTheta2*sinTheta3);

	Quaternion quat_p(pointSpaceIso, 0);
	Quaternion quat_p_rotated;
	Vector3 p_rotated;

	Quaternion quat_q(q4, q1, q2, q3);
	Quaternion quat_q_conjugated;
	quat_q_conjugated = quat_q.conjugate();

	quat_p_rotated = (quat_q*quat_p)*quat_q_conjugated;

	p_rotated[0] = quat_p_rotated.qx;
	p_rotated[1] = quat_p_rotated.qy;
	p_rotated[2] = quat_p_rotated.qz;

	Vector3 p_translated;
	Vector3 _translation;
	_translation[0] = translation_[0];
	_translation[1] = translation_[1];
	_translation[2] = translation_[2];

	p_translated = p_rotated + isoc + _translation;

	Vector3f vectorTransformed;
	vectorTransformed[2] = p_translated[0];
	vectorTransformed[1] = p_translated[1];
	vectorTransformed[0] = p_translated[2];

	return vectorTransformed;
}

ML_END_NAMESPACE