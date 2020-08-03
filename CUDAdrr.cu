
#include "CUDAdrr.cuh"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <algorithm> 
#include <iostream>
#include <vector>

#define KERNEL                      __global__
#define HOST                        __host__
#define DEVICE                      __device__
#define HOST_AND_DEVICE             __host__ __device__
#define DEVICE_CONST                __device__ __constant__

// This variable contains the DICOM set
float* d_object3D;

// This variable contains the 2D output from CUDA
float *d_object2D;

// This variable contains the mask output 
unsigned *d_mask2D;

// Constants depending on the DICOM
DEVICE_CONST int d_sizeCT[3];
DEVICE_CONST  float ctPixelSpacing[3];

// Constant depending on image output
DEVICE_CONST int DRRImageSize[2];

// Constants dependion on the specific DRR
DEVICE_CONST  float d_DRR_Parameters[12];

DEVICE_CONST  bool d_useMask;

// This variable contains the DICOM loaded as a Texture ( read-only, fast-cached memory)
cudaTextureObject_t tex_object3D = 0;

// This variable contains the output Mask loaded as a Texture ( read-only, fast-cached memory)
cudaTextureObject_t tex_mask2D = 0;

cudaStream_t stream1;

void loadDICOMInGPUMemory(float *cpp_object3D, int *sizeCT, float *pixelSpacing)
{
	long int object3Dsize = sizeCT[0] * sizeCT[1] * sizeCT[2];
	cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);


	cudaMalloc((void**)&d_object3D, object3Dsize * sizeof(float));
	cudaMemcpyAsync(d_object3D, cpp_object3D, object3Dsize * sizeof(float), cudaMemcpyHostToDevice, stream1);

	cudaMemcpyToSymbol(ctPixelSpacing, pixelSpacing, 3 * sizeof(float), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_sizeCT, sizeCT, 3 * sizeof(int), 0, cudaMemcpyHostToDevice);
	

	// create texture object
	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeLinear;
	resDesc.res.linear.devPtr = d_object3D;
	resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
	resDesc.res.linear.desc.x = 32; // bits per channel
	resDesc.res.linear.sizeInBytes = object3Dsize * sizeof(float);

	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.readMode = cudaReadModeElementType;

	// create CUDA texture object
	cudaDestroyTextureObject(tex_object3D);
	cudaCreateTextureObject(&tex_object3D, &resDesc, &texDesc, NULL);

	cudaStreamDestroy(stream1);

}


void updateMaskFlagInGPUMemory(bool useMask)
{	
	cudaMemcpyToSymbol(d_useMask, &useMask, 1 * sizeof(bool), 0, cudaMemcpyHostToDevice);
}

void loadMaskInGPUMemory(unsigned char* mask_2D, int dimX, int dimZ, bool useMask)
{
	long int mask2DSize = dimX * dimZ;
	cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);		

	cudaMalloc((void**)&d_mask2D, mask2DSize * sizeof(unsigned char));
	cudaMemcpyAsync(d_mask2D, mask_2D, mask2DSize * sizeof(unsigned char), cudaMemcpyHostToDevice, stream1);

	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.readMode = cudaReadModeElementType;

	// create texture object
	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.res.linear.devPtr = d_mask2D;
	resDesc.resType = cudaResourceTypeLinear;	
	resDesc.res.linear.desc.f = cudaChannelFormatKindUnsigned;
	resDesc.res.linear.desc.x = 8; // bits per channel
	resDesc.res.linear.sizeInBytes = mask2DSize * sizeof(unsigned char);	

	// create CUDA texture object
	cudaDestroyTextureObject(tex_mask2D);
	cudaCreateTextureObject(&tex_mask2D, &resDesc, &texDesc, NULL);

	updateMaskFlagInGPUMemory(useMask);

	cudaStreamDestroy(stream1);
}

void loadOuputVariablesInGPUMemory(int dimX, int dimZ)
{
	long int vectorSize = dimX* dimZ;
	int OutputImageSize[2] = { dimX, dimZ };

	cudaMalloc((void**)&d_object2D, vectorSize * sizeof(float));

	cudaMemcpyToSymbol(DRRImageSize, OutputImageSize, 2 * sizeof(int), 0, cudaMemcpyHostToDevice);
}


void freeDICOMFromGPUMemory()
{
	cudaFree(d_object3D);
}

void freeAuxiliaryVariablesInGPUMemory()
{
	cudaFree(d_object2D);
}

							


__global__ void drrCUDA(float* object2D, cudaTextureObject_t tex_object3D, cudaTextureObject_t tex_mask2D)
{

	float stepInX[3] = { d_DRR_Parameters[0],d_DRR_Parameters[1],d_DRR_Parameters[2] };
	float stepInY[3] = { d_DRR_Parameters[3],d_DRR_Parameters[4],d_DRR_Parameters[5] };
	float corner00[3] = { d_DRR_Parameters[6],d_DRR_Parameters[7],d_DRR_Parameters[8] };
	float SourceWorld[3] = { d_DRR_Parameters[9],d_DRR_Parameters[10],d_DRR_Parameters[11] };
	 
	int total_dx = DRRImageSize[0];
	int total_dz = DRRImageSize[1];
	
	//Every thread calculates its own id number
	long int idx = (blockIdx.x*blockDim.x) + threadIdx.x;	

	// This checks if the thread number is bigger than the amount of pixels
	if (idx >= total_dx * total_dz)
		return;

	// Converting number of pixels to rows and columns
	int dz = idx / total_dx;
	int dx = idx - dz*total_dx;


	if (d_useMask)
	{
		//unsigned char maskValue = 0;
		unsigned char maskValue = tex1Dfetch<unsigned char>(tex_mask2D, idx);
		if (maskValue == 0)
		{
			object2D[idx] = 0;
			return;
		}			
	}

	//Calculate the spatial position of the pixel
	//drrPixelWorld_0[idx] = *corner00_0 + ((*stepInX_0)*(threadIdx.x)) + ((*stepInY_0)*(blockIdx.x));
	//drrPixelWorld_1[idx] = *corner00_1 + ((*stepInX_1)*(threadIdx.x)) + ((*stepInY_1)*(blockIdx.x));
	//drrPixelWorld_2[idx] = *corner00_2 + ((*stepInX_2)*(threadIdx.x)) + ((*stepInY_2)*(blockIdx.x));
	float drrPixelWorld[3] = { 0 };
	drrPixelWorld[0] = corner00[0] + ((stepInX[0])*dx) + ((stepInY[0])*dz);
	drrPixelWorld[1] = corner00[1] + ((stepInX[1])*dx) + ((stepInY[1])*dz);
	drrPixelWorld[2] = corner00[2] + ((stepInX[2])*dx) + ((stepInY[2])*dz);

	//Calculate the ray vector
	float rayVector[3] = { 0 };
	rayVector[0] = drrPixelWorld[0] - SourceWorld[0];
	rayVector[1] = drrPixelWorld[1] - SourceWorld[1];
	rayVector[2] = drrPixelWorld[2] - SourceWorld[2];

	float alpha1[3];
	float alphaN[3];
	float auxalphaMin[3] = {-2, -2, -2};
	float auxalphaMax[3] = {2 , 2 , 2};


	//Calculate alphaMin and alphaMax 
	if (rayVector[2] != 0)
	{
		alpha1[0] = (0.0 - (SourceWorld[2])) / rayVector[2];
		alphaN[0] = ((d_sizeCT[2]) * (ctPixelSpacing[2]) - (SourceWorld[2])) / rayVector[2];
		auxalphaMin[0] = alphaN[0];
		auxalphaMax[0] = alpha1[0];

		if (alpha1[0] < alphaN[0]) 
		{
			auxalphaMin[0] = alpha1[0];
			auxalphaMax[0] = alphaN[0];
		}
	}

	if (rayVector[1] != 0)
	{
		alpha1[1] = (0.0 - (SourceWorld[1])) / rayVector[1];
		alphaN[1] = ((d_sizeCT[1]) * (ctPixelSpacing[1]) - (SourceWorld[1])) / rayVector[1];
		auxalphaMin[1] = alphaN[1];
		auxalphaMax[1] = alpha1[1];

		if (alpha1[1] < alphaN[1]) 
		{
			auxalphaMin[1] = alpha1[1];
			auxalphaMax[1] = alphaN[1];
		}
	}


	if (rayVector[0] != 0)
	{
		alpha1[2] = (0.0 - (SourceWorld[0])) / rayVector[0];
		alphaN[2] = ((d_sizeCT[0]) * (ctPixelSpacing[0]) - (SourceWorld[0])) / rayVector[0];
		auxalphaMin[2] = alphaN[2];
		auxalphaMax[2] = alpha1[2];

		if (alpha1[2] < alphaN[2]) 
		{
			auxalphaMin[2] = alpha1[2];
			auxalphaMax[2] = alphaN[2];
		}
	}

	
	float alphaMin;

	if (auxalphaMin[0] > auxalphaMin[1]) //x > y
	{ 
		alphaMin = auxalphaMin[2];
		if (auxalphaMin[0] > alphaMin) { //x > y, x > z
			alphaMin = auxalphaMin[0];
		}
	}
	else //y > x
	{ 
		alphaMin = auxalphaMin[2];
		if (auxalphaMin[1] > alphaMin)  //y > x, y > z
			alphaMin = auxalphaMin[1];
	}

	float alphaMax;

	if (auxalphaMax[0] < auxalphaMax[1])  // x < y
	{
		alphaMax = auxalphaMax[2];
		if (auxalphaMax[0] < alphaMax)  // x < y, x < z
			alphaMax = auxalphaMax[0];	
	}
	else // y < x
	{ 
		alphaMax = auxalphaMax[2];
		if (auxalphaMax[1] < alphaMax)  // y < x, y < z
			alphaMax = auxalphaMax[1];
	}

	float firstIntersection[3], firstIntersectionIndex[3], firstIntersectionIndexUp[3], firstIntersectionIndexDown[3];

	//Calculate the first intersection of the ray with the planes (alphaX, alphaY and alphaZ)
	firstIntersection[0] = (SourceWorld[0]) + (alphaMin * rayVector[0]);
	firstIntersection[1] = (SourceWorld[1]) + (alphaMin * rayVector[1]);
	firstIntersection[2] = (SourceWorld[2]) + (alphaMin * rayVector[2]);
	
	firstIntersectionIndex[0] = firstIntersection[0] / (ctPixelSpacing[0]);
	firstIntersectionIndex[1] = firstIntersection[1] / (ctPixelSpacing[1]);
	firstIntersectionIndex[2] = firstIntersection[2] / (ctPixelSpacing[2]);


	firstIntersectionIndexUp[0] = (int)ceil(firstIntersectionIndex[0]);
	firstIntersectionIndexUp[1] = (int)ceil(firstIntersectionIndex[1]);
	firstIntersectionIndexUp[2] = (int)ceil(firstIntersectionIndex[2]);

	firstIntersectionIndexDown[0] = (int)floor(firstIntersectionIndex[0]);
	firstIntersectionIndexDown[1] = (int)floor(firstIntersectionIndex[1]);
	firstIntersectionIndexDown[2] = (int)floor(firstIntersectionIndex[2]);

	float alpha[3] = {2,2,2}, alphaIntersectionUp[3], alphaIntersectionDown[3];

	if (rayVector[2] != 0)
	{
		alphaIntersectionUp[2] = (firstIntersectionIndexUp[2] * (ctPixelSpacing[2]) - (SourceWorld[2])) / rayVector[2];
		alphaIntersectionDown[2] = (firstIntersectionIndexDown[2] * (ctPixelSpacing[2]) - (SourceWorld[2])) / rayVector[2];
		alpha[0] = alphaIntersectionDown[2];
		if (alphaIntersectionUp[2] > alpha[0])
			alpha[0] = alphaIntersectionUp[2];							
	}

	if (rayVector[1] != 0)
	{
		alphaIntersectionUp[1] = (firstIntersectionIndexUp[1] * (ctPixelSpacing[1]) - (SourceWorld[1])) / rayVector[1];
		alphaIntersectionDown[1] = (firstIntersectionIndexDown[1] * (ctPixelSpacing[1]) - (SourceWorld[1])) / rayVector[1];
		alpha[1] = alphaIntersectionDown[1];
		if (alphaIntersectionUp[1] > alpha[1])
			alpha[1] = alphaIntersectionUp[1];					
	}

	if (rayVector[0] != 0)
	{
		alphaIntersectionUp[0] = (firstIntersectionIndexUp[0] * (ctPixelSpacing[0]) - (SourceWorld[0])) / rayVector[0];
		alphaIntersectionDown[0] = (firstIntersectionIndexDown[0] * (ctPixelSpacing[0]) - (SourceWorld[0])) / rayVector[0];
		alpha[2] = alphaIntersectionDown[0];
		if (alphaIntersectionUp[0] > alpha[2])
			alpha[2] = alphaIntersectionUp[0];					
	}

	float alphaU[3] = { 999,999,999 };
	//Calculate incremental values (alphaUx, alphaUx, alphaUz) when the ray intercepts the planes
	if (rayVector[2] != 0)
		alphaU[0] = (ctPixelSpacing[2]) / fabs(rayVector[2]);	

	if (rayVector[1] != 0)
		alphaU[1] = (ctPixelSpacing[1]) / fabs(rayVector[1]);	

	if (rayVector[0] != 0)
		alphaU[2] = (ctPixelSpacing[0]) / fabs(rayVector[0]);
	

	float U[3] = { -1,-1,-1 };
	// Calculate voxel index incremental values along the ray path
	if ((SourceWorld[2]) < drrPixelWorld[2])
		U[0] = 1;	

	if ((SourceWorld[1]) < drrPixelWorld[1])
		U[1] = 1;

	if ((SourceWorld[0]) < drrPixelWorld[0])
		U[2] = 1;	


	//Initialize the weighted sum to zero
	float d12 = 0.0, alphaCmin, alphaCminPrev;

	//Initialize the current ray position (alphaCmin)
	if (alpha[0] < alpha[1]) //x < y
	{ 
		alphaCmin = alpha[2];
		if (alpha[0] < alphaCmin)  //x < y, x < z
			alphaCmin = alpha[0];
	}
	else //y < x
	{ 
		alphaCmin = alpha[2];
		if (alpha[1] < alphaCmin)  //y < x, y < z
			alphaCmin = alpha[1];
	}

	// Initialize the current voxel index.
	float cIndexNumber[3] = { firstIntersectionIndexDown[0] , firstIntersectionIndexDown[1] , firstIntersectionIndexDown[2] };

	//The while loop represents when the ray is inside the volume
	while (alphaCmin < alphaMax)
	{
		// Store the current ray position 
		alphaCminPrev = alphaCmin;
	
		if ((alpha[0] <= alpha[1]) && (alpha[0] <= alpha[2])) //Ray front intercepts with x-plane. Update alphaX
		{
			alphaCmin = alpha[0];
			cIndexNumber[2] = cIndexNumber[2] + U[0];
			alpha[0] = alpha[0] + alphaU[0];
		}
		else if ((alpha[1] <= alpha[0]) && (alpha[1] <= alpha[2])) //Ray front intercepts with y-plane. Update alphaY
		{ 
			alphaCmin = alpha[1];
			cIndexNumber[1] = cIndexNumber[1] + U[1];
			alpha[1] = alpha[1] + alphaU[1];
		}
		else                                                                //Ray front intercepts with z-plane. Update alphaZ
		{
			alphaCmin = alpha[2];
			cIndexNumber[0] = cIndexNumber[0] + U[2];
			alpha[2] = alpha[2] + alphaU[2];
		}
	
	
		if ((cIndexNumber[0] >= 0) && (cIndexNumber[0] < (d_sizeCT[0])) &&
			(cIndexNumber[1] >= 0) && (cIndexNumber[1] < (d_sizeCT[1])) &&
			(cIndexNumber[2] >= 0) && (cIndexNumber[2] < (d_sizeCT[2])))
		{
			//If it is a valid index, get the voxel intensity
		
			int cIndexCoordinate[3] = { static_cast<int> (cIndexNumber[2]) ,static_cast<int> (cIndexNumber[1]) ,static_cast<int> (cIndexNumber[0]) };

			//Get current position from flat object
			long int currentPos3D  = cIndexCoordinate[0] + (cIndexCoordinate[1] *(d_sizeCT[2])) + (cIndexCoordinate[2] * (d_sizeCT[2])*(d_sizeCT[1]));
	
			//Retrieve intensity value from flat object
			float value = tex1Dfetch<float>(tex_object3D,currentPos3D);
	
			//Ignore voxels whose intensities are below the desired threshold
			if (value > 0)
				d12 += value * (alphaCmin - alphaCminPrev) ;//weighted sum				
		}
	} //end of the while-loop

	float pixval = d12;
	if (pixval < 0)
		pixval = 0;	

	if (pixval >255)
		pixval = 255;

	//Assign the calculated value for the pixel to its corresponding position in the output array
	object2D[idx] = pixval; 

}

void calculateDRRwithCUDA(Image3D cpp_object3D, Image2D cpp_object2D, CUDAParamerters CUDA_Parameters, DRRParameters DRR_Parameters)
{
		
	cudaMemcpyToSymbol(d_DRR_Parameters, DRR_Parameters.stepInX, 12 * sizeof(float), 0, cudaMemcpyHostToDevice);

	//Block 6
	int num_Threads = CUDA_Parameters.numThreads;
	int num_Blocks = CUDA_Parameters.numBlocks;
	
	//------------------------------------------------------------
	//Launching the threads
	drrCUDA <<< num_Blocks, num_Threads >>> (d_object2D, tex_object3D, tex_mask2D);
	//------------------------------------------------------------

	//Copying the result from the calculations from device to host
	long int vectorSize = cpp_object2D.size[0] * cpp_object2D.size[1];
	float *h_object2D = cpp_object2D.image;
	cudaMemcpy(h_object2D, d_object2D, vectorSize * sizeof(float), cudaMemcpyDeviceToHost);

	return;
}


void HandleCudaKernelError(const cudaError_t CudaError, const char* pName /*= ""*/)
{
	if (CudaError == cudaSuccess)
		return;

	std::cerr << "The '" << pName << " kernel caused the following CUDA runtime error: " << cudaGetErrorString(CudaError) << std::endl;
}
