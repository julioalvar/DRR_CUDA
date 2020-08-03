
#pragma once
#include <vector>

extern float* d_object3D;

// This variable contains the 2D output from CUDA
extern float *d_object2D;

struct Image3D
{
	float *image;
	float PixelSpacingCT[3];
	float isoCenter[3];
	int SizeCT[3];
};

struct Image2D
{
	float *image;
	int size[2]; // [rows, cols]
	//int rows, cols;
};

struct CUDAParamerters
{
	int numThreads;
	int numBlocks;
};

struct DRRParameters
{
	float stepInX[3];
	float stepInY[3];
	float corner00[3];
	float SourceWorld[3];
	int ROI[5]; // first 4 positions are the roi, the last one is the flag
	//bool useROI;
	
};

void loadDICOMInGPUMemory(float *cpp_object3D, int *sizeCT, float *pixelSpacing);
void loadOuputVariablesInGPUMemory(int dimX, int dimZ);
void loadMaskInGPUMemory(unsigned char* mask_2D, int dimX, int dimZ, bool useMask);
void updateMaskFlagInGPUMemory(bool useMask);
void freeDICOMFromGPUMemory();
void freeAuxiliaryVariablesInGPUMemory();

void calculateDRRwithCUDA(Image3D cpp_object3D, Image2D cpp_object2D, CUDAParamerters CUDA_Parameters, DRRParameters DRR_Parameters);

