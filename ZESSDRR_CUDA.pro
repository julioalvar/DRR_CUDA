# -----------------------------------------------------------------------------
# ZESSDRR_CUDA project profile
#
# \file
# \author  Alvarez
# \date    2020-03-17
# -----------------------------------------------------------------------------


TEMPLATE   = lib
TARGET     = ZESSDRR_CUDA

DESTDIR    = $$(MLAB_CURRENT_PACKAGE_DIR)/lib
DLLDESTDIR = $$(MLAB_CURRENT_PACKAGE_DIR)/lib

# Set high warn level (warn 4 on MSVC)
WARN = HIGH

# Add used projects here (see included pri files below for available projects)
CONFIG += dll ML CUDA OpenCV OpenCV_photo MLTools MLDicomTree MLDicomTreeImagePropertyExtension

MLAB_PACKAGES += ZESS_DRR \
				 CUDAPacket_General \
                 MeVisLab_Standard

# make sure that this file is included after CONFIG and MLAB_PACKAGES
include ($(MLAB_MeVis_Foundation)/Configuration/IncludePackages.pri)

DEFINES += ZESSDRR_CUDA_EXPORTS

# Enable ML deprecated API warnings. To completely disable the deprecated API, change WARN to DISABLE.
DEFINES += ML_WARN_DEPRECATED

CUDA_SOURCES += CUDAdrr.cu

HEADERS += \
    ZESSDRR_CUDAInit.h \
    ZESSDRR_CUDASystem.h \
    mlDRR_CUDA.h \
	CUDAdrr.cuh 

SOURCES += \
    ZESSDRR_CUDAInit.cpp \
    mlDRR_CUDA.cpp \
	InOutImageProcessing.cpp \
	DRR_functions.cpp \
	ReadTagsDicom.cpp \
	