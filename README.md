# Intel_MatMul

## Description
This scope of this project is to integrate and run the OpenCL Matrix Multiplication API offered by Intel, with Xilinx SDAccel products. The project is derived from the following APIs:

	1. [Intel Matrix Multiplication example](https://www.intel.com/content/www/us/en/programmable/support/support-resources/design-examples/design-software/opencl/matrix-multiplication.html)

	2. [Xilinx SDAccel_Examples](https://github.com/Xilinx/SDAccel_Examples/tree/2017.4)

## Overview of Files
	Intel_MM/src/main.cpp - Original Intel host file
	Intel_MM/src/matrixMult.h - Intel header file
	Intel_MM/src/mmult.cl - Original Intel OpenCL kernel file
	Intel_MM/src/host.cpp - Host file derived from SDAccel_Examples
	Intel_MM/Makefile - Makefile file derived from SDAccel_Examples
	libs - opencl and xcl2 libraries from SDAccel_Examples
	
