/**********
Copyright (c) 2018, Xilinx, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**********/

/*****
* Changes from Lab 3:

To achieve better performance on the matrix multiplication the input and output
arrays are partitioned and the innermost loop is unrolled in the kernel.
*****/

//OpenCL utility layer include

#include "stdlib.h"
#include "limits.h"
#include <vector>
#include <fstream>
#include "matrixMult.h"
#include "xcl2.hpp"
//Max Array Size
#define MAX_SIZE 8192

//Array Size to access
#define DATA_SIZE 32


uint64_t get_duration_ns (const cl::Event &event) {
    uint64_t nstimestart, nstimeend;
    event.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_START,&nstimestart);
    event.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_END,&nstimeend);
    return(nstimeend-nstimestart);
}

//Matrix multiply, out = in1 x in2
void MatMul(float *in1, float *in2, float *out, int outRow, int outCol, int midSize){
  for(int i = 0; i < outRow; i++) {
    for(int j = 0; j < outCol; j++) {
        for(int k = 0; k < midSize; k++) {
	    out[i * outCol + j] = out[i * outCol + j] + in1[i * midSize + k] * in2[k * outCol + j];
        }
    }
  }
}

//Functionality to setup OpenCL context and trigger the Kernel
uint64_t MM_fpga (
    std::vector<float,aligned_allocator<float>>& source_in1,   //Input Matrix 1
    std::vector<float,aligned_allocator<float>>& source_in2,   //Input Matrix 2
    std::vector<float,aligned_allocator<float>>& source_fpga_results,    //Output Matrix
    int size                                         //One dimension of matrix
)
{
    size_t matrix_size_bytes = sizeof(float) * size * size;

    cl::Event event;//, event1, event2;
    uint64_t kernel_duration = 0;

    //The get_xil_devices will return vector of Xilinx Devices
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];

    //Creating Context and Command Queue for selected Device
    cl::Context context(device);
    cl::CommandQueue q1(context, device, CL_QUEUE_PROFILING_ENABLE);

    std::string device_name = device.getInfo<CL_DEVICE_NAME>();

    //import_binary() command will find the OpenCL binary file created using the
    //xocc compiler load into OpenCL Binary and return as Binaries
    //OpenCL and it can contain many functions which can be executed on the
    //device.
    std::string binaryFile = xcl::find_binary_file(device_name,"matrixMult");
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
    devices.resize(1);
    cl::Program program(context, devices, bins);

    //This call will extract a kernel out of the program we loaded in the
    //previous line. A kernel is an OpenCL function that is executed on the
    //FPGA. This function is defined in the src/mmult.cl file.
    // cl::Kernel kernel1(program,"matrixMult");
    auto mm = cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, int, int>(program, "matrixMult");
 	//const unsigned num_block_rows = C_height / BLOCK_SIZE;

	//These commands will allocate memory on the FPGA. The cl::Buffer
	//objects can be used to reference the memory locations on the device.
	//The cl::Buffer object cannot be referenced directly and must be passed
	//to other OpenCL functions.
	cl::Buffer buffer_in1(context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, 			    	matrix_size_bytes,source_in1.data());
	cl::Buffer buffer_in2(context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, 			    	matrix_size_bytes,source_in2.data());
	cl::Buffer buffer_output(context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, matrix_size_bytes,source_fpga_results.data());

    	//These commands will load the source_in1 and source_in2 vectors from the host
    	//application into the buffer_in1 and buffer_in2 cl::Buffer objects. The data
    	//will be be transferred from system memory over PCIe to the FPGA on-board
    	//DDR memory.
  //
  //   	q1.enqueueMigrateMemObjects({buffer_in1},0/* 0 means from host*/);
  //   	q1.enqueueMigrateMemObjects({buffer_in2},0/* 0 means from host*/);
	// q1.enqueueMigrateMemObjects({buffer_output},0/* 0 means from host*/);
  //
  //
  //   		    int narg = 0;
	// 	    kernel1.setArg(narg++, buffer_output);
	// 	    kernel1.setArg(narg++, buffer_in1);
	// 	    kernel1.setArg(narg++, buffer_in2);
  //
	//     	    kernel1.setArg(narg++, DATA_SIZE);
	// 	    kernel1.setArg(narg++, DATA_SIZE);
		    //Launch the kernel
    		    //q1.enqueueTask(kernel1, NULL, &event);
    		    const size_t global_work_size[2] = {DATA_SIZE, DATA_SIZE};
    		    const size_t local_work_size[2]  = {BLOCK_SIZE, BLOCK_SIZE};

		    cl::NDRange Range1(DATA_SIZE, DATA_SIZE);
		    cl::NDRange Range2(BLOCK_SIZE, BLOCK_SIZE);
        printf("Launching for device %d (global size: %zd, %zd)\n", 0, global_work_size[0], global_work_size[1]);
        printf("Launching for device %d (local size: %zd, %zd)\n", 0, local_work_size[0], local_work_size[1]);
    		    // q1.enqueueNDRangeKernel(kernel1, cl::NullRange, Range1, Range2, NULL, &event);
            mm(cl::EnqueueArgs(q1, cl::NDRange(DATA_SIZE, DATA_SIZE), cl::NDRange(BLOCK_SIZE, BLOCK_SIZE)), buffer_output, buffer_in1, buffer_in2, DATA_SIZE, DATA_SIZE);
            cl::copy(q1, buffer_output, source_fpga_results.begin(), source_fpga_results.end());
		    //wait();
		    kernel_duration += get_duration_ns(event);
      //
    	// q1.enqueueMigrateMemObjects({buffer_in1},CL_MIGRATE_MEM_OBJECT_HOST);
    	// q1.enqueueMigrateMemObjects({buffer_in2},CL_MIGRATE_MEM_OBJECT_HOST);
    	// q1.enqueueMigrateMemObjects({buffer_output},CL_MIGRATE_MEM_OBJECT_HOST);
    	q1.finish();


    return kernel_duration;
}

int main(int argc, char** argv)
{
    if (DATA_SIZE > MAX_SIZE) {
        std::cout << "Size is bigger than internal buffer size,"
        << " please use a size smaller than " << MAX_SIZE << "!" << std::endl;
        return EXIT_FAILURE;
    }

    //Allocate Memory in Host Memory
    int size = DATA_SIZE;
    size_t matrix_size_bytes = sizeof(float) * size * size;
    int A_height=size, A_width=size, B_height=size, B_width=size, C_height=size, C_width=size;

     printf("Matrix sizes:\n  A: %d x %d\n  B: %d x %d\n  C: %d x %d\n",
      A_height, A_width, B_height, B_width, C_height, C_width);

  // Spot check matrix sizes. They all must be a multiple of BLOCK_SIZE,
  // although it is relatively straightforward to handle non-multiples
  // by adding padding. For simplicity, this example does not pad.
  if((A_height % BLOCK_SIZE) != 0 || (A_width % BLOCK_SIZE) != 0 ||
     (B_height % BLOCK_SIZE) != 0 || (B_width % BLOCK_SIZE) != 0 ||
     (C_height % BLOCK_SIZE) != 0 || (C_width % BLOCK_SIZE) != 0) {
    printf("Matrix sizes must be a multiple of %d.\n", BLOCK_SIZE);
    return -1;
  }

    //When creating a buffer with user pointer, under the hood user ptr is
    //used if and only if it is properly aligned (page aligned). When not
    //aligned, runtime has no choice but to create its own host side buffer
    //that backs user ptr. This in turn implies that all operations that move
    //data to/from device incur an extra memcpy to move data to/from runtime's
    //own host buffer from/to user pointer. So it is recommended to use this
    //allocator if user wish to Create Buffer/Memory Object to align user buffer
    //to the page boundary. It will ensure that user buffer will be used when
    //user create Buffer/Mem Object.
    std::vector<float,aligned_allocator<float>> source_in1(matrix_size_bytes);
    std::vector<float,aligned_allocator<float>> source_in2(matrix_size_bytes);
    std::vector<float,aligned_allocator<float>> source_fpga_results(matrix_size_bytes/sizeof(float));
    std::vector<float,aligned_allocator<float>> source_cpu_results(matrix_size_bytes);

    //Create the test data and Software Result
    for(int i = 0 ; i < DATA_SIZE * DATA_SIZE ; i++){
        source_in1[i] = rand() % size;
        source_in2[i] = rand() % size;
        source_cpu_results[i] = 0;
        source_fpga_results[i] = 0;
    }

        // Display the numbers read:
/*        std::cout << "The numbers are: ";
        for (int ct = 0; ct < size*size; ct++){
	    if(ct % size == 0)    std::cout << std::endl;
            std::cout << source_in1[ct] << " ";
        }
        std::cout << std::endl;
*/
    uint64_t kernel_duration = 0;

    //FW_cpu(source_in1.data(), source_cpu_results.data(), size);
    std::cout << "Computing MM on CPU...\n";
    MatMul(source_in1.data(), source_in2.data(), source_cpu_results.data(), size, size, size);

    std::cout << "Finished. \n";

    std::cout << "Computing MM on FPGA...\n";
    //Test();
    //Compute FPGA Results
    kernel_duration = MM_fpga(source_in1, source_fpga_results, source_cpu_results, size);

        // Display the numbers produced:
/*        std::cout << "The CPU results are: ";
        for (int ct = 0; ct < size*size; ct++){
	    if(ct % size == 0)    std::cout << std::endl;
            std::cout <<  source_cpu_results[ct] << " ";
        }

        std::cout << std::endl;


       std::cout << "The FPGA results are: ";
        for (int ct = 0; ct < size*size; ct++){
	    if(ct % size == 0)    std::cout << std::endl;
            std::cout <<  source_in1[ct] << " ";
        }

        std::cout << std::endl;
*/
    std::cout << "Finished. \n";
    //Compare the results of the FPGA to CPU
    bool match = true;
    for (int i = 0 ; i < size * size; i++){
        if (source_cpu_results[i] != source_in1[i]){
            std::cout << "Error: Result mismatch" << std::endl;
            std::cout << "i = " << i << " CPU result = " << source_cpu_results[i]
                << " FPGA result = " << source_in1[i] << std::endl;
            match = false;
            break;
        }
    }

    std::cout << "TEST " << (match ? "PASSED" : "FAILED") << std::endl;

    std::cout << "Wall Clock Time (Kernel execution): " << kernel_duration << std::endl;
    std::cout << "Note: Wall Clock Time is meaningful for real hardware execution only,"
            << "not for emulation." << std::endl;

    return (match ? EXIT_SUCCESS :  EXIT_FAILURE);
}
