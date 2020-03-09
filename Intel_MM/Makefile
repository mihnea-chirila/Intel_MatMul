#+-------------------------------------------------------------------------------
# The following parameters are assigned with default values. These parameters can
# be overridden through the make command line
#+-------------------------------------------------------------------------------

# Run Target:
#   hw  - Compile for hardware
#   sw_emu/hw_emu - Compile for software/hardware emulation
# FPGA Board Platform (Default ~ ku115)

include utils.mk
REPORT := yes
PROFILE := yes
DEBUG := no

TARGETS := sw_emu
TARGET := $(TARGETS)
DEVICES := xilinx_kcu1500_dynamic_5_0
DEVICE := $(DEVICES)
XCLBIN := ./xclbin
DSA := $(call device2sandsa, $(DEVICE))

CXX := $(XILINX_SDX)/bin/xcpp
XOCC := $(XILINX_SDX)/bin/xocc

# Points to Utility Directory
COMMON_REPO = ../
ABS_COMMON_REPO = $(shell readlink -f $(COMMON_REPO))

CXXFLAGS := $(opencl_CXXFLAGS) -Wall -O0 -g -std=c++14
LDFLAGS := $(opencl_LDFLAGS)

HOST_SRCS = src/host.cpp

# Host compiler global settings
CXXFLAGS = -I $(XILINX_SDX)/runtime/include/1_2/ -I/$(XILINX_SDX)/Vivado_HLS/include/ -O0 -g -Wall -fmessage-length=0 -std=c++14
LDFLAGS = -lOpenCL -lpthread -lrt -lstdc++ -L$(XILINX_SDX)/runtime/lib/x86_64

# Kernel compiler global settings
CLFLAGS = -t $(TARGET) --platform $(DEVICE) --save-temps -g --kernel_frequency 60 #--optimize 3
CLFLAGS += --xp "param:compiler.preserveHlsOutput=1" --xp "param:compiler.generateExtraRunData=true"


#'estimate' for estimate report generation
#'system' for system report generation
ifneq ($(REPORT), no)
CLFLAGS += --report estimate
LDCLFLAGS += --profile_kernel data:all:all:all --profile_kernel stall:all:all:all --profile_kernel exec:all:all:all #--report system
endif

#Generates profile summary report
ifeq ($(PROFILE), yes)
CLFLAGS += --profile_kernel stall:all:all:all #--profile_kernel data:all:all:all
endif

#Generates debug summary report
ifeq ($(DEBUG), yes)
CLFLAGS += --dk protocol:all:all:all
endif

EXECUTABLE = host

BINARY_CONTAINERS += $(XCLBIN)/matrixMult.$(TARGET).$(DSA).xclbin #$(XCLBIN)/mmult.$(TARGET).$(DSA).xclbin $(XCLBIN)/FW.$(TARGET).$(DSA).xclbin 
BINARY_CONTAINER_1_OBJS += $(XCLBIN)/matrixMult.$(TARGET).$(DSA).xo #$(XCLBIN)/mmult.$(TARGET).$(DSA).xo $(XCLBIN)/FW.$(TARGET).$(DSA).xo 
#BINARY_CONTAINER_1_OBJS += $(XCLBIN)/rKleene_mmultA.$(TARGET).$(DSA).xo $(XCLBIN)/rKleene_mmultB.$(TARGET).$(DSA).xo $(XCLBIN)/mmult.$(TARGET).$(DSA).xo
#ALL_KERNEL_OBJS += $(XCLBIN)/mmult.$(TARGET).$(DSA).xo

#Include Libraries
include $(ABS_COMMON_REPO)/libs/opencl/opencl.mk
include $(ABS_COMMON_REPO)/libs/xcl2/xcl2.mk
CXXFLAGS += $(xcl2_CXXFLAGS)
LDFLAGS += $(xcl2_LDFLAGS)
HOST_SRCS += $(xcl2_SRCS)

CP = cp -rf

.PHONY: all clean cleanall docs
all: $(EXECUTABLE) $(BINARY_CONTAINERS)

.PHONY: exe
exe: $(EXECUTABLE)

# Building kernel
#$(XCLBIN)/rKleene_mmultA.$(TARGET).$(DSA).xo: ./src/mmult.cl
#	mkdir -p $(XCLBIN)
#	$(XOCC) $(CLFLAGS) -c -k rKleene_mmultA -I'$(<D)' -o'$@' '$<'
#$(XCLBIN)/rKleene_mmultB.$(TARGET).$(DSA).xo: ./src/mmult.cl
#	mkdir -p $(XCLBIN)
#	$(XOCC) $(CLFLAGS) -c -k rKleene_mmultB -I'$(<D)' -o'$@' '$<'
#$(XCLBIN)/FW.$(TARGET).$(DSA).xo: ./src/mmult.cl
#	mkdir -p $(XCLBIN)
#	$(XOCC) $(CLFLAGS) -c -k FW -I'$(<D)' -o'$@' '$<'
#$(XCLBIN)/mmult.$(TARGET).$(DSA).xo: ./src/mmult.cl
#	mkdir -p $(XCLBIN)
#	$(XOCC) $(CLFLAGS) -c -k mmult -I'$(<D)' -o'$@' '$<'
$(XCLBIN)/matrixMult.$(TARGET).$(DSA).xo: ./src/mmult.cl
	mkdir -p $(XCLBIN)
	$(XOCC) $(CLFLAGS) -c -k matrixMult -I'$(<D)' -o'$@' '$<'

#$(XCLBIN)/rKleene_mmultA.$(TARGET).$(DSA).xclbin: $(BINARY_CONTAINER_1_OBJS)
#	$(XOCC) $(CLFLAGS) -l $(LDCLFLAGS) --nk rKleene_mmultA:1 -o'$@' $(+)
#$(XCLBIN)/rKleene_mmultB.$(TARGET).$(DSA).xclbin: $(BINARY_CONTAINER_1_OBJS)
#	$(XOCC) $(CLFLAGS) -l $(LDCLFLAGS) --nk rKleene_mmultB:1 -o'$@' $(+)
#$(XCLBIN)/FW.$(TARGET).$(DSA).xclbin: $(BINARY_CONTAINER_1_OBJS)
#	$(XOCC) $(CLFLAGS) -l $(LDCLFLAGS) --nk FW:1 -o'$@' $(+)
#$(XCLBIN)/mmult.$(TARGET).$(DSA).xclbin: $(BINARY_CONTAINER_1_OBJS)
#	$(XOCC) $(CLFLAGS) -l $(LDCLFLAGS) --nk mmult:1 -o'$@' $(+)
$(XCLBIN)/matrixMult.$(TARGET).$(DSA).xclbin: $(BINARY_CONTAINER_1_OBJS)
	$(XOCC) $(CLFLAGS) -l $(LDCLFLAGS) --nk matrixMult:1 -o'$@' $(+)

# Building Host
$(EXECUTABLE): $(HOST_SRCS)
	mkdir -p $(XCLBIN)
	$(CXX) $(CXXFLAGS) $(HOST_SRCS) -o '$@' $(LDFLAGS)

check: all
ifeq ($(TARGET),$(filter $(TARGET),sw_emu hw_emu))
	emconfigutil --platform $(DEVICE) --od .
	XCL_EMULATION_MODE=$(TARGET) ./$(EXECUTABLE)
	sdx_analyze profile -i sdaccel_profile_summary.csv -f html
endif

# Cleaning stuff
RM = rm -f
RMDIR = rm -rf
clean:
	-$(RMDIR) $(EXECUTABLE) $(XCLBIN)/{*sw_emu*,*hw_emu*} 
	-$(RMDIR) sdaccel_* TempConfig system_estimate.xtxt *.rpt
	-$(RMDIR) src/*.ll _xocc_* .Xil emconfig.json dltmp* xmltmp* *.log *.jou *.wcfg *.wdb

cleanall: clean
	-$(RMDIR) $(XCLBIN)
	-$(RMDIR) ./_x

ECHO:= @echo

.PHONY: help

help::
	$(ECHO) "Makefile Usage:"
	$(ECHO) "  make all TARGET=<sw_emu/hw_emu/hw> DEVICE=<FPGA platform>"
	$(ECHO) "      Command to generate the design for specified Target and Device."
	$(ECHO) ""
	$(ECHO) "  make clean "
	$(ECHO) "      Command to remove the generated non-hardware files."
	$(ECHO) ""
	$(ECHO) "  make cleanall"
	$(ECHO) "      Command to remove all the generated files."
	$(ECHO) ""
	$(ECHO) "  make check TARGET=<sw_emu/hw_emu/hw> DEVICE=<FPGA platform>"
	$(ECHO) "      Command to run application in emulation."
	$(ECHO) ""

docs: README.md

README.md: description.json
	$(ABS_COMMON_REPO)/utility/readme_gen/readme_gen.py description.json

