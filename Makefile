#include ../CommonMake.mak

SRC = sc_ocl.cpp 

INC = -I$(INC_DIR)

OPENCV_FLAG = `pkg-config --cflags --libs opencv`

CFLAGS = -g -Wfatal-errors -Wl,--allow-multiple-definition -Wall -Wno-unknown-pragmas -fPIC --shared  $(INC) $(SHARE_LIB) -L$(LIB_DIR)

OCL_FLAGS = -I/usr/local/cuda/include -lOpenCL
CPP = g++

LOCAL_TARGET = sc_ocl

.PHONY: all clean 

all: $(LIB_DIR) $(LOCAL_TARGET)

sc_ocl: sc_ocl.cpp
	@g++ -L/usr/local/lib/ sc_ocl.cpp -g -I/usr/local/cuda-5.5/include -lOpenCL -o sc_ocl -lopencv_core -lopencv_highgui -lopencv_imgproc

$(LIB_DIR):
	mkdir -p $(LIB_DIR)

clean:
	rm -f $(LOCAL_TARGET)
