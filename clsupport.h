/**
 *  Histogram of oriented gradient on GPU 
 *  by Jakub Vojvoda, vojvoda@swdeveloper.sk
 *  2015  
 *
 *  licence: GNU LGPL v3
 *  file: clsupport.h
 */
 
#ifndef __clsupport_h__
#define __clsupport_h__

#include <CL/cl.hpp>

#define SELECTED_DEVICE_TYPE CL_DEVICE_TYPE_CPU

cl::Device SelectDevice(int device);
char* LoadProgSource(const char* cFilename);

const char *CLErrorString(cl_int _err);
void CheckOpenCLError(cl_int _ciErr, const char *_sMsg, ...);

double GetTime(void);
double getEventTime(cl::Event i_event);

unsigned int iCeilTo(unsigned int data, unsigned int align_size);


#endif