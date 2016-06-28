/**
 *  Histogram of oriented gradient on GPU 
 *  by Jakub Vojvoda, vojvoda@swdeveloper.sk
 *  2015  
 *
 *  licence: GNU LGPL v3
 *  file: clsupport.cpp
 */ 
 
#include "clsupport.h"

#ifdef _WIN32
	#include <windows.h>
#else
	#include <sys/time.h>
#endif

#include <stdarg.h>


cl::Device SelectDevice(int device)
{
	cl_int err_msg;

	std::vector<cl::Platform> platforms;
	std::vector<cl::Device> platform_devices;
	CheckOpenCLError(cl::Platform::get(&platforms), "cl::Platform::get");

	printf("Platforms:\n");

	for (unsigned int i = 0; i < platforms.size(); i++) {

		printf(" %d. platform name: %s.\n", i, platforms[i].getInfo<CL_PLATFORM_NAME>(&err_msg).c_str());
		CheckOpenCLError(err_msg, "cl::Platform::getInfo<CL_PLATFORM_NAME>");

		CheckOpenCLError(platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &platform_devices), "getDevices");

		for (unsigned int j = 0; j < platform_devices.size(); j++) {
			printf("  %d. device name: %s.\n", j, platform_devices[j].getInfo<CL_DEVICE_NAME>(&err_msg).c_str());
			CheckOpenCLError(err_msg, "cl::Device::getInfo<CL_DEVICE_NAME>");
		}

		platform_devices.clear();
	}

	cl::Device selected_device;
	bool device_found = false;

	for (unsigned int i = 0; i < platforms.size(); i++) {

		CheckOpenCLError(platforms[i].getDevices(device, &platform_devices), "getDevices");

		if (platform_devices.size() != 0) {
			device_found = true;
			selected_device = platform_devices[0];
			break;
		}
	}

	if (!device_found) {
		CheckOpenCLError(CL_DEVICE_NOT_FOUND, "GPU device");
	}

	if (selected_device.getInfo<CL_DEVICE_TYPE>() == device) {
		printf("\nSelected device type: Correct\n");
	}
	else {
		printf("\nSelected device type: Incorrect\n");
	}

	printf("Selected device name: %s.\n", selected_device.getInfo<CL_DEVICE_NAME>().c_str());
	platforms.clear();

	return selected_device;
}

char* LoadProgSource(const char* cFilename)
{
	FILE* pFileStream = NULL;
	size_t szSourceLength;

#ifdef _WIN32 
	if (fopen_s(&pFileStream, cFilename, "rb") != 0) {
		return NULL;
	}
#else 
	pFileStream = fopen(cFilename, "rb");

	if (pFileStream == 0) {
		return NULL;
	}
#endif

	fseek(pFileStream, 0, SEEK_END);
	szSourceLength = ftell(pFileStream);
	fseek(pFileStream, 0, SEEK_SET);

	char* cSourceString = (char *)malloc(szSourceLength + 1);

	if (fread(cSourceString, szSourceLength, 1, pFileStream) != 1) {
		fclose(pFileStream);
		free(cSourceString);
		return 0;
	}

	fclose(pFileStream);
	cSourceString[szSourceLength] = '\0';

	return cSourceString;
}

const char *CLErrorString(cl_int _err) {
	switch (_err) {
	case CL_SUCCESS:
		return "Success!";
	case CL_DEVICE_NOT_FOUND:
		return "Device not found.";
	case CL_DEVICE_NOT_AVAILABLE:
		return "Device not available";
	case CL_COMPILER_NOT_AVAILABLE:
		return "Compiler not available";
	case CL_MEM_OBJECT_ALLOCATION_FAILURE:
		return "Memory object allocation failure";
	case CL_OUT_OF_RESOURCES:
		return "Out of resources";
	case CL_OUT_OF_HOST_MEMORY:
		return "Out of host memory";
	case CL_PROFILING_INFO_NOT_AVAILABLE:
		return "Profiling information not available";
	case CL_MEM_COPY_OVERLAP:
		return "Memory copy overlap";
	case CL_IMAGE_FORMAT_MISMATCH:
		return "Image format mismatch";
	case CL_IMAGE_FORMAT_NOT_SUPPORTED:
		return "Image format not supported";
	case CL_BUILD_PROGRAM_FAILURE:
		return "Program build failure";
	case CL_MAP_FAILURE:
		return "Map failure";
	case CL_INVALID_VALUE:
		return "Invalid value";
	case CL_INVALID_DEVICE_TYPE:
		return "Invalid device type";
	case CL_INVALID_PLATFORM:
		return "Invalid platform";
	case CL_INVALID_DEVICE:
		return "Invalid device";
	case CL_INVALID_CONTEXT:
		return "Invalid context";
	case CL_INVALID_QUEUE_PROPERTIES:
		return "Invalid queue properties";
	case CL_INVALID_COMMAND_QUEUE:
		return "Invalid command queue";
	case CL_INVALID_HOST_PTR:
		return "Invalid host pointer";
	case CL_INVALID_MEM_OBJECT:
		return "Invalid memory object";
	case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
		return "Invalid image format descriptor";
	case CL_INVALID_IMAGE_SIZE:
		return "Invalid image size";
	case CL_INVALID_SAMPLER:
		return "Invalid sampler";
	case CL_INVALID_BINARY:
		return "Invalid binary";
	case CL_INVALID_BUILD_OPTIONS:
		return "Invalid build options";
	case CL_INVALID_PROGRAM:
		return "Invalid program";
	case CL_INVALID_PROGRAM_EXECUTABLE:
		return "Invalid program executable";
	case CL_INVALID_KERNEL_NAME:
		return "Invalid kernel name";
	case CL_INVALID_KERNEL_DEFINITION:
		return "Invalid kernel definition";
	case CL_INVALID_KERNEL:
		return "Invalid kernel";
	case CL_INVALID_ARG_INDEX:
		return "Invalid argument index";
	case CL_INVALID_ARG_VALUE:
		return "Invalid argument value";
	case CL_INVALID_ARG_SIZE:
		return "Invalid argument size";
	case CL_INVALID_KERNEL_ARGS:
		return "Invalid kernel arguments";
	case CL_INVALID_WORK_DIMENSION:
		return "Invalid work dimension";
	case CL_INVALID_WORK_GROUP_SIZE:
		return "Invalid work group size";
	case CL_INVALID_WORK_ITEM_SIZE:
		return "Invalid work item size";
	case CL_INVALID_GLOBAL_OFFSET:
		return "Invalid global offset";
	case CL_INVALID_EVENT_WAIT_LIST:
		return "Invalid event wait list";
	case CL_INVALID_EVENT:
		return "Invalid event";
	case CL_INVALID_OPERATION:
		return "Invalid operation";
	case CL_INVALID_GL_OBJECT:
		return "Invalid OpenGL object";
	case CL_INVALID_BUFFER_SIZE:
		return "Invalid buffer size";
	case CL_INVALID_MIP_LEVEL:
		return "Invalid mip-map level";
	default:
		return "Unknown";
	}
}

void CheckOpenCLError(cl_int _ciErr, const char *_sMsg, ...)
{
	unsigned int uiDebug = 1;
	char buffer[1024];

	va_list arg;
	va_start(arg, _sMsg);
	vsprintf(buffer, _sMsg, arg);
	va_end(arg);

	if (_ciErr != CL_SUCCESS && _ciErr != CL_DEVICE_NOT_FOUND) {
		printf("%f: ERROR: %s: (%i)%s\n", GetTime(), buffer, _ciErr, CLErrorString(_ciErr));
		system("PAUSE");
		exit(1);
	}
	else if (uiDebug>1) {
		printf("%f:    OK: %s\n", GetTime(), buffer);
	}
}

double GetTime(void)
{
#if _WIN32  															
	static int initialized = 0;
	static LARGE_INTEGER frequency;
	LARGE_INTEGER value;

	if (!initialized) {
		initialized = 1;
		if (QueryPerformanceFrequency(&frequency) == 0) {
			exit(-1);
		}
	}

	QueryPerformanceCounter(&value);
	return (double)value.QuadPart / (double)frequency.QuadPart;

#else                                         							
	struct timeval tv;
	if (gettimeofday(&tv, NULL) == -1) {
		exit(-2);
	}
	return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.;
#endif
}

double getEventTime(cl::Event i_event)
{
	return double(i_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - i_event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) / 1000000;
}

unsigned int iCeilTo(unsigned int data, unsigned int align_size)
{
	return ((data + align_size - 1) / align_size) * align_size;
}
