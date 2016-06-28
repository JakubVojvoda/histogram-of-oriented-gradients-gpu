/**
 *  Histogram of oriented gradient on GPU 
 *  by Jakub Vojvoda, vojvoda@swdeveloper.sk
 *  2015  
 *
 *  licence: GNU LGPL v3
 *  file: main.cpp
 */ 

#pragma comment( lib, "OpenCL" )

#include <iostream>

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>

#include <CL/cl.hpp>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "clsupport.h"
#include "model.h"

#ifdef _WIN32
	#include <windows.h>
#else
	#include <sys/time.h>
#endif

#ifndef M_PI
	#define M_PI 3.14159265f
#endif

#define ARG_DEVICE   1
#define ARG_IN_IMAGE 2
#define ARG_IN_VIDEO 3
#define ARG_VISUAL   4 
#define ARG_LOAD     5
#define ARG_TRAIN	   6
#define ARG_UNKNOWN  7


typedef struct input_paths {

	bool isdef_video;
	std::string input_path;

	bool isdef_load;
	std::string model_path;

	bool isdef_train;
	std::string pos_path;
	std::string neg_path;

	input_paths() : 
		isdef_video(false), 
		isdef_load(false),
		isdef_train(false),
		input_path(""),
		model_path(""), 
		pos_path(""), 
		neg_path("") {}

} input_paths_t;

static void usage(std::string name);
int parseArgsTypes(std::string arg);
bool parseArgs(int argc, char **argv, int &device, input_paths_t &in, bool &visual);


int main(int argc, char* argv[])
{
	int device = SELECTED_DEVICE_TYPE;
	input_paths_t in;
	bool is_visual = true;

	
	if (!parseArgs(argc, argv, device, in, is_visual)) {
		usage(argv[0]);
		return 1;
	}
	
	cl_int err_msg;
	cl::Device selected_device = SelectDevice(device);

	cl::Context context(selected_device, NULL, NULL, NULL, &err_msg);
	CheckOpenCLError(err_msg, "cl::Context");

	cl::CommandQueue queue(context, selected_device, CL_QUEUE_PROFILING_ENABLE, &err_msg);
	CheckOpenCLError(err_msg, "cl::CommandQueue");

	char *program_source = LoadProgSource("kernel/hog.cl");
	cl::Program::Sources sources;
	sources.push_back(std::pair<const char *, size_t>(program_source, 0));

	cl::Program program(context, sources, &err_msg);
	CheckOpenCLError(err_msg, "clCreateProgramWithSource");

	if ((err_msg = program.build(std::vector<cl::Device>(1, selected_device), "-g -s \"kernel/hog.cl\"", NULL, NULL)) == CL_BUILD_PROGRAM_FAILURE) {
		printf("Build log:\n %s", program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(selected_device, &err_msg).c_str());
		CheckOpenCLError(err_msg, "cl::Program::getBuildInfo<CL_PROGRAM_BUILD_LOG>");
	}

	CheckOpenCLError(err_msg, "clBuildProgram");

	cl::make_kernel<cl::Buffer&, cl_int&, cl_int&, cl_int&, cl_int&, cl::Buffer&, cl::Buffer&, cl_float, cl::Buffer&>compute_hog_descriptor =
		cl::make_kernel<cl::Buffer&, cl_int&, cl_int&, cl_int&, cl_int&, cl::Buffer&, cl::Buffer&, cl_float, cl::Buffer&>(program, "compute_hog_descriptor", &err_msg);
	
	cl::UserEvent img_event(context, &err_msg);
	cl::UserEvent svm_y_event(context, &err_msg);

	cl::UserEvent read_desc_event(context, &err_msg);
	cl::UserEvent read_prediction_event(context, &err_msg);

	cv::VideoCapture stream;

	if (in.isdef_video && !stream.open(in.input_path)) {
		std::cerr << "Cannot open input stream '" << in.input_path << "'." << std::endl;
		return 1;
	}

	std::vector<float> weights;
	float bias;
	bool file_err = false;

	if (in.isdef_train) {
		file_err = trainSVM(in.pos_path, in.neg_path, weights, bias);
	}
	else if (in.isdef_load) {
		file_err = loadSVM(in.model_path, weights, bias);
	}

	if (!is_visual && !file_err) {
		std::cerr << "Cannot train or load model from file." << std::endl;
		return 1;
	}

	cv::Mat frame, image, tmp_image;

	int window_width = iCeilTo(WINDOW_WIDTH, BLOCK_WIDTH);
	int window_height = iCeilTo(WINDOW_HEIGHT, BLOCK_HEIGHT);

	float scaling_step = SCALING_STEP;
	double etime = 0.0, ctime = 0.0;

	while (1) {

		if (in.isdef_video) {
			stream >> tmp_image;

			if (tmp_image.empty()) {
				break;
			}
		}
		else {
			tmp_image = cv::imread(in.input_path);

			if (tmp_image.empty()) {
				std::cerr << "Input frame is empty." << std::endl;
				return 1;
			}
		}

		for (float scale = 1.0; scale > MIN_SCALE; scale -= scaling_step) {

			cv::resize(tmp_image, frame, cv::Size(), scale, scale);

			int w = iCeilTo(frame.size().width, 16);
			int h = iCeilTo(frame.size().height, 16);

			if (w > frame.size().width)  { w -= 16; }
			if (h > frame.size().height) { h -= 16; }

			image = frame(cv::Rect(0, 0, w, h));

			int image_width = image.size().width;
			int image_height = image.size().height;

			if (is_visual) {
				window_width = image_width;
				window_height = image_height;
			}

			cl::NDRange local(16, 16);
			cl::NDRange global(iCeilTo(image_width, local[0]), iCeilTo(image_height, local[1]));
			cl::EnqueueArgs args(queue, global, local);

			std::vector<std::vector<float> > image_descriptor;

			cv::Mat image_gray;
			cv::cvtColor(image, image_gray, CV_BGR2GRAY);

			cl_uchar *img_buff = reinterpret_cast<cl_uchar *>(image_gray.data);

			size_t size = image_width * image_height;

			int total_win_x = (image_width / BLOCK_WIDTH) - (window_width / BLOCK_WIDTH) + 1;
			int total_win_y = (image_height / BLOCK_HEIGHT) - (window_height / BLOCK_HEIGHT) + 1;
			int window_size = (window_width / BLOCK_WIDTH) * (window_height / BLOCK_HEIGHT) * DESCRIPTOR_SIZE;
			int descriptor_size = total_win_x * total_win_y * window_size;

			int total_windows = total_win_x * total_win_y;
			int window_desc_size = (window_width / BLOCK_WIDTH) * (window_height / BLOCK_HEIGHT) * DESCRIPTOR_SIZE;

			cl_float cb = 0;
			cl_float *cy = (cl_float *)malloc(sizeof(cl_float) * window_desc_size);
			memset(cy, 0, sizeof(cl_float) * window_desc_size);

			if (!is_visual) {

				if (weights.size() != window_desc_size) {
					std::cerr << "Wrong model format: different size features." << std::endl;
					free(cy);
					return 1;
				}

				for (unsigned int i = 0; i < weights.size(); i++) {
					cy[i] = weights.at(i);
				}
				
				cb = bias;
			}

			cl::Buffer image_buffer(context, CL_MEM_READ_ONLY, sizeof(cl_uchar) * size, NULL, &err_msg);
			CheckOpenCLError(err_msg, "clCreateBuffer: image_buffer");

			cl::Buffer desc_buffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float) * descriptor_size, NULL, &err_msg);
			CheckOpenCLError(err_msg, "clCreateBuffer: desc_buffer");

			cl::Buffer svm_y(context, CL_MEM_READ_ONLY, sizeof(cl_float) * window_desc_size, NULL, &err_msg);
			CheckOpenCLError(err_msg, "clCreateBuffer: svm_y");

			cl::Buffer svm_predict(context, CL_MEM_WRITE_ONLY, sizeof(cl_float) * total_windows, NULL, &err_msg);
			CheckOpenCLError(err_msg, "clCreateBuffer: svm_predict");

			err_msg = queue.enqueueWriteBuffer(image_buffer, CL_FALSE, 0, sizeof(cl_uchar) * size, img_buff, NULL, &img_event);
			CheckOpenCLError(err_msg, "clEnqueueWriteBuffer: image_buffer");

			err_msg = queue.enqueueWriteBuffer(svm_y, CL_FALSE, 0, sizeof(cl_float) * window_desc_size, cy, NULL, &svm_y_event);
			CheckOpenCLError(err_msg, "clEnqueueWriteBuffer: image_buffer");

			cl_float svm_b = cb;

			cl::Event kernel_compute_hog = compute_hog_descriptor(args,
				image_buffer, image_width, image_height, window_width, window_height, desc_buffer, svm_y, svm_b, svm_predict);

			cl_float *descriptor = (cl_float *)malloc(sizeof(cl_float) * descriptor_size);
			cl_float *prediction = (cl_float *)malloc(sizeof(cl_float) * total_windows);

			err_msg = queue.enqueueReadBuffer(desc_buffer, CL_FALSE, 0, sizeof(cl_float) * descriptor_size, descriptor, NULL, &read_desc_event);
			CheckOpenCLError(err_msg, "enqueueReadBuffer: descriptor");

			err_msg = queue.enqueueReadBuffer(svm_predict, CL_FALSE, 0, sizeof(cl_float) * total_windows, prediction, NULL, &read_prediction_event);
			CheckOpenCLError(err_msg, "enqueueReadBuffer: descriptor");

			CheckOpenCLError(queue.finish(), "clFinish");

			etime += getEventTime(kernel_compute_hog);
			ctime += getEventTime(img_event) + getEventTime(svm_y_event) + getEventTime(read_desc_event) + getEventTime(read_prediction_event);
 
			for (int y = 0; y < total_win_y; y++) {
				for (int x = 0; x < total_win_x; x++) {

					int index = x + y * total_win_x;

					std::vector<float> local_descriptor;

					for (int i = 0; i < window_desc_size; i++) {
						local_descriptor.push_back(descriptor[i + index * window_desc_size]);
					}

					image_descriptor.push_back(local_descriptor);
					local_descriptor.clear();

					if (prediction[index] > PREDICTION_THRESHOLD) {
						float p = 1.0f / scale;

						int px = static_cast<int>(x * BLOCK_WIDTH * p);
						int py = static_cast<int>(y * BLOCK_HEIGHT * p);
						int pw = static_cast<int>(window_width * p);
						int ph = static_cast<int>(window_height * p);

						cv::Rect r = cv::Rect(px, py, pw, ph);
						cv::rectangle(tmp_image, r, cv::Scalar(0, 255, 255));
					}
				}
			}

			free(cy);
			free(descriptor);
			free(prediction);

			if (is_visual) {
				tmp_image = visualization(tmp_image, window_width, window_height, image_descriptor);
				image_descriptor.clear();
				break;
			}

			image_descriptor.clear();
		}

		std::cout << "Copy: " << ctime << std::endl;
		std::cout << "Execution time: " << etime << " ms" << std::endl;

		ctime = 0.0;
		etime = 0.0;

		int delay = (in.isdef_video) ? 1 : 0;
		cv::imshow("Image", tmp_image);

		if (cv::waitKey(delay) > 0) {
			break;
		}
	}

	return 0;
}


bool parseArgs(int argc, char **argv, int &device, input_paths_t &in, bool &visual)
{
	if (argc < 3) {
		return false;
	}

	for (int i = 1; i < argc; i++) {

		std::string arg(argv[i]), tmp;
		int type = parseArgsTypes(arg);

		switch (type) {
		case ARG_DEVICE:
			tmp = std::string(argv[i + 1]);
			device = (tmp.compare("GPU") == 0) ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU;
			i += 1;
			break;
		case ARG_IN_VIDEO:
			in.isdef_video = true;
		case ARG_IN_IMAGE:
			in.input_path = (i + 1 < argc) ? std::string(argv[i + 1]) : "";
			i += 1;
			break;
		case ARG_VISUAL:
			visual = true;
			break;
		case ARG_LOAD:
			visual = false;
			in.isdef_load = true;
			in.model_path = (i + 1 < argc) ? std::string(argv[i + 1]) : "";
			i += 1;
			break;
		case ARG_TRAIN:
			visual = false;
			in.isdef_train = true;
			in.pos_path = (i + 1 < argc) ? std::string(argv[i + 1]) : "";
			in.neg_path = (i + 2 < argc) ? std::string(argv[i + 2]) : "";
			i += 2;
		case ARG_UNKNOWN:
			std::cerr << "Unknown command line argument '" << arg << "'." << std::endl;
		default:
			return false;
		}
	}

	return true;
}

static void usage(std::string name)
{
	std::cout
		<< "usage: " << name << " input [device] [load] [train] [visual]" << std::endl
		<< "  input:  --input 'filepath', or --video 'filepath'" << std::endl
		<< "  device: --device GPU, or --device CPU (implicit)" << std::endl
		<< "  load:   --load 'filepath', path to svm model containing weights and bias" << std::endl
		<< "  train:  --train 'positive' 'negative', filepaths to features" << std::endl
		<< "  visual: --visual, visualization of HOG descriptor" << std::endl
		<< std::endl;
}

int parseArgsTypes(std::string arg)
{
	if (arg.compare("--device") == 0) {
		return ARG_DEVICE;
	} else if (arg.compare("--input") == 0) {
		return ARG_IN_IMAGE;
	} else if (arg.compare("--video") == 0) {
		return ARG_IN_VIDEO;
	} else if (arg.compare("--visual") == 0) {
		return ARG_VISUAL;
	} else if (arg.compare("--load") == 0) {
		return ARG_LOAD;
	} else if (arg.compare("--train") == 0) {
		return ARG_TRAIN;
	} else {
		return ARG_UNKNOWN;
	}
}
