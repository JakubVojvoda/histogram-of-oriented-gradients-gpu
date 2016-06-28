/**
 *  Histogram of oriented gradient on GPU 
 *  by Jakub Vojvoda, vojvoda@swdeveloper.sk
 *  2015  
 *
 *  licence: GNU LGPL v3
 *  file: model.h
 */

#ifndef __model_h__
#define __model_h__

#include <string>
#include <vector>

#include <opencv2/highgui/highgui.hpp>

#define SVM_ITERATIONS 50
#define PREDICTION_THRESHOLD 0

#define MIN_SCALE 0.6
#define SCALING_STEP 0.1f

#define HISTOGRAM_BINS 9
#define CELL_WIDTH  8
#define CELL_HEIGHT 8
#define BLOCK_WIDTH  16
#define BLOCK_HEIGHT 16

#define CELLS_IN_BLOCK  ((BLOCK_WIDTH/CELL_WIDTH) * (BLOCK_HEIGHT/CELL_HEIGHT))
#define DESCRIPTOR_SIZE (CELLS_IN_BLOCK * HISTOGRAM_BINS)

#define WINDOW_WIDTH   48
#define WINDOW_HEIGHT 144

#ifndef M_PI
	#define M_PI 3.14159265f
#endif

bool loadSVM(std::string filename, std::vector<float> &y, float &b);
bool trainSVM(std::string positive, std::string negative, std::vector<float> &y, float &b);

void setDescriptor(std::vector<float> desc, float *window_desc, int size);

void drawHistogram(cv::Mat &image, cv::Point position, std::vector<float> histogram);
void windowVisualization(cv::Mat &image, cv::Rect roi, std::vector<float> desc);
cv::Mat visualization(cv::Mat image, int win_width, int win_height, std::vector<std::vector<float> > desc, bool window_grid = false, bool block_grid = false);

#endif