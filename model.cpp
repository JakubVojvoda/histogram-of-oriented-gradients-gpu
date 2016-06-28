/**
 *  Histogram of oriented gradient on GPU 
 *  by Jakub Vojvoda, vojvoda@swdeveloper.sk
 *  2015  
 *
 *  licence: GNU LGPL v3
 *  file: model.cpp
 */

#include "model.h"

#include <fstream>
#include <sstream>
#include <iterator>

#include <opencv2/imgproc/imgproc.hpp>

#include "sgdSVM.h"

bool loadSVM(std::string filename, std::vector<float> &y, float &b)
{
	std::ifstream file(filename.c_str());

	if (!file.is_open()) {
		return false;
	}

	std::string model;
	std::getline(file, model);

	std::istringstream s(model);
	std::copy(std::istream_iterator<float>(s), std::istream_iterator<float>(), std::back_inserter(y));

	b = y.back();
	y.pop_back();

	return true;
}

bool trainSVM(std::string positive, std::string negative, std::vector<float> &y, float &b)
{
	std::ifstream pos, neg;
	pos.open(positive.c_str());
	neg.open(negative.c_str());

	if (!pos.is_open() || !neg.is_open()) {
		return false;
	}

	std::vector<std::vector<float> > training_data;
	std::vector<double> training_labels;

	std::string line;

	while (std::getline(pos, line)) {

		std::istringstream s(line);

		std::vector<float> row;
		std::copy(std::istream_iterator<float>(s), std::istream_iterator<float>(), std::back_inserter(row));

		training_data.push_back(row);
		training_labels.push_back(+1);
	}

	while (std::getline(neg, line)) {

		std::istringstream s(line);

		std::vector<float> row;
		std::copy(std::istream_iterator<float>(s), std::istream_iterator<float>(), std::back_inserter(row));

		training_data.push_back(row);
		training_labels.push_back(-1);
	}

	std::vector<TFloatRep *> svm_data;
	const int dim = (int)training_data[0].size();

	for (unsigned int i = 0; i < training_data.size(); i++) {

		svm_data.push_back(new TFloatRep[dim]);

		for (int j = 0; j < dim; j++) {
			svm_data.back()[j] = training_data[i][j];
		}
	}

	TSVMSgd svm(dim, 0.001, "HINGELOSS");

	for (int i = 0; i < SVM_ITERATIONS; i++) {
		svm.train(svm_data, training_labels);
	}

	y = svm.getW();
	b = y.back();
	y.pop_back();

	return true;
}

void setDescriptor(std::vector<float> desc, float *window_desc, int size)
{
	for (unsigned int i = 0; i < desc.size(); i++) {

		if ((int)i < size) {
			window_desc[i] = desc.at(i);
		}
	}
}

// HOG descriptor visualization
// 
void drawHistogram(cv::Mat &image, cv::Point position, std::vector<float> histogram)
{
	int cx = position.x;
	int cy = position.y;

	float bin_range = M_PI / (float)histogram.size();

	float norm = 1.0;

	for (unsigned int i = 0; i < histogram.size(); i++) {
		norm += histogram.at(i);
	}

	norm /= (float)histogram.size();

	for (unsigned int i = 0; i < histogram.size(); i++) {

		float strength = histogram.at(i);
		float rad = i * bin_range + bin_range * 2.0f;

		float vx = cos(rad);
		float vy = sin(rad);
		float len = 6.0f;
		float scale = 0.8f;

		if (histogram.at(i) > 1) {
			strength = 1;
		}

		int x1 = static_cast<int>(cx - vx * strength * len * scale);
		int x2 = static_cast<int>(cx + vx * strength * len * scale);

		int y1 = static_cast<int>(cy - vy * strength * len * scale);
		int y2 = static_cast<int>(cy + vy * strength * len * scale);

		cv::line(image, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0));
	}

}

// Window HOG descriptor visualization
//
void windowVisualization(cv::Mat &image, cv::Rect roi, std::vector<float> desc)
{
	int bins = HISTOGRAM_BINS;
	float bin_range = M_PI / (float)bins;

	int block_desc_size = 4 * bins;

	int index = 0;

	for (int y = 0; y < roi.height; y += BLOCK_HEIGHT) {
		for (int x = 0; x < roi.width; x += BLOCK_WIDTH) {

			std::vector<float> cell_1(bins);
			std::vector<float> cell_2(bins);
			std::vector<float> cell_3(bins);
			std::vector<float> cell_4(bins);

			for (int i = 0; i < bins; i++) {

				cell_1.at(i) = desc.at(i + index * block_desc_size);
				cell_2.at(i) = desc.at((i + bins) + index * block_desc_size);
				cell_3.at(i) = desc.at((i + 2 * bins) + index * block_desc_size);
				cell_4.at(i) = desc.at((i + 3 * bins) + index * block_desc_size);
			}

			drawHistogram(image, cv::Point(roi.x + x + 4, roi.y + y + 4), cell_1);
			drawHistogram(image, cv::Point(roi.x + x + 12, roi.y + y + 4), cell_2);
			drawHistogram(image, cv::Point(roi.x + x + 4, roi.y + y + 12), cell_3);
			drawHistogram(image, cv::Point(roi.x + x + 12, roi.y + y + 12), cell_4);

			index += 1;
		}
	}
}

// HOG descriptor visualization
//
cv::Mat visualization(cv::Mat image, int win_width, int win_height, std::vector<std::vector<float> > desc, bool window_grid, bool block_grid)
{
	cv::Mat gray, result;
	cv::cvtColor(image, gray, CV_BGR2GRAY);
	cv::cvtColor(gray, result, CV_GRAY2BGR);

	result = cv::Mat::zeros(image.size(), CV_8UC3);

	int img_width = image.size().width;
	int img_height = image.size().height;

	if (block_grid) {

		for (int x = 0; x < img_width; x += 8) {
			cv::line(result, cv::Point(x, 0), cv::Point(x, img_height), cv::Scalar(0, 255, 0));
		}

		for (int y = 0; y < img_height; y += 8) {
			cv::line(result, cv::Point(0, y), cv::Point(img_width, y), cv::Scalar(0, 255, 0));
		}
	}

	int index = 0;

	for (int y = 0; y < img_height; y += win_height) {
		for (int x = 0; x < img_width; x += win_width) {

			cv::Rect roi(x, y, win_width, win_height);

			if (window_grid) {
				cv::rectangle(result, roi, cv::Scalar(0, 255, 255));
			}

			windowVisualization(result, roi, desc.at(index));
			index += 1;
		}
	}

	return result;
}
