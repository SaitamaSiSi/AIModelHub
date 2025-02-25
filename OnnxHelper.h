#pragma once
#include <string>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

// 参考文章 https://blog.csdn.net/fengbingchun/article/details/139203567

namespace AIModelLoader {
	constexpr bool cuda_enabled{ false };
	constexpr int image_size[2]{ 640, 640 }; // {height,width}, input shape (1, 3, 640, 640) BCHW and output shape(s) (1, 6, 8400)
	constexpr float model_score_threshold{ 0.25 }; // confidence threshold
	constexpr float model_nms_threshold{ 0.40 }; // iou threshold

#ifdef _MSC_VER
	const constexpr char* v5_onnx_file{ "../111NotGitHub/testData/onnx/yolov5n.onnx" };
	const constexpr char* v8_onnx_file{ "../111NotGitHub/testData/onnx/yolov8x.onnx" };
	//const constexpr char* torchscript_file{ "../111NotGitHub/testData/best.torchscript" };
	const constexpr char* images_dir{ "../111NotGitHub/testData/predict" };
	const constexpr char* result_dir{ "../111NotGitHub/testData/result" };
	const constexpr char* classes_file{ "../111NotGitHub/testData/images/labels.txt" };
#else
	const constexpr char* v5_onnx_file{ ".testData/yolov5n.onnx" };
	const constexpr char* v8_onnx_file{ ".testData/yolov8x.onnx" };
	//const constexpr char* torchscript_file{ "./testData/best.torchscript" };
	const constexpr char* images_dir{ "./testData/images/predict" };
	const constexpr char* result_dir{ "./testData/result" };
	const constexpr char* classes_file{ "./testData/images/labels.txt" };
#endif

	const float anchors_640[3][6] = { {10.0,  13.0, 16.0,  30.0,  33.0,  23.0},
					  {30.0,  61.0, 62.0,  45.0,  59.0,  119.0},
					  {116.0, 90.0, 156.0, 198.0, 373.0, 326.0} };

	const float anchors_1280[4][6] = { {19, 27, 44, 40, 38, 94},
									   {96, 68, 86, 152, 180, 137},
									   {140, 301, 303, 264, 238, 542},
					   {436, 615, 739, 380, 925, 792} };

	class OnnxHelper
	{
	public:
		OnnxHelper();
		int yolov5_detect();
		int yolov8_detect();

	private:
		cv::Mat modify_image_size(const cv::Mat& img);
		std::vector<std::string> parse_classes_file(const char* name);
		std::map<std::string, std::string> get_dir_images(const char* name);
		void draw_boxes(const std::vector<std::string>& classes, const std::vector<int>& ids, const std::vector<float>& confidences,
			const std::vector<cv::Rect>& boxes, const std::string& name, cv::Mat& frame);
	};
}


