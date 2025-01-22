#include "OnnxHelper.h"
#include <fstream>
#include <iostream>
#include <filesystem>
#include <random>

using namespace std;

namespace AIModelLoader {
	OnnxHelper::OnnxHelper()
	{
	}

	int OnnxHelper::yolov5_detect()
	{
		// reference: ultralytics/examples/YOLOv8-CPP-Inference
		namespace fs = std::filesystem;

		auto net = cv::dnn::readNetFromONNX(v5_onnx_file);
		if (net.empty()) {
			std::cerr << "Error: there are no layers in the network: " << v5_onnx_file << std::endl;
			return -1;
		}

		if (cuda_enabled) {
			net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
			net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
		}
		else {
			net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
			net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
		}

		if (!fs::exists(result_dir)) {
			fs::create_directories(result_dir);
		}

		auto classes = parse_classes_file(classes_file);
		if (classes.size() == 0) {
			std::cerr << "Error: fail to parse classes file: " << classes_file << std::endl;
			return -1;
		}

		std::cout << "classes: ";
		for (const auto& val : classes) {
			std::cout << val << " ";
		}
		std::cout << std::endl;

		for (const auto& [key, val] : get_dir_images(images_dir)) {
			cv::Mat frame = cv::imread(val, cv::IMREAD_COLOR);
			if (frame.empty()) {
				std::cerr << "Warning: unable to load image: " << val << std::endl;
				continue;
			}

			cv::Mat bgr = modify_image_size(frame);

			cv::Mat blob;
			cv::dnn::blobFromImage(bgr, blob, 1.0 / 255.0, cv::Size(image_size[1], image_size[0]), cv::Scalar(), true, false);
			net.setInput(blob);

			std::vector<cv::Mat> outputs;
			net.forward(outputs, net.getUnconnectedOutLayersNames());

			// 获取输出的维度信息
			int rows = outputs[0].size[1]; // 每个预测框的总数（25200）
			int dimensions = outputs[0].size[2]; // 每个预测框的维度（num_classes + 5）

			float* data = (float*)outputs[0].data; // 获取模型输出的原始数据
			float x_factor = bgr.cols * 1.f / image_size[1]; // 缩放因子，用于将边界框坐标映射回原始图像
			float y_factor = bgr.rows * 1.f / image_size[0];

			std::vector<int> class_ids;
			std::vector<float> confidences;
			std::vector<cv::Rect> boxes;

			for (auto i = 0; i < rows; ++i) {
				float* row_data = data + i * dimensions; // 当前行的起始位置

				float confidence = row_data[4]; // 第5个值是置信度
				if (confidence > model_score_threshold) { // 置信度大于阈值
					cv::Mat scores(1, classes.size(), CV_32FC1, row_data + 5); // 类别分数从第6个值开始
					cv::Point class_id;
					double max_class_score;

					cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id); // 找到最大类别分数及其索引

					if (max_class_score > model_score_threshold) { // 类别分数大于阈值
						confidences.push_back(confidence * max_class_score); // 综合置信度和类别分数
						class_ids.push_back(class_id.x);

						float x = row_data[0];
						float y = row_data[1];
						float w = row_data[2];
						float h = row_data[3];

						int left = int((x - 0.5 * w) * x_factor);
						int top = int((y - 0.5 * h) * y_factor);
						int width = int(w * x_factor);
						int height = int(h * y_factor);

						boxes.push_back(cv::Rect(left, top, width, height));
					}
				}
			}

			std::vector<int> nms_result;
			cv::dnn::NMSBoxes(boxes, confidences, model_score_threshold, model_nms_threshold, nms_result);

			std::vector<int> ids;
			std::vector<float> confs;
			std::vector<cv::Rect> rects;
			for (size_t i = 0; i < nms_result.size(); ++i) {
				ids.emplace_back(class_ids[nms_result[i]]);
				confs.emplace_back(confidences[nms_result[i]]);
				rects.emplace_back(boxes[nms_result[i]]);
			}
			draw_boxes(classes, ids, confs, rects, key, frame);
		}

		return 0;
	}

	int OnnxHelper::yolov8_detect()
	{
		// reference: ultralytics/examples/YOLOv8-CPP-Inference
		namespace fs = std::filesystem;

		auto net = cv::dnn::readNetFromONNX(v8_onnx_file);
		if (net.empty()) {
			std::cerr << "Error: there are no layers in the network: " << v8_onnx_file << std::endl;
			return -1;
		}

		if (cuda_enabled) {
			net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
			net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
		}
		else {
			net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
			net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
		}

		if (!fs::exists(result_dir)) {
			fs::create_directories(result_dir);
		}

		auto classes = parse_classes_file(classes_file);
		if (classes.size() == 0) {
			std::cerr << "Error: fail to parse classes file: " << classes_file << std::endl;
			return -1;
		}

		std::cout << "classes: ";
		for (const auto& val : classes) {
			std::cout << val << " ";
		}
		std::cout << std::endl;

		for (const auto& [key, val] : get_dir_images(images_dir)) {
			cv::Mat frame = cv::imread(val, cv::IMREAD_COLOR);
			if (frame.empty()) {
				std::cerr << "Warning: unable to load image: " << val << std::endl;
				continue;
			}

			cv::Mat bgr = modify_image_size(frame);

			cv::Mat blob;
			cv::dnn::blobFromImage(bgr, blob, 1.0 / 255.0, cv::Size(image_size[1], image_size[0]), cv::Scalar(), true, false);
			net.setInput(blob);

			std::vector<cv::Mat> outputs;
			net.forward(outputs, net.getUnconnectedOutLayersNames());

			int rows = outputs[0].size[1];
			int dimensions = outputs[0].size[2];

			// yolov5 has an output of shape (batchSize, 25200, num classes+4+1) (Num classes + box[x,y,w,h] + confidence[c])
			// yolov8 has an output of shape (batchSize, num classes + 4,  8400) (Num classes + box[x,y,w,h])
			if (dimensions > rows) { // Check if the shape[2] is more than shape[1] (yolov8)
				rows = outputs[0].size[2];
				dimensions = outputs[0].size[1];

				outputs[0] = outputs[0].reshape(1, dimensions);
				cv::transpose(outputs[0], outputs[0]);
			}

			float* data = (float*)outputs[0].data;
			float x_factor = bgr.cols * 1.f / image_size[1];
			float y_factor = bgr.rows * 1.f / image_size[0];

			std::vector<int> class_ids;
			std::vector<float> confidences;
			std::vector<cv::Rect> boxes;

			for (auto i = 0; i < rows; ++i) {
				float* classes_scores = data + 4;

				cv::Mat scores(1, classes.size(), CV_32FC1, classes_scores);
				cv::Point class_id;
				double max_class_score;

				cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

				if (max_class_score > model_score_threshold) {
					confidences.push_back(max_class_score);
					class_ids.push_back(class_id.x);

					float x = data[0];
					float y = data[1];
					float w = data[2];
					float h = data[3];

					int left = int((x - 0.5 * w) * x_factor);
					int top = int((y - 0.5 * h) * y_factor);

					int width = int(w * x_factor);
					int height = int(h * y_factor);

					boxes.push_back(cv::Rect(left, top, width, height));
				}

				data += dimensions;
			}

			std::vector<int> nms_result;
			cv::dnn::NMSBoxes(boxes, confidences, model_score_threshold, model_nms_threshold, nms_result);

			std::vector<int> ids;
			std::vector<float> confs;
			std::vector<cv::Rect> rects;
			for (size_t i = 0; i < nms_result.size(); ++i) {
				ids.emplace_back(class_ids[nms_result[i]]);
				confs.emplace_back(confidences[nms_result[i]]);
				rects.emplace_back(boxes[nms_result[i]]);
			}
			draw_boxes(classes, ids, confs, rects, key, frame);
		}

		return 0;
	}

	cv::Mat OnnxHelper::modify_image_size(const cv::Mat& img)
	{
		auto max = std::max(img.rows, img.cols);
		cv::Mat ret = cv::Mat::zeros(max, max, CV_8UC3);
		img.copyTo(ret(cv::Rect(0, 0, img.cols, img.rows)));

		return ret;
	}

	std::vector<std::string> OnnxHelper::parse_classes_file(const char* name)
	{
		std::vector<std::string> classes;

		std::ifstream file(name);
		if (!file.is_open()) {
			std::cerr << "Error: fail to open classes file: " << name << std::endl;
			return classes;
		}

		std::string line;
		while (std::getline(file, line)) {
			auto pos = line.find_first_of(" ");
			classes.emplace_back(line.substr(pos, line.length()));
		}

		file.close();
		return classes;
	}

	std::map<std::string, std::string> OnnxHelper::get_dir_images(const char* name)
	{
		std::map<std::string, std::string> images; // image name, image path + image name

		for (auto const& dir_entry : std::filesystem::directory_iterator(name)) {
			if (dir_entry.is_regular_file())
				images[dir_entry.path().filename().string()] = dir_entry.path().string();
		}

		return images;
	}

	void OnnxHelper::draw_boxes(const std::vector<std::string>& classes, const std::vector<int>& ids, const std::vector<float>& confidences, const std::vector<cv::Rect>& boxes, const std::string& name, cv::Mat& frame)
	{
		if (ids.size() != confidences.size() || ids.size() != boxes.size() || confidences.size() != boxes.size()) {
			std::cerr << "Error: their lengths are inconsistent: " << ids.size() << ", " << confidences.size() << ", " << boxes.size() << std::endl;
			return;
		}

		std::cout << "image name: " << name << ", number of detections: " << ids.size() << std::endl;

		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_int_distribution<int> dis(100, 255);

		for (auto i = 0; i < ids.size(); ++i) {
			auto color = cv::Scalar(dis(gen), dis(gen), dis(gen));
			cv::rectangle(frame, boxes[i], color, 2);

			std::string class_string = classes[ids[i]] + ' ' + std::to_string(confidences[i]).substr(0, 4);
			cv::Size text_size = cv::getTextSize(class_string, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
			cv::Rect text_box(boxes[i].x, boxes[i].y - 40, text_size.width + 10, text_size.height + 20);

			cv::rectangle(frame, text_box, color, cv::FILLED);
			cv::putText(frame, class_string, cv::Point(boxes[i].x + 5, boxes[i].y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
		}

		cv::imshow("Inference", frame);
		cv::waitKey(-1);

		std::string path(result_dir);
		path += "/" + name;
		cv::imwrite(path, frame);
	}
}




