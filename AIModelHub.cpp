#include <opencv2/opencv.hpp>
#include "OnnxHelper.h"

void TestOpenCv()
{
	cv::Mat src = cv::imread("./testData//test.jpg", 1);
	//没有图像输入
	if (src.empty()) {
		printf("....\n");
	}
	//namedWindow("输入窗口", WINDOW_FREERATIO);
	cv::imshow("输入窗口", src);
	cv::waitKey(0);
	cv::destroyAllWindows();
}

int main(int argc, char** argv) {
	AIModelLoader::OnnxHelper helper;
	helper.test_detect_opencv();
	return 0;
}

