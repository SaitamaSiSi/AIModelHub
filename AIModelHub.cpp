#include <opencv2/opencv.hpp>
#include "OnnxHelper.h"

#include "Top100Liked.h"

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
	// AIModelLoader::OnnxHelper helper;
	// helper.yolov5_detect();
	// helper.yolov8_detect();

	/*Top100Liked::Solution::ListNode2* AList;
	Top100Liked::Solution::ListNode2** insertA = &AList;
	vector<int> dataA{ 4,1,8,4,5 };
	for (int i = 0; i < dataA.size(); ++i) {
		*insertA = new Top100Liked::Solution::ListNode2(dataA[i]);
		insertA = &((*insertA)->next);
	}
	Top100Liked::Solution::ListNode2* temp = nullptr;
	temp = AList;
	while (temp != nullptr) {
		auto* current = temp;
		temp = temp->next;
		delete current;
	}*/

	Top100Liked::Solution solution;

	return 0;
}

