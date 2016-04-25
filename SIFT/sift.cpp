#include <iostream>
//OpenCV Headers
#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv/cvaux.h>
#include <opencv/highgui.h>
#include <vector>

using namespace cv;
using namespace std;

int main(int argc, const char* argv[])
{
    const cv::Mat input = cv::imread("img1.jpg", 0); //Load as grayscale

    SiftFeatureDetector detector;
    vector<cv::KeyPoint> keypoints;
    detector.detect(input, keypoints);

    // Add results to image and save.
    cv::Mat output;
    cv::drawKeypoints(input, keypoints, output);
    cv::imwrite("sift_result.jpg", output);

    return 0;
}
