/*
* @Author: Pratyush Kar
* @Date:   2016-03-03 02:33:08
* @Last Modified by:   Pratyush Kar
* @Last Modified time: 2016-03-03 02:33:08
*/

#include <iostream>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <string>
#include <ctime>
//OpenCV Headers
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "svmPredictVector.h"
#include "HOG.h"

using namespace std;
using namespace cv;

#define IMG_HEIGHT 64
#define IMG_WIDTH 64
#define MAX_RAW_IMG_HEIGHT 600
#define MAX_RAW_IMG_WIDTH 1200
#define DETECTOR_THRESHOLD 0.9
#define DETECTION_WINDOW 5
// #define NEARBY_LOOKUP

int main(int argc, char const *argv[])
{
    Mat orig_img = imread("../Dataset/Original Images/mandawa-19mar45.jpg", CV_LOAD_IMAGE_COLOR);
    if(orig_img.rows == 0)
    {
        printf("no image\n");
        return -1;
    }
    string model_file = "./face.model";
    struct svm_model* model = svm_load_model(model_file.c_str());
    Mat img;
    if(orig_img.rows > MAX_RAW_IMG_HEIGHT)
    {
        int height = MAX_RAW_IMG_HEIGHT;
        int width = (int) (((double) orig_img.cols / (double) orig_img.rows) * MAX_RAW_IMG_HEIGHT);
        Size sz(width, height);
        resize(orig_img, img, sz);
    }
    else
        img = orig_img;
    vector<Rect> detections;
    Mat img_gray;
    cvtColor(img, img_gray, CV_RGB2GRAY);
    int width = img.cols;
    int height = img.rows;
    float progress = 0.0;
    double aver_time_svm = 0.0;
    double aver_time_hog = 0.0;
    Size win_size = Size(64, 64);
    Size win_stride = Size(8, 8);
    Size block_size = Size(16, 16);
    Size block_stride = Size(8, 8);
    Size cell_size = Size(8, 8);
    int nbins = 9;
    clock_t tStart = clock();
    HOG hog(img_gray, win_size, win_stride, block_size, block_stride, cell_size, nbins);
    aver_time_hog += ((double)(clock() - tStart) * 1000/CLOCKS_PER_SEC);
    int nwindows = hog.windowsInImage();
    printf("Total number of windows to be detected: %d\n", nwindows);
    for (int i = 0; i < nwindows; i++)
    {
        tStart = clock();
        vector<float> descriptorValues = hog.getWindowDescriptor(i);
        aver_time_hog += ((double)(clock() - tStart) * 1000/CLOCKS_PER_SEC);
        Rect myROI = hog.getWindowRect(i);
        tStart = clock();
        int class_label = predictVectorModel(descriptorValues, model);
        aver_time_svm += ((double)(clock() - tStart) * 1000/CLOCKS_PER_SEC);
        if(class_label == 1)
            detections.push_back(myROI);
        progress = double (i)/ double (nwindows);
        int barWidth = 70;
        std::cout << "[";
        int pos = barWidth * progress;
        for (int i = 0; i < barWidth; ++i) {
            if (i < pos) std::cout << "=";
            else if (i == pos) std::cout << ">";
            else std::cout << " ";
        }
        std::cout << "] " << int(progress * 100.0) << " %\r";
        std::cout.flush();
    }
    for (int i = 0; i < detections.size(); ++i)
    {
        Rect myROI = detections[i];
        int lefttopx = myROI.x;
        int lefttopy = myROI.y;
        int rightbottomx = lefttopx + myROI.width;
        int rightbottomy = lefttopy + myROI.height;
        rectangle(img, Point(lefttopx, lefttopy), Point(rightbottomx, rightbottomy), CV_RGB(0, 255, 0));
    }
    aver_time_svm /= (double) nwindows;
    aver_time_hog /= (double) nwindows;
    printf("\nThe average time for the classifier to run: %lf (ms)\n", aver_time_svm);
    printf("The average time for the HOG detector to run: %lf (ms)\n", aver_time_hog);
    imshow("detections", img);
    imwrite("detections.jpg", img);
    // waitKey();
    return 0;
}