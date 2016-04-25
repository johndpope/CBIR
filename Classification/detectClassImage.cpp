/*
* @Author: pkar
* @Date:   2016-02-22 19:59:09
* @Last Modified by:   pkar
* @Last Modified time: 2016-02-29 00:42:14
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

using namespace std;
using namespace cv;

#define IMG_HEIGHT 64
#define IMG_WIDTH 64
#define MAX_RAW_IMG_HEIGHT 600
#define MAX_RAW_IMG_WIDTH 1200
#define DETECTOR_THRESHOLD 0.9
#define DETECTION_WINDOW 5
// #define NEARBY_LOOKUP

vector<float> getHogFeatureVector(Mat img)
{
    Mat img_gray;
    // imshow("input", img);
    cvtColor(img, img_gray, CV_RGB2GRAY);
    // imshow("gray", img_gray);
    Size win_size = Size(64, 64);
    Size block_size = Size(16, 16);
    Size block_stride = Size(8, 8);
    Size cell_size = Size(8, 8);
    int nbins = 9;
    HOGDescriptor hog(win_size, block_size, block_stride, cell_size, nbins);
    vector<float> descriptorValues;
    vector<Point> locations;
    hog.compute(img_gray, descriptorValues, Size(0,0), Size(0,0), locations);
    return descriptorValues;
}

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
    int width = img.cols;
    int height = img.rows;
    float progress = 0.0;
    double aver_time_hog = 0.0;
    int iterations = 0;
    for (int i = 1; i < width; i += IMG_WIDTH/8)
    {
        for (int j = 1; j < height; j += IMG_HEIGHT/8)
        {
            iterations++;
            int lefttopx = i;
            int lefttopy = j;
            int rightbottomx = lefttopx + IMG_WIDTH;
            int rightbottomy = lefttopy + IMG_HEIGHT;
            if(lefttopx < 0 || lefttopy < 0 || rightbottomx >= width || rightbottomy >= height)
                continue;
            Rect myROI(lefttopx, lefttopy, IMG_WIDTH, IMG_HEIGHT);
            Mat roi = img(myROI);
            Mat sample;
            Size sz(64, 64);
            resize(roi, sample, sz);
            clock_t tStart = clock();
            vector<float> descriptorValues = getHogFeatureVector(sample);
            aver_time_hog += ((double)(clock() - tStart) * 1000/CLOCKS_PER_SEC);
            int class_label = predictVectorModel(descriptorValues, model);
            #ifdef NEARBY_LOOKUP
            if(class_label == 1)
            {
                int detect_cnt = 0;
                int valid_cnt = 0;
                for (int ii = -DETECTION_WINDOW; ii < DETECTION_WINDOW; ++ii)
                {
                    for (int jj = -DETECTION_WINDOW; jj < DETECTION_WINDOW; ++jj)
                    {
                        int templeftx = lefttopx + ii;
                        int templefty = lefttopy + jj;
                        int wdth = IMG_WIDTH - 2*ii;
                        int hgth = IMG_HEIGHT - 2*jj;
                        int temprightx = templeftx + wdth;
                        int temprighty = templefty + hgth;
                        if(templeftx < 0 || templefty < 0 || temprightx >= width || temprighty >= height)
                            continue;
                        valid_cnt++;
                        Rect mroi(templeftx, templefty, wdth, hgth);
                        Mat dimg = img(mroi);
                        Mat smpl;
                        resize(dimg, smpl, sz);
                        vector<float> dvalue = getHogFeatureVector(smpl);
                        int clabel = predictVectorModel(dvalue, model);
                        if(clabel == 1)
                            detect_cnt++;
                    }
                }
                printf("detect_cnt: %d valid_cnt: %d occ: %lf\n", detect_cnt, valid_cnt, (double (detect_cnt))/(double (valid_cnt)));
                if((double (detect_cnt))/(double (valid_cnt)) > DETECTOR_THRESHOLD)
                    detections.push_back(myROI);
            }
            #endif
            #ifndef NEARBY_LOOKUP
            if(class_label == 1)
                detections.push_back(myROI);
            #endif
            progress = double (i * height + j)/ double (width * height);
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
    aver_time_hog /= (double) iterations;
    printf("\nThe average time for the HOG detector to run: %lf (ms)\n", aver_time_hog);
    imshow("detections", img);
    imwrite("detections.jpg", img);
    waitKey();
    return 0;
}