/*
* @Author: Pratyush Kar
* @Date:   2016-03-03 20:22:39
* @Last Modified by:   Pratyush Kar
* @Last Modified time: 2016-03-22 13:59:15
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
// Eigen Headers
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Eigen>
#include <eigen3/Eigen/Dense>
// PCA
#include "PCA.h"
// HOG
#include "HOG.h"

using namespace std;
using namespace cv;
using namespace Eigen;

#define IMG_HEIGHT 64
#define IMG_WIDTH 64
#define MAX_RAW_IMG_HEIGHT 600
#define MAX_RAW_IMG_WIDTH 1200
#define DETECTOR_THRESHOLD 0.9
#define DETECTION_WINDOW 5
#define OVERLAP_AREA 0.4
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

MatrixXf getEigenVectsfromFile(const char* str)
{
    FILE *fp = fopen(str, "r");
    int m, n;
    fscanf(fp, "%d%d", &m, &n);
    MatrixXf eigen_vects(m, n);
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
            fscanf(fp, "%f", &eigen_vects(i,j));
    }
    return eigen_vects;
}

vector<float> applyPCAtoVector(vector<float> &descriptorValues, MatrixXf &eigen_vects)
{
    MatrixXf datapoint(1,descriptorValues.size());
    for (int i = 0; i < descriptorValues.size(); ++i)
        datapoint(0,i) = descriptorValues[i];
    MatrixXf reduceddatapnt = pca::transformPointMatrix(datapoint, eigen_vects);
    vector<float> retfeatvect(reduceddatapnt.cols());
    for (int i = 0; i < reduceddatapnt.cols(); ++i)
        retfeatvect[i] = reduceddatapnt(0,i);
    return retfeatvect;
}

vector<vector<float> > applyPCAtoVector2D(vector<vector<float> > &descriptorValues, MatrixXf &eigen_vects)
{
    MatrixXf datapoints(descriptorValues.size(),descriptorValues[0].size());
    for (int i = 0; i < descriptorValues.size(); ++i)
        for (int j = 0; j < descriptorValues[0].size(); ++j)
            datapoints(i, j) = descriptorValues[i][j];
    MatrixXf reduceddatapnts = pca::transformPointMatrix(datapoints, eigen_vects);
    vector<vector<float> > retfeatvects(reduceddatapnts.rows(), vector<float>(reduceddatapnts.cols()));
    for (int i = 0; i < reduceddatapnts.rows(); ++i)
        for (int j = 0; j < reduceddatapnts.cols(); ++j)
            retfeatvects[i][j] = reduceddatapnts(i,j);
    return retfeatvects;
}

vector<Rect> naiveRemoveOverlappingDetections(vector<Rect> detections)
{
    if(detections.size() == 0)
        return detections;
    double orig_area = detections[0].width * detections[0].height;
    vector<Rect> second;
    for (int i = 0; i < detections.size(); ++i)
    {
        int x11 = detections[i].x;
        int y11 = detections[i].y;
        int x12 = detections[i].x + detections[i].width;
        int y12 = detections[i].y + detections[i].height;
        bool flag = false;
        for (int j = 0; j < second.size(); ++j)
        {
            int x21 = detections[j].x;
            int y21 = detections[j].y;
            int x22 = detections[j].x + detections[j].width;
            int y22 = detections[j].y + detections[j].height;
            double x_overlap = max(0, min(x12, x22) - max(x11, x21));
            double y_overlap = max(0, min(y12, y22) - max(y11, y21));
            if(x_overlap > detections[i].width/2 && y_overlap > detections[i].height/2)
            {
                flag = true;
                break;
            }
            double overlap_area = x_overlap * y_overlap;
            if(overlap_area/orig_area > OVERLAP_AREA)
            {
                printf("yes\n");
                flag = true;
                break;
            }
        }
        if(flag)
            continue;
        else
            second.push_back(detections[i]);
    }
    for (int i = 2; i < 4; ++i)
    {
        int x11 = second[i].x;
        int y11 = second[i].y;
        int x12 = second[i].x + second[i].width;
        int y12 = second[i].y + second[i].height;
        printf("x11: %d y11: %d x12: %d y12: %d\n", x11, y11, x12, y12);
    }
    return second;
}

int main(int argc, char const *argv[])
{
    clock_t totalStart = clock();
    Mat orig_img = imread("../Dataset/Original Images/mandawa-19mar36_21.jpg", CV_LOAD_IMAGE_COLOR);
    MatrixXf eigen_vects = getEigenVectsfromFile("eigen.txt");
    if(orig_img.rows == 0)
    {
        printf("no image\n");
        return -1;
    }
    string model_file = "./flower.model";
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
    int iterations = 0;
    double aver_time_hog = 0.0;
    double aver_time_pca = 0.0;
    double aver_time_svm = 0.0;
    Size win_size = Size(64, 64);
    Size win_stride = Size(8, 8);
    Size block_size = Size(16, 16);
    Size block_stride = Size(8, 8);
    Size cell_size = Size(8, 8);
    int nbins = 9;
    printf("Calculating HOG Descriptors... ");
    fflush(stdout);
    clock_t tStart = clock();
    HOG hog(img_gray, win_size, win_stride, block_size, block_stride, cell_size, nbins);
    int nwindows = hog.windowsInImage();
    vector<vector<float> > descriptorValues = hog.getAllWindowDescriptors();
    aver_time_hog += ((double)(clock() - tStart) * 1000/CLOCKS_PER_SEC);
    printf("Done\n");
    printf("Calculating PCA transform... ");
    fflush(stdout);
    tStart = clock();
    vector<vector<float> > datapoints = applyPCAtoVector2D(descriptorValues, eigen_vects);
    aver_time_pca += ((double)(clock() - tStart) * 1000/CLOCKS_PER_SEC);
    printf("Done\n");
    printf("Total number of windows to be detected: %d\n", nwindows);
    for (int i = 0; i < nwindows; i++)
    {
        tStart = clock();
        int class_label = predictVectorModel(datapoints[i], model);
        aver_time_svm += ((double)(clock() - tStart) * 1000/CLOCKS_PER_SEC);
        if(class_label == 1)
            detections.push_back(hog.getWindowRect(i));
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
    vector<Rect> nonoverlapdetections = naiveRemoveOverlappingDetections(detections);
    for (int i = 0; i < nonoverlapdetections.size(); ++i)
    {
        Rect myROI = nonoverlapdetections[i];
        int lefttopx = myROI.x;
        int lefttopy = myROI.y;
        int rightbottomx = lefttopx + myROI.width;
        int rightbottomy = lefttopy + myROI.height;
        rectangle(img, Point(lefttopx, lefttopy), Point(rightbottomx, rightbottomy), CV_RGB(0, 255, 0));
    }
    aver_time_hog /= (double) nwindows;
    aver_time_svm /= (double) nwindows;
    aver_time_pca /= (double) nwindows;
    printf("\nThe average time for the HOG detector to run: %lf (ms)\n", aver_time_hog);
    printf("The average time for the SVM detector to run: %lf (ms)\n", aver_time_svm);
    printf("The average time for the PCA transform to run: %lf (ms)\n", aver_time_pca);
    double total_time_taken = ((double)(clock() - totalStart) * 1000/CLOCKS_PER_SEC);
    printf("Total time taken for detection: %lf (s)\n", total_time_taken/1000);
    imshow("detections", img);
    imwrite("detections.jpg", img);
    waitKey();
    return 0;
}