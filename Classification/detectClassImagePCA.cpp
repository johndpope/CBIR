/*
* @Author: pkar
* @Date:   2016-02-28 18:57:12
* @Last Modified by:   pkar
* @Last Modified time: 2016-02-29 00:46:34
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

using namespace std;
using namespace cv;
using namespace Eigen;

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

MatrixXf getEigenVectsfromFile(const char* str)
{
    FILE *fp = fopen(str, "r");
    int m, n;
    fscanf(fp, "%d%d", &m, &n);
    // printf("%d %d\n", m, n);
    MatrixXf eigen_vects(m, n);
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            fscanf(fp, "%f", &eigen_vects(i,j));
            // printf("%f ", eigen_vects(i,j));
        }
        // printf("\n");
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

int main(int argc, char const *argv[])
{
    clock_t totalStart = clock();
    Mat orig_img = imread("../Dataset/Original Images/mandawa-19mar45.jpg", CV_LOAD_IMAGE_COLOR);
    MatrixXf eigen_vects = getEigenVectsfromFile("eigen.txt");
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
    int iterations = 0;
    double aver_time_hog = 0.0;
    double aver_time_pca = 0.0;
    double aver_time_svm = 0.0;
    clock_t tStart;
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
            tStart = clock();
            vector<float> descriptorValues = getHogFeatureVector(sample);
            aver_time_hog += ((double)(clock() - tStart) * 1000/CLOCKS_PER_SEC);
            tStart = clock();
            vector<float> reducedVect = applyPCAtoVector(descriptorValues, eigen_vects);
            aver_time_pca += ((double)(clock() - tStart) * 1000/CLOCKS_PER_SEC);
            tStart = clock();
            int class_label = predictVectorModel(reducedVect, model);
            aver_time_svm += ((double)(clock() - tStart) * 1000/CLOCKS_PER_SEC);
            #ifdef NEARBY_LOOKUP
            if(class_label == 1)
            {
                int detect_cnt = 0;
                int valid_cnt = 0;
                for (int ii = -DETECTION_WINDOW; ii < DETECTION_WINDOW; ++ii)
                {
                    for (int jj = -DETECTION_WINDOW; jj < DETECTION_WINDOW; ++jj)
                    {
                        iterations++;
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
                        vector<float> redDesc = applyPCAtoVector(dvalue, eigen_vects);
                        int clabel = predictVectorModel(redDesc, model);
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
    aver_time_svm /= (double) iterations;
    aver_time_pca /= (double) iterations;
    // printf("Number of total iterations: %d\n", iterations);
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