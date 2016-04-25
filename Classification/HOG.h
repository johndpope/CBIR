/*
* @Author: pkar
* @Date:   2016-03-03 01:43:43
* @Last Modified by:   pkar
* @Last Modified time: 2016-03-03 01:46:19
*/

#ifndef HOG_H
#define HOG_H

#include <iostream>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <string>
//OpenCV Headers
#include <opencv/cv.h>
#include <opencv/highgui.h>

using namespace std;
using namespace cv;

class HOG
{
    Mat img;
    Size img_size;
    Size win_size;
    Size win_stride;
    Size block_size;
    Size block_stride;
    Size cell_size;
    int nbins;
    vector<float> descriptors;
    vector<Point> locations;
    HOGDescriptor hog;
public:
    HOG(Mat frame);
    HOG(Mat frame, Size win_size, Size win_stride, Size block_size, Size block_stride, Size cell_size, int nbins);
    int getDescriptorSize();
    int windowsInImage();
    Rect getWindowRect(int idx);
    vector<float> getWindowDescriptor(int idx);
    vector<vector<float> > getAllWindowDescriptors();
};

#endif