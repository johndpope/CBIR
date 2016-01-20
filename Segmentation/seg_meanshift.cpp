#include <cstdio>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <image.h>
#include <misc.h>
#include <pnmfile.h>
#include "segment-image.h"

//OpenCV Headers
#include <opencv/cv.h>
#include <opencv/highgui.h>

using namespace std;
using namespace cv;

int main(int argc, char const *argv[])
{
	if(argc != 2){
		fprintf(stderr, "Usage: ./%s <FILENAME>\n", argv[0]);
		return -1;
	}
	Mat frame = imread(argv[1]);
	Mat output;
	pyrMeanShiftFiltering(frame, output, 30, 30, 3);
	imshow("Original", frame);
	imshow("Mean Shift", output);
	cvWaitKey();
	return 0;
}