/*
* @Author: pkar
* @Date:   2016-02-22 18:34:37
* @Last Modified by:   pkar
* @Last Modified time: 2016-02-22 19:54:35
*/

#include <iostream>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <string>
//OpenCV Headers
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "svmPredictVector.h"

using namespace std;
using namespace cv;

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

vector<string> getImagePathNameVector(string image_root_dir, string file_names_list)
{
    file_names_list = image_root_dir + file_names_list;
    vector<string> img_file_path_vec;
    FILE* fp = fopen(file_names_list.c_str(), "r");
    char img_name_str[1000];
    while(fscanf(fp, "%s", img_name_str) != EOF)
    {
        string img_name(img_name_str);
        img_name = image_root_dir + img_name;
        img_file_path_vec.push_back(img_name);
    }
    return img_file_path_vec;
}

int main(int argc, char const *argv[])
{
    Mat img;
    string dataset_folder = "../Dataset/Classes/everyclass/";
    string file_names_list = "file_names.txt";
    string model_file = "./face.model";
    struct svm_model* model = svm_load_model(model_file.c_str());
    vector<string> img_file_path_vec = getImagePathNameVector(dataset_folder, file_names_list);
    for (int i = 0; i < img_file_path_vec.size(); ++i)
    {
        img = imread(img_file_path_vec[i].c_str());
        vector<float> descriptorValues = getHogFeatureVector(img);
        // printf("class_label: %d\n", predictVector(descriptorValues, model_file.c_str()));
        int class_label = predictVectorModel(descriptorValues, model);
        char window_name[100];
        sprintf(window_name, "input - class: %d", class_label);
        imshow(window_name, img);
        int c = waitKey();
        if(c == 'p' || c == 'P')
            i -= 2;
    }
    return 0;
}