/*
* @Author: pkar
* @Date:   2016-02-18 22:19:04
* @Last Modified by:   pkar
* @Last Modified time: 2016-02-19 17:16:03
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

using namespace std;
using namespace cv;

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

vector<float> getHogFeatureVector(Mat img)
{
    Mat img_gray;
    imshow("input", img);
    cvtColor(img, img_gray, CV_RGB2GRAY);
    imshow("gray", img_gray);
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

float meanVect(vector<float> feats)
{
    float sum = 0.0;
    for (int i = 0; i < feats.size(); ++i)
        sum += feats[i];
    return sum/feats.size();
}

vector<vector<float> > getFeatureVectorfromXML(string xmlfile)
{
    FileStorage read_hogXml("V_writeTest.xml", FileStorage::READ);
    int row, col;
    Mat M;
    read(read_hogXml["Descriptor_of_images"], M);
    row = M.rows;
    col = M.cols;
    vector<vector<float> > v_descriptorValues;
    for (int i = 0; i < row; ++i)
    {
        for (int i = 0; i < col; ++i)
        {
            vector<float> temp;
            int start = col * i * sizeof(float);
            int end = start + col * sizeof(float)-1;
            temp.assign( (float*)(&(M.data[start])), (float*)(&(M.data[end])) );
            v_descriptorValues.push_back(temp);
        }
    }
    read_hogXml.release();
    return v_descriptorValues;
}

int main(){
    string dataset_folder = "../Dataset/Classes/face/";
    string file_names_list = "file_names.txt";
    vector<string> class_name_list;
    class_name_list.push_back("face");
    class_name_list.push_back("other");
    vector<string> img_file_path_vec = getImagePathNameVector(dataset_folder, file_names_list);
    vector<vector<float> > v_descriptorValues;
    for (int i = 0; i < img_file_path_vec.size(); ++i)
    {
        Mat img;
        string file_name = img_file_path_vec[i];
        img = imread(file_name.c_str());
        vector<float> descriptorValues = getHogFeatureVector(img);
        v_descriptorValues.push_back(descriptorValues);
        flip(img, img, 1);
        descriptorValues = getHogFeatureVector(img);
        v_descriptorValues.push_back(descriptorValues);
        printf("Size: %d\tMean: %lf\n", (int) descriptorValues.size(), meanVect(descriptorValues));
    }
    FileStorage hogXml("output.xml", FileStorage::WRITE);
    int row = v_descriptorValues.size(), col = v_descriptorValues[0].size();
    Mat M(row,col,CV_32F);
    for(int i=0; i< row; ++i)
        memcpy( &(M.data[col * i * sizeof(float) ]) ,v_descriptorValues[i].data(),col*sizeof(float));
    write(hogXml, "descriptors",  M);
    hogXml.release();
    return 0;
}