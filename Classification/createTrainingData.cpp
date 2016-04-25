/*
* @Author: pkar
* @Date:   2016-02-22 17:45:43
* @Last Modified by:   pkar
* @Last Modified time: 2016-02-28 18:10:28
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

int main(int argc, char const *argv[]){
    string dataset_folder_face = "../Dataset/Classes/face/";
    string dataset_folder_other = "../Dataset/Classes/other/";
    string file_names_list_face = "file_names.txt";
    string file_names_list_other = "file_names.txt";
    vector<string> img_file_path_vec_face = getImagePathNameVector(dataset_folder_face, file_names_list_face);
    vector<vector<float> > v_descriptorValues_face;
    for (int i = 0; i < img_file_path_vec_face.size(); ++i)
    {
        Mat img;
        string file_name = img_file_path_vec_face[i];
        img = imread(file_name.c_str());
        vector<float> descriptorValues = getHogFeatureVector(img);
        v_descriptorValues_face.push_back(descriptorValues);
        flip(img, img, 1);
        descriptorValues = getHogFeatureVector(img);
        v_descriptorValues_face.push_back(descriptorValues);
    }
    printf("Number of Face Datapoints: %lu\n", v_descriptorValues_face.size());
    vector<string> img_file_path_vec_other = getImagePathNameVector(dataset_folder_other, file_names_list_other);
    vector<vector<float> > v_descriptorValues_other;
    for (int i = 0; i < img_file_path_vec_other.size(); ++i)
    {
        Mat img;
        string file_name = img_file_path_vec_other[i];
        img = imread(file_name.c_str());
        vector<float> descriptorValues = getHogFeatureVector(img);
        v_descriptorValues_other.push_back(descriptorValues);
        flip(img, img, 1);
        descriptorValues = getHogFeatureVector(img);
        v_descriptorValues_other.push_back(descriptorValues);
    }
    printf("Number of Other Datapoints: %lu\n", v_descriptorValues_other.size());
    FILE *fp = fopen("face.dat", "w");
    for (int i = 0; i < v_descriptorValues_face.size(); ++i)
    {
        fprintf(fp, "+1 ");
        for (int j = 0; j < v_descriptorValues_face[0].size(); ++j)
            fprintf(fp, "%d:%f ", j+1, v_descriptorValues_face[i][j]);
        fprintf(fp, "\n");
    }
    for (int i = 0; i < v_descriptorValues_other.size(); ++i)
    {
        fprintf(fp, "-1 ");
        for (int j = 0; j < v_descriptorValues_other[0].size(); ++j)
            fprintf(fp, "%d:%f ", j+1, v_descriptorValues_other[i][j]);
        fprintf(fp, "\n");
    }
    fclose(fp);
    return 0;
}