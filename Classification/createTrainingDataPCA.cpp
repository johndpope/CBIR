/*
* @Author: pkar
* @Date:   2016-02-28 18:09:54
* @Last Modified by:   Pratyush Kar
* @Last Modified time: 2016-03-21 16:50:56
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
// Eigen Headers
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Eigen>
#include <eigen3/Eigen/Dense>
// PCA
#include "PCA.h"

#define NUM_EIGEN_VECTORS 30

using namespace std;
using namespace cv;
using namespace Eigen;

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
    cvtColor(img, img_gray, CV_RGB2GRAY);
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

MatrixXf getMatrixXffromVector(vector<vector<float> > &data)
{
    MatrixXf matr(data.size(), data[0].size());
    for (int i = 0; i < data.size(); ++i)
        for (int j = 0; j < data[0].size(); ++j)
            matr(i,j) = data[i][j];
    return matr;
}

int main(int argc, char const *argv[]){
    // string dataset_folder_face = "../Dataset/Classes/face/";
    // string dataset_folder_other = "../Dataset/Classes/other/";
    string dataset_folder_face = "../Dataset/Classes/flower/";
    string dataset_folder_other = "../Dataset/Classes/othernew/";
    string file_names_list_face = "file_names.txt";
    string file_names_list_other = "file_names.txt";
    vector<string> img_file_path_vec_face = getImagePathNameVector(dataset_folder_face, file_names_list_face);
    vector<vector<float> > v_descriptor;
    vector<bool> v_classtype;
    for (int i = 0; i < img_file_path_vec_face.size(); ++i)
    {
        Mat img;
        string file_name = img_file_path_vec_face[i];
        img = imread(file_name.c_str());
        vector<float> descriptorValues = getHogFeatureVector(img);
        v_descriptor.push_back(descriptorValues);
        v_classtype.push_back(1);
        flip(img, img, 1);
        descriptorValues = getHogFeatureVector(img);
        v_descriptor.push_back(descriptorValues);
        v_classtype.push_back(1);
    }
    vector<string> img_file_path_vec_other = getImagePathNameVector(dataset_folder_other, file_names_list_other);
    for (int i = 0; i < img_file_path_vec_other.size(); ++i)
    {
        Mat img;
        string file_name = img_file_path_vec_other[i];
        img = imread(file_name.c_str());
        vector<float> descriptorValues = getHogFeatureVector(img);
        v_descriptor.push_back(descriptorValues);
        v_classtype.push_back(0);
        flip(img, img, 1);
        descriptorValues = getHogFeatureVector(img);
        v_descriptor.push_back(descriptorValues);
        v_classtype.push_back(0);
    }
    MatrixXf datamat = getMatrixXffromVector(v_descriptor);
    pca::PCA pca(datamat);
    pca.showVariance(NUM_EIGEN_VECTORS);
    MatrixXf eigenVects = pca.getEigenVectorsCnt(NUM_EIGEN_VECTORS);
    MatrixXf reducedDatapoints = pca::transformPointMatrix(datamat, eigenVects);
    // FILE *fp = fopen("face.dat", "w");
    FILE *fp = fopen("flower.dat", "w");
    for (int i = 0; i < reducedDatapoints.rows(); ++i)
    {
        if(v_classtype[i])
            fprintf(fp, "+1 ");
        else
            fprintf(fp, "-1 ");
        for (int j = 0; j < reducedDatapoints.cols(); ++j)
            fprintf(fp, "%d:%f ", j+1, reducedDatapoints(i, j));
        fprintf(fp, "\n");
    }
    fclose(fp);
    return 0;
}