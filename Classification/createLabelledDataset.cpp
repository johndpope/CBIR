/*
* @Author: pkar
* @Date:   2016-02-18 18:08:52
* @Last Modified by:   Pratyush Kar
* @Last Modified time: 2016-03-21 17:02:12
*/

#include <iostream>
#include <vector>
#include <cstdio>
#include <algorithm>
#include <string>
//OpenCV Headers
#include <opencv/cv.h>
#include <opencv/highgui.h>

using namespace std;
using namespace cv;

#define MAX_RAW_IMG_HEIGHT 600
#define MAX_RAW_IMG_WIDTH 1200
#define IMG_HEIGHT 64
#define IMG_WIDTH 64
#define BRIGHT_DIFF 50
#define SCALE_STEP 0.25

struct callbackData
{
    Mat frame;
    Mat frame_dec;
    char* windowname;
    double scale;
    Mat data_frame;
    string dataset_folder;
    int data_idx;
};

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

Mat getHighlightedImg(Mat frame, Rect myROI)
{
    Mat img;
    frame.copyTo(img);
    int lefttopx = myROI.x;
    int lefttopy = myROI.y;
    int rightbottomx = lefttopx + myROI.width;
    int rightbottomy = lefttopy + myROI.height;
    Mat roi = img(myROI);
    roi = roi + Scalar(BRIGHT_DIFF, BRIGHT_DIFF, BRIGHT_DIFF);
    rectangle(img, Point(lefttopx, lefttopy), Point(rightbottomx, rightbottomy), CV_RGB(255, 255, 255));
    return img;
}

void showBoundingRect(int event, int x, int y, int flag, void* userdata)
{
    callbackData* obj = (callbackData*) userdata;
    if(event == EVENT_LBUTTONDOWN)
    {
        int width = obj->frame.cols;
        int height = obj->frame.rows;
        int frame_width = (int) ((double)IMG_WIDTH * obj->scale);
        int frame_height = (int) ((double)IMG_HEIGHT * obj->scale);
        int lefttopx = x - frame_width/2;
        int lefttopy = y - frame_height/2;
        int rightbottomx = x + frame_width/2;
        int rightbottomy = y + frame_height/2;
        if(lefttopx < 0 || lefttopy < 0 || rightbottomx >= width || rightbottomy >= height)
        {
            imshow(obj->windowname, obj->frame);
            return;
        }
        Rect myROI(lefttopx, lefttopy, frame_width, frame_height);
        Mat roi = obj->frame(myROI);
        Size sz(IMG_WIDTH, IMG_HEIGHT);
        resize(roi, obj->data_frame, sz);
        char file_name[1000];
        sprintf(file_name, "%s/%d.jpg", obj->dataset_folder.c_str(), obj->data_idx);
        imwrite(file_name, obj->data_frame);
        obj->data_idx++;
    }
    else if(event == EVENT_RBUTTONDOWN)
    {
        int width = obj->frame.cols;
        int height = obj->frame.rows;
        int frame_width = (int) ((double)IMG_WIDTH * obj->scale);
        int frame_height = (int) ((double)IMG_HEIGHT * obj->scale);
        int lefttopx = x - frame_width/2;
        int lefttopy = y - frame_height/2;
        int rightbottomx = x + frame_width/2;
        int rightbottomy = y + frame_height/2;
        if(lefttopx < 0 || lefttopy < 0 || rightbottomx >= width || rightbottomy >= height)
        {
            imshow(obj->windowname, obj->frame);
            return;
        }
        Rect myROI(lefttopx, lefttopy, frame_width, frame_height);
        Mat roi = obj->frame(myROI);
        Size sz(IMG_WIDTH, IMG_HEIGHT);
        resize(roi, obj->data_frame, sz);
        cvNamedWindow("out");
        imshow("out", obj->data_frame);
    }
    else if(event == EVENT_MOUSEMOVE)
    {
        int width = obj->frame.cols;
        int height = obj->frame.rows;
        int frame_width = (int) ((double)IMG_WIDTH * obj->scale);
        int frame_height = (int) ((double)IMG_HEIGHT * obj->scale);
        int lefttopx = x - frame_width/2;
        int lefttopy = y - frame_height/2;
        int rightbottomx = x + frame_width/2;
        int rightbottomy = y + frame_height/2;
        if(lefttopx < 0 || lefttopy < 0 || rightbottomx >= width || rightbottomy >= height)
        {
            imshow(obj->windowname, obj->frame);
            return;
        }
        Rect myROI(lefttopx, lefttopy, frame_width, frame_height);
        Mat img = getHighlightedImg(obj->frame_dec, myROI);
        imshow(obj->windowname, img);
    }
}

int main(){
    string image_root_dir = "../Dataset/Original Images/";
    string file_names_list = "file_names.txt";
    string dataset_folder = "../Dataset/Classes/";
    vector<string> class_name_list;
    class_name_list.push_back("face");
    class_name_list.push_back("temp");
    class_name_list.push_back("flower");
    class_name_list.push_back("other");
    vector<string> img_file_path_vec = getImagePathNameVector(image_root_dir, file_names_list);
    if(img_file_path_vec.size() == 0)
    {
        printf("No images to load.\n");
        return 0;
    }
    int img_idx = 0;
    int cls_idx = 1;
    callbackData obj;
    obj.scale = 1.0;
    obj.data_idx = 1;
    while(1)
    {
        Mat orig_img = imread(img_file_path_vec[img_idx].c_str(), CV_LOAD_IMAGE_COLOR);
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
        char windowname[1000];
        sprintf(windowname, "Class Name: %s | Capturing image number: %d | Scale: %.2lf", class_name_list[cls_idx].c_str(), obj.data_idx, obj.scale);
        cvNamedWindow(windowname);
        cvMoveWindow(windowname, 20, 20);
        img.copyTo(obj.frame);
        img.copyTo(obj.frame_dec);
        obj.frame_dec = obj.frame_dec - Scalar(BRIGHT_DIFF, BRIGHT_DIFF, BRIGHT_DIFF);
        obj.windowname = windowname;
        obj.data_frame = Mat(IMG_WIDTH, IMG_HEIGHT, CV_8UC3);
        obj.dataset_folder = dataset_folder + class_name_list[cls_idx];
        setMouseCallback(windowname, showBoundingRect, &obj);
        imshow(windowname, img);
        int c = cvWaitKey();
        if(c == 27)
            break;
        else if(c == 'n' || c == 'N')
        {
            destroyAllWindows();
            img_idx = (img_idx + 1) % img_file_path_vec.size();
        }
        else if(c == 'p' || c == 'P')
        {
            destroyAllWindows();
            img_idx = (img_file_path_vec.size() + img_idx - 1) % img_file_path_vec.size();
        }
        else if(c == ']' || c == '=' || c == '+')
        {
            destroyAllWindows();
            obj.scale = obj.scale + SCALE_STEP;
        }
        else if(c == '[' || c == '-' || c == '_')
        {
            destroyAllWindows();
            obj.scale = obj.scale - SCALE_STEP;
        }
        else if(c == '0')
        {
            destroyAllWindows();
            obj.scale = 1.0;
        }
    }
    return 0;
}