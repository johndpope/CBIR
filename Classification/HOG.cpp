/*
* @Author: pkar
* @Date:   2016-03-03 02:03:27
* @Last Modified by:   pkar
* @Last Modified time: 2016-03-03 02:03:27
*/

#include "HOG.h"

HOG::HOG(Mat frame)
{
    frame.copyTo(img);
    img_size = frame.size();
    win_size = Size(64,64);
    win_stride = Size(8,8);
    block_size = Size(16,16);
    block_stride = Size(8,8);
    cell_size = Size(8,8);
    nbins = 9;
    descriptors.clear();
    locations.clear();
    hog = HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins);
    hog.compute(img, descriptors, win_stride, Size(0,0), locations);
}

HOG::HOG(Mat frame, Size win_size, Size win_stride, Size block_size, Size block_stride, Size cell_size, int nbins)
{
    frame.copyTo(img);
    img_size = frame.size();
    this->win_size = win_size;
    this->win_stride = win_stride;
    this->block_size = block_size;
    this->block_stride = block_stride;
    this->cell_size = cell_size;
    this->nbins = nbins;
    descriptors.clear();
    locations.clear();
    hog = HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins);
    hog.compute(img, descriptors, win_stride, Size(0,0), locations);
}

int HOG::getDescriptorSize()
{
    return (int) nbins * (block_size.width/cell_size.width) * (block_size.height/cell_size.height) * ((win_size.width - block_size.width)/block_stride.width + 1) * ((win_size.height - block_size.height)/block_stride.height + 1);
}

// padding is not supported
int HOG::windowsInImage()
{
    Size winsz = Size((img_size.width - win_size.width)/win_stride.width + 1, (img_size.height - win_size.height)/win_stride.height + 1);
    return winsz.area();
}

Rect HOG::getWindowRect(int idx)
{
    int nwindowsX = (img_size.width - win_size.width)/win_stride.width + 1;
    int y = idx / nwindowsX;
    int x = idx - nwindowsX*y;
    return Rect(x*win_stride.width, y*win_stride.height, win_size.width, win_size.height);
}

vector<float> HOG::getWindowDescriptor(int idx)
{
    int dsize = getDescriptorSize();
    vector<float> d_vect(descriptors.begin() + idx*dsize, descriptors.begin() + (idx + 1)*dsize);
    return d_vect;
}

vector<vector<float> > HOG::getAllWindowDescriptors()
{
    int dsize = getDescriptorSize();
    int nwindows = windowsInImage();
    vector<vector<float> > d_vects(nwindows, vector<float>(dsize));
    for (int i = 0; i < nwindows; ++i)
        for (int j = 0; j < dsize; ++j)
            d_vects[i][j] = descriptors[i*dsize + j];
    return d_vects;
}
