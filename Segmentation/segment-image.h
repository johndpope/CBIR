#ifndef SEGMENT_IMAGE
#define SEGMENT_IMAGE

#include <cstdlib>
#include <image.h>
#include <misc.h>
#include <filter.h>
#include <vector>
#include "segment-graph.h"

using namespace std;

// random color
rgb random_rgb(){
  rgb c;
  double r;

  c.r = (uchar)random();
  c.g = (uchar)random();
  c.b = (uchar)random();

  return c;
}

struct clusterImage
{
  universe *u;
  image<rgb> *im;
};

// dissimilarity measure between pixels
static inline float diff(image<float> *r, image<float> *g, image<float> *b,
			 int x1, int y1, int x2, int y2) {
  return sqrt(square(imRef(r, x1, y1)-imRef(r, x2, y2)) +
	      square(imRef(g, x1, y1)-imRef(g, x2, y2)) +
	      square(imRef(b, x1, y1)-imRef(b, x2, y2)));
}

/*
* Coalesce segments
*
* Merges neighbouring clusters whose size ratio is
* is less than the coalesce_ratio
*
* u: input disjoint-set structure
* width: width of the image
* height: height of the image
* coalesce_ratio: coalesce_ratio measure used to decide merger
*/
void mergeSizeBased(universe *u, int width, int height, float coalesce_ratio) {
for (int y = 0; y < height; y++){
    for (int x = 0; x < width; x++){
      int elem = y * width + x;
      int surr = -1;
      int max_size = -1;
      if(x < width-1){
        int temp = y * width + (x + 1);
        int p = u->find(temp);
        if(max_size < u->size(p)){
          max_size = u->size(p);
          surr = temp;
        }
      }
      if(x > 0){
        int temp = y * width + (x - 1);
        int p = u->find(temp);
        if(max_size < u->size(p)){
          max_size = u->size(p);
          surr = temp;
        }
      }
      if(y < height-1){
        int temp = (y + 1) * width + x;
        int p = u->find(temp);
        if(max_size < u->size(p)){
          max_size = u->size(p);
          surr = temp;
        }
      }
      if(y > 0){
        int temp = (y - 1) * width + x;
        int p = u->find(temp);
        if(max_size < u->size(p)){
          max_size = u->size(p);
          surr = temp;
        }
      }
      if(surr == -1)
        continue;
      int a = u->find(elem);
      int b = u->find(surr);
      if(u->size(a) > u->size(b))
        swap(a, b);
      if(a != b && (float) u->size(a) < (float) (u->size(b) * coalesce_ratio))
        u->join(a, b);
    }
  }
}

Mat computeConvexHullMask(Mat img)
{
  vector<vector<Point> > contours;
  vector<Vec4i> hierarchy;
  blur(img, img, Size(3,3));
  threshold(img, img, 100, 255, THRESH_BINARY);
  findContours(img, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0,0));
  vector<vector<Point> > hull(contours.size());
  for (int i = 0; i < contours.size(); ++i)
    convexHull(Mat(contours[i]), hull[i], false);
  Mat mask = Mat::zeros(img.size(), CV_8UC1);
  if(contours.size() == 0)
    return mask;
  vector<Point> hull_contour;
  int max_hull_idx = 0;
  int max_hull_area = 0;
  for (int i = 0; i < contours.size(); ++i)
  {
    approxPolyDP(Mat(hull[i]), hull_contour, 0.001, true);
    int area = (int) contourArea(Mat(hull_contour));
    if(area > max_hull_area)
    {
      max_hull_idx = i;
      max_hull_area = area;
    }
  }
  drawContours(mask, hull, max_hull_idx, CV_RGB(255,255,255), 1, 8, vector<Vec4i>(), 0, Point());
  int centerx = 0;
  int centery = 0;
  for (int i = 0; i < hull[max_hull_idx].size(); ++i)
  {
    centerx += hull[max_hull_idx][i].x;
    centery += hull[max_hull_idx][i].y;
  }
  centerx /= hull[max_hull_idx].size();
  centery /= hull[max_hull_idx].size();
  floodFill(mask, Point(centerx, centery), CV_RGB(40,40,40));
  imshow("mask", mask);
  return mask;
}

void showClassforSegment(int event, int x, int y, int flags, void* cluster_img)
{
  if(flags != EVENT_MOUSEMOVE && flags != EVENT_LBUTTONDOWN)
    return;
  clusterImage* cimg = (clusterImage*) cluster_img;
  universe* u = cimg->u;
  image<rgb>* im = cimg->im;
  Mat img = getMat(im);
  Mat cluster_mask(img.rows, img.cols, CV_8UC1, Scalar(0));
  int width = img.cols;
  int height = img.rows;
  int pidx = u->find(y * width + x);
  for (int i = 0; i < width; ++i)
  {
    for (int j = 0; j < height; ++j)
    {
      if(pidx == u->find(j * width + i))
        cluster_mask.at<uchar>(Point2i(i,j)) = 255;
    }
  }
  Mat mask = computeConvexHullMask(cluster_mask);
  Mat rgb_mask = Mat::zeros(mask.size(), CV_8UC3);
  cvtColor(mask, rgb_mask, CV_GRAY2RGB);
  img = img - Scalar(25, 25, 25);
  add(img, rgb_mask, img);
  imshow("img", img);
}

// void mergeColorBased(universe *u, int width, int height, float color_similarity) {

// }

/*
* Segment an image
*
* Returns a color image representing the segmentation.
*
* im: image to segment.
* sigma: to smooth the image.
* c: constant for threshold function.
* min_size: minimum component size (enforced by post-processing stage).
* num_ccs: number of connected components in the segmentation.
*/
image<rgb> *segment_image(image<rgb> *im, bool merge, float coalesce_ratio, float sigma, float c, int min_size,
			  int *num_ccs) {
  int width = im->width();
  int height = im->height();

  image<float> *r = new image<float>(width, height);
  image<float> *g = new image<float>(width, height);
  image<float> *b = new image<float>(width, height);

  // smooth each color channel
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      imRef(r, x, y) = imRef(im, x, y).r;
      imRef(g, x, y) = imRef(im, x, y).g;
      imRef(b, x, y) = imRef(im, x, y).b;
    }
  }
  image<float> *smooth_r = smooth(r, sigma);
  image<float> *smooth_g = smooth(g, sigma);
  image<float> *smooth_b = smooth(b, sigma);
  delete r;
  delete g;
  delete b;

  // build graph
  edge *edges = new edge[width*height*4];
  // store the color information (required for merge)
  vector<rgb> colorVect(width*height);
  int num = 0;
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = y * width + x;
      colorVect[idx].r = imRef(smooth_r, x, y);
      colorVect[idx].g = imRef(smooth_g, x, y);
      colorVect[idx].b = imRef(smooth_b, x, y);
      if (x < width-1) {
	edges[num].a = y * width + x;
	edges[num].b = y * width + (x+1);
	edges[num].w = diff(smooth_r, smooth_g, smooth_b, x, y, x+1, y);
	num++;
      }

      if (y < height-1) {
	edges[num].a = y * width + x;
	edges[num].b = (y+1) * width + x;
	edges[num].w = diff(smooth_r, smooth_g, smooth_b, x, y, x, y+1);
	num++;
      }

      if ((x < width-1) && (y < height-1)) {
	edges[num].a = y * width + x;
	edges[num].b = (y+1) * width + (x+1);
	edges[num].w = diff(smooth_r, smooth_g, smooth_b, x, y, x+1, y+1);
	num++;
      }

      if ((x < width-1) && (y > 0)) {
	edges[num].a = y * width + x;
	edges[num].b = (y-1) * width + (x+1);
	edges[num].w = diff(smooth_r, smooth_g, smooth_b, x, y, x+1, y-1);
	num++;
      }
    }
  }
  delete smooth_r;
  delete smooth_g;
  delete smooth_b;

  // segment
  universe *u = segment_graph(width*height, num, edges, colorVect, c);

  // post process small components
  for (int i = 0; i < num; i++) {
    int a = u->find(edges[i].a);
    int b = u->find(edges[i].b);
    if ((a != b) && ((u->size(a) < min_size) || (u->size(b) < min_size)))
      u->join(a, b);
  }

  // merging small components into adjacent large components subject to coalesce_ratio
  if(merge) {
    mergeSizeBased(u, width, height, coalesce_ratio);
  }

  delete [] edges;
  *num_ccs = u->num_sets();

  image<rgb> *output = new image<rgb>(width, height);

  // pick random colors for each component
  rgb *colors = new rgb[width*height];
  for (int i = 0; i < width*height; i++)
    colors[i] = random_rgb();

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int comp = u->find(y * width + x);
      imRef(output, x, y) = colors[comp];
    }
  }
  clusterImage cimg;
  cimg.u = u;
  cimg.im = im;
  displayOpenCV(output, "out");
  setMouseCallback("out", showClassforSegment, &cimg);
  setMouseCallback("img", showClassforSegment, &cimg);
  waitKey();
  delete [] colors;
  delete u;

  return output;
}

#endif
