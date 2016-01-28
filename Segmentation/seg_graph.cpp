// Courtesy http://cs.brown.edu/~pff/segment/

#include <cstdio>
#include <cstdlib>
#include <string>
#include <image.h>
#include <misc.h>
#include <pnmfile.h>
#include "segment-image.h"

using namespace std;

void printHelp(const char* str){
    fprintf(stderr, "Usage: %s <FILENAME> [-chkmsr]\n", str);
    fprintf(stderr, "-c:\tcoalesce small clusters bounded by large clusters\n");
    fprintf(stderr, "-h:\tdisplay help\n");
    fprintf(stderr, "-k:\tconstant for threshold function [default 500]\n");
    fprintf(stderr, "-m:\tminimum component size, enforced by post processing stage [default 15]\n");
    fprintf(stderr, "-s:\tconstant required for the image smoothing stage [default 0.5]\n");
    fprintf(stderr, "-r:\t provide COALESCE_RATIO [default 0.02]\n");
}

int main(int argc, char **argv) {
    bool merge = false;
    float coalesce_ratio = 0.02;
    float sigma = 0.5;
    float k = 500;
    int min_size = 15;
    int c;
    extern char *optarg;
    extern int optind;

    if(argc < 2){
        fprintf(stderr, "No input file provided.\n");
        printHelp(argv[0]);
        return -1;
    }

    while((c = getopt(argc, argv, "chk:m:s:r:")) != -1){
        switch(c) {
            case 'c':
                merge = true;
                break;
            case 'h':
                printHelp(argv[0]);
                return 0;
            case 'k':
                k = atoi(optarg);
                break;
            case 'm':
                min_size = atoi(optarg);
                break;
            case 's':
                sigma = atof(optarg);
                break;
            case 'r':
                coalesce_ratio = atof(optarg);
                break;
            default:
                printHelp(argv[0]);
                return -1;
        }
    }

    printf("loading input image.\n");
    image<rgb> *input = loadOpenCV(argv[argc - 1]);

    printf("processing\n");
    int num_ccs; 
    image<rgb> *seg = segment_image(input, merge, coalesce_ratio, sigma, k, min_size, &num_ccs);
    cvNamedWindow("in", CV_WINDOW_AUTOSIZE);
    cvNamedWindow("out", CV_WINDOW_AUTOSIZE);
    cvMoveWindow("in", 50, 100);
    cvMoveWindow("out", 100 + input->width(), 100);
    displayOpenCV(input, "in");
    displayOpenCV(seg, "out");
    int ch = cvWaitKey();
    string str(argv[argc - 1]);
    str.insert(str.find_last_of("."), "_seg");
    if(ch == 's' || ch == 'S')
        saveJPG(seg, str.c_str());

    printf("got %d components\n", num_ccs);
    return 0;
}

