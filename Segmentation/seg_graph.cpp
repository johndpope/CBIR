// Courtesy http://cs.brown.edu/~pff/segment/

#include <cstdio>
#include <cstdlib>
#include <string>
#include <image.h>
#include <misc.h>
#include <pnmfile.h>
#include "segment-image.h"

using namespace std;

int main(int argc, char **argv) {
    float sigma = 0.5;
    float k = 500;
    int min_size = 20;
    if(argc < 2) {
        fprintf(stderr, "usage: %s <FILENAME> [sigma k min]\n", argv[0]);
        return 1;
    }
    if(argc > 2)
        sigma = atof(argv[2]);
    if(argc > 3)
        k = atof(argv[3]);
    if(argc > 4)
        min_size = atoi(argv[4]);
    if(argc > 5){
        fprintf(stderr, "usage: %s <FILENAME> [sigma k min]\n", argv[0]);
        return -1;
    }

    printf("loading input image.\n");
    image<rgb> *input = loadOpenCV(argv[1]);

    printf("processing\n");
    int num_ccs; 
    image<rgb> *seg = segment_image(input, sigma, k, min_size, &num_ccs); 
    displayOpenCV(input, "in");
    displayOpenCV(seg, "out");
    int c = cvWaitKey();
    string str(argv[1]);
    str.insert(str.find_last_of("."), "_seg");
    if(c == 's' || c == 'S')
        saveJPG(seg, str.c_str());

    printf("got %d components\n", num_ccs);
    return 0;
}

