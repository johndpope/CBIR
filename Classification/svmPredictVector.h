/*
* @Author: pkar
* @Date:   2016-02-22 17:45:43
* @Last Modified by:   pkar
* @Last Modified time: 2016-02-22 18:10:11
*/

#ifndef _SVM_VECTOR_PREDICT_H
#define _SVM_VECTOR_PREDICT_H

#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <vector>
#include "svm.h"

using namespace std;

int predictVector(vector<float> &input, const char* model_file);
int predictVectorModel(vector<float> &input, struct svm_model* model);

#endif