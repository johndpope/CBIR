/*
* @Author: pkar
* @Date:   2016-02-22 18:40:19
* @Last Modified by:   pkar
* @Last Modified time: 2016-02-22 19:55:43
*/

#include "svmPredictVector.h"

static int (*info)(const char *fmt,...) = &printf;

// Only C_SVC SVM supported
int predictVector(vector<float> &input, const char* model_file)
{
    struct svm_model* model = svm_load_model(model_file);
    int max_nr_attr = input.size() + 1;
    struct svm_node *x = (struct svm_node *) malloc(max_nr_attr * sizeof(struct svm_node));
    int svm_type = svm_get_svm_type(model);
    int nr_class = svm_get_nr_class(model);
    double *prob_estimates=NULL;
    for (int i = 0; i < input.size(); ++i)
    {
        x[i].index = i + 1;
        x[i].value = input[i];
    }
    x[input.size()].index = -1;
    int predict_label = svm_predict(model, x);
    svm_free_and_destroy_model(&model);
    free(x);
    return predict_label;
}

int predictVectorModel(vector<float> &input, struct svm_model* model)
{
    int max_nr_attr = input.size() + 1;
    struct svm_node *x = (struct svm_node *) malloc(max_nr_attr * sizeof(struct svm_node));
    int svm_type = svm_get_svm_type(model);
    int nr_class = svm_get_nr_class(model);
    double *prob_estimates=NULL;
    for (int i = 0; i < input.size(); ++i)
    {
        x[i].index = i + 1;
        x[i].value = input[i];
    }
    x[input.size()].index = -1;
    int predict_label = svm_predict(model, x);
    free(x);
    return predict_label;
}