/*
* @Author: pkar
* @Date:   2016-02-28 16:34:31
* @Last Modified by:   pkar
* @Last Modified time: 2016-02-28 17:50:39
*/

#include <iostream>
#include "PCA.h"
#include <vector>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Eigen>

using namespace std;
using namespace Eigen;
using namespace pca;

int main(int argc, char const *argv[])
{
    unsigned int m = 100;  // number of points
    unsigned int n = 5;   // dimension of each point
    MatrixXd DataPoints = MatrixXd::Random(m, n);
    PCA pca(DataPoints);
    pca.showParameters(3);
    pca.getEigenVectorsCnt(3);
    return 0;
}