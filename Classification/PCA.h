#ifndef PCA_H
#define PCA_H

#include <iostream>
#include <cstdio>
#include <fstream>
#include <vector>
#include <eigen3/Eigen/Dense>

namespace pca
{
    class PCA {
    private:
        Eigen::MatrixXf data;
        Eigen::VectorXf eigenValues;
        Eigen::MatrixXf eigenVectors;
        Eigen::VectorXf convertedOrigin;
    public:
        PCA (const Eigen::MatrixXf& data, bool normalizeEachData = false, bool normalizeOrigin = false);
        Eigen::MatrixXf getEigenVectors() const
        {
            return eigenVectors;
        }
        Eigen::VectorXf getEigenValues() const
        {
            return eigenValues;
        }
        Eigen::VectorXf getConvertedOrigin() const
        {
            return convertedOrigin;
        }
        void showParameters(int eigen_cnt);
        void showVariance(int eigen_cnt);
        Eigen::MatrixXf getEigenVectorsCnt(int eigen_cnt);
    };
    Eigen::MatrixXf transformPointMatrix(Eigen::MatrixXf data, Eigen::MatrixXf eigenvects);
}
#endif
