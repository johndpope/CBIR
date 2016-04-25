#include "PCA.h"

using namespace pca;
using namespace std;

PCA::PCA(const Eigen::MatrixXf& data, bool normalizeEachData, bool normalizeOrigin)
:data(data), convertedOrigin(Eigen::VectorXf::Zero(data.cols()))
{
    const int DIM = (int)data.cols();

    Eigen::MatrixXf W = data;
    Eigen::MatrixXf covarianceMatrix = data;

    if (normalizeEachData) {
        Eigen::VectorXf mean = data.rowwise().mean();
        W = W.colwise() - mean;
    }
    if (normalizeOrigin) {
        convertedOrigin = data.colwise().mean();
        W = W.rowwise() - convertedOrigin.transpose();
    }
    cout<<"Number of features in each datapoint: "<<data.cols()<<"\n";
    cout<<"Number of data points: "<<data.rows()<<"\n";
    cout<<"Computing covariance matrix...";
    cout.flush();
    covarianceMatrix = W.transpose() * W;
    cout<<" Done\n";
    cout<<"Normalizing covariance matrix...";
    cout.flush();
    covarianceMatrix.normalize();
    cout<<" Done\n";
    //calculate eigens
    cout<<"PCA Solver running...";
    cout.flush();
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> solver(covarianceMatrix);
    cout<<" Done\n";
    cout<<"Computing Eigen Values...";
    cout.flush();
    Eigen::VectorXf eigenValues = solver.eigenvalues();
    cout<<" Done\n";
    cout<<"Computing Eigen Vectors...";
    cout.flush();
    Eigen::MatrixXf eigenVectors = solver.eigenvectors();
    cout<<" Done\n";
    //make eigenvalues and eigen matrix
    this->eigenValues = Eigen::VectorXf(DIM);
    this->eigenVectors = Eigen::MatrixXf(DIM, DIM);
    for(int i=0; i<DIM; i++) {
        int r = DIM-i-1;
        this->eigenValues(i) = eigenValues(r);
        this->eigenVectors.col(i) = eigenVectors.col(r);
    }
}

Eigen::MatrixXf PCA::getEigenVectorsCnt(int eigen_cnt)
{
    Eigen::MatrixXf mat = this->eigenVectors.block(0, 0, this->data.cols(), eigen_cnt);
    // cout<<"Eigen Vect: "<<endl;
    // cout<<mat<<endl;
    ofstream fp;
    fp.open("eigen.txt");
    fp << mat.rows() << " " << mat.cols() << endl;
    fp << mat << endl;
    fp.close();
    return mat;
}

void PCA::showVariance(int eigen_cnt)
{
    double variance = 0.0;
    double total = 0.0;
    for (int i = 0; i < eigen_cnt; ++i)
        variance += this->eigenValues(i);
    for (int i = 0; i < this->eigenValues.rows(); ++i)
        total += this->eigenValues(i);
    cout<<"Total Captured Variance is: "<<(variance/total) * 100<<"%"<<endl;
}

void PCA::showParameters(int eigen_cnt)
{
    double variance = 0.0;
    double total = 0.0;
    for (int i = 0; i < eigen_cnt; ++i)
        variance += this->eigenValues(i);
    for (int i = 0; i < this->eigenValues.rows(); ++i)
        total += this->eigenValues(i);
    cout<<"Total Captured Variance is: "<<(variance/total) * 100<<"%"<<endl;
    std::cout << "eigen values : " << std::endl;
    for (int i = 0; i < eigen_cnt; ++i)
        cout<<this->eigenValues(i)<<endl;

    std::cout << "eigen vectors : " << std::endl;
    cout<<"cols: "<<this->eigenVectors.cols()<<"\trows: "<<this->eigenVectors.rows()<<endl;
    for (int i = 0; i < eigen_cnt; ++i)
        cout<<this->eigenVectors.col(i)<<endl<<endl;

    // std::cout << "Origin : " << std::endl
    // << pca.getConvertedOrigin() << std::endl << std::endl;
}

Eigen::MatrixXf pca::transformPointMatrix(Eigen::MatrixXf data, Eigen::MatrixXf eigenvects)
{
    Eigen::MatrixXf transpoint = data*eigenvects;
    return transpoint;
}