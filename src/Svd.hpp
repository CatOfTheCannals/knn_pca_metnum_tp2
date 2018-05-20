#ifndef TP1_METODOS_SVD_H
#define TP1_METODOS_SVD_H

#include <cassert>
#include <chrono>
#include "Matrix.hpp"

Matrix pca(const Matrix &A, unsigned int num_components, double epsilon);

tuple<Matrix, double> power_method(Matrix& x_0, Matrix& input, double epsilon);

tuple<Matrix, Matrix> svd(const Matrix &A, unsigned int num_components, double epsilon);

Matrix ones(int rows, int cols);

#endif //TP1_METODOS_SVD_H
