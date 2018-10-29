#ifndef TP1_METODOS_SVD_H
#define TP1_METODOS_SVD_H

#include <cassert>
#include <chrono>
#include "Matrix.h"

#define GET_TIME std::chrono::high_resolution_clock::now()
#define     GET_TIME_DELTA(begin, end) \
     std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()

tuple<Matrix, Matrix> pca(const Matrix &A, unsigned int num_components, double epsilon);

tuple<Matrix, double> power_method(Matrix& x_0, Matrix& input, double epsilon);
tuple<Matrix, double> power_method(Matrix& x_0, Matrix& input, double epsilon, int max_iters);

//pair<double, Matrix> powerMethodQ1(long N,const Matrix &a )
tuple<Matrix, Matrix> svd(const Matrix &A, unsigned int num_components, double epsilon);

Matrix ones(int rows, int cols);

Matrix random(int rows, int cols);

#endif //TP1_METODOS_SVD_H
