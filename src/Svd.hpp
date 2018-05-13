#ifndef TP1_METODOS_SVD_H
#define TP1_METODOS_SVD_H

#include <cassert>
#include <chrono>
#include "matrix.hpp"

tuple<Matrix, double> power_method(Matrix& x_0, Matrix& input,
                                   double epsilon);

#endif //TP1_METODOS_SVD_H
