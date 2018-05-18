#ifndef TP1_METODOS_KNN_H
#define TP1_METODOS_KNN_H

#include <cassert>
#include <chrono>
#include "Matrix.hpp"

Matrix distance(const Matrix& input, const Matrix& image);
int mostAppears(const int array[], const int length);//fixme: check the parameters

struct int_pair{
    int key;
    int value;
};


#endif //TP1_METODOS_KKN_H
