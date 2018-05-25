#ifndef TP1_METODOS_KNN_H
#define TP1_METODOS_KNN_H

#include <cassert>
#include <chrono>
#include "Matrix.h"



Matrix distance(const Matrix& input, const Matrix& image);
int mostAppears(const vector<int> repetitions);
int kNN(const Matrix& data, const Matrix& index, const Matrix& image, int k);

struct int_pair{
    int key;
    int value;
};


#endif //TP1_METODOS_KKN_H
