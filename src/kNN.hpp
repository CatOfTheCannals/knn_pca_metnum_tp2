#ifndef TP1_METODOS_KNN_H
#define TP1_METODOS_KNN_H

#include <cassert>
#include <chrono>
#include "Matrix.hpp"

Matrix distance(const Matrix& input, const Matrix& image);
int mostAppears(const vector<int> repetitions);//fixme: check the parameters
int kNN(const Matrix& data, const Matrix& image, int k, const int numberOfPeople, const int numberOfPicturesPerPeople);


struct int_pair{
    int key;
    int value;
};


#endif //TP1_METODOS_KKN_H
