#ifndef TP2_METODOS_METRICS_H
#define TP2_METODOS_METRICS_H

#include <cassert>
#include "Matrix.h"
#include <sstream>
#include <iterator>

string vecOfDoublesToString(std::vector<double> vec);
std::vector<double> matrixToVec(const Matrix &m);
std::tuple<double, std::vector<double>, std::vector<double>>
    allMetricsWrapper(const Matrix &groundTruth, const Matrix &estimation);
double mean(std::vector<double> series);
double recallPerLabel(const Matrix &groundTruth, const Matrix &estimation, const int person);
double precisionPerLabel(const Matrix &groundTruth, const Matrix &estimation, const int person);
double accuracy(const Matrix &groundTruth, const Matrix &estimation);

#endif //TP2_METODOS_METRICS_H
