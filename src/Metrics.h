#ifndef TP2_METODOS_METRICS_H
#define TP2_METODOS_METRICS_H

#include <cassert>
#include "Matrix.h"

std::tuple<double, double, double> allMetricsAveraged(const Matrix &groundTruth, const Matrix &estimation);
double recallPerPerson(const Matrix &groundTruth, const Matrix &estimation, const int person);
double precisionPerPerson(const Matrix &groundTruth, const Matrix &estimation, const int person);
double accuracy(const Matrix &groundTruth, const Matrix &estimation);

#endif //TP2_METODOS_METRICS_H
