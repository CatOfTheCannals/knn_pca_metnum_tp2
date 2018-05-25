#include "Metrics.h"

double recallPerPerson(const Matrix &groundTruth, const Matrix &estimation, const int person) {
    assert(groundTruth.rows() == estimation.rows());
    assert(groundTruth.cols() == 1 && estimation.cols() == 1);
    double amountOfGoodPredictions = 0; // tp
    double personApparitions = 0; // tp + fn

    for(int i = 0; i < groundTruth.rows(); i++) {
        if(groundTruth(i) == person) {
            std::cout << "groundTruth: " << groundTruth(i) << ", estimation: " << estimation(i) << std::endl;
            personApparitions ++;
            if(estimation(i) == person) amountOfGoodPredictions++;
        }
    }
    std::cout << "amountOfGoodPredictions: " << amountOfGoodPredictions  << ", personApparitions: " << personApparitions << std::endl;
    return amountOfGoodPredictions / personApparitions;
}

double precisionPerPerson(const Matrix &groundTruth, const Matrix &estimation, const int person) {
    assert(groundTruth.rows() == estimation.rows());
    assert(groundTruth.cols() == 1 && estimation.cols() == 1);
    double amountOfGoodPredictions = 0; // tp
    double personApparitions = 0; // tp + fp
    for(int i = 0; i < groundTruth.rows(); i++) {
        if(estimation(i) == person) {
            personApparitions ++;
            if(groundTruth(i) == person) amountOfGoodPredictions ++;
        }
    }
    return amountOfGoodPredictions / personApparitions;
}

double accuracy(const Matrix &groundTruth, const Matrix &estimation) {
    assert(groundTruth.rows() == estimation.rows());
    assert(groundTruth.cols() == 1 && estimation.cols() == 1);
    double amountOfGoodPredictions = 0;
    for(int i = 0; i < groundTruth.rows(); i++) {
        if(groundTruth(i) == estimation(i)) amountOfGoodPredictions++;
    }
    return amountOfGoodPredictions / (double)groundTruth.rows();
}