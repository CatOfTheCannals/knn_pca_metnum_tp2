#include "Metrics.h"


std::tuple<double, double, double> allMetricsAveraged(const Matrix &groundTruth, const Matrix &estimation) {
    assert(groundTruth.rows() == estimation.rows());
    assert(groundTruth.cols() == 1 && estimation.cols() == 1);
    double amount_of_defined_rpp = 0;
    double amount_of_defined_ppp = 0;
    double rpp_sum = 0;
    double ppp_sum = 0;
    for(int i = 1; i < 41; i++) {
        auto rpp = recallPerPerson(groundTruth, estimation, i);
        if(!isnan(rpp)) {
            rpp_sum += rpp;
            amount_of_defined_rpp ++;
        }
        auto ppp = precisionPerPerson(groundTruth, estimation, i);
        if(!isnan(ppp)) {
            ppp_sum += ppp;
            amount_of_defined_ppp ++;
        }
    }
    return std::make_tuple(accuracy(groundTruth, estimation), rpp_sum / amount_of_defined_rpp, ppp_sum / amount_of_defined_ppp);
}

double recallPerPerson(const Matrix &groundTruth, const Matrix &estimation, const int person) {
    assert(groundTruth.rows() == estimation.rows());
    assert(groundTruth.cols() == 1 && estimation.cols() == 1);
    double amountOfGoodPredictions = 0; // tp
    double personApparitions = 0; // tp + fn

    for(int i = 0; i < groundTruth.rows(); i++) {
        if(groundTruth(i) == person) {
            // std::cout << "groundTruth: " << groundTruth(i) << ", estimation: " << estimation(i) << std::endl;
            personApparitions ++;
            if(estimation(i) == person) amountOfGoodPredictions++;
        }
    }
    // std::cout << "amountOfGoodPredictions: " << amountOfGoodPredictions  << ", personApparitions: " << personApparitions << std::endl;
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
            if(groundTruth(i) == person) {
                amountOfGoodPredictions++;
                // std::cout << "groundTruth: " << groundTruth(i) << ", estimation: " << estimation(i) << std::endl;
            }
        }
    }
    // std::cout << "amountOfGoodPredictions: " << amountOfGoodPredictions  << ", personApparitions: " << personApparitions << std::endl;
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