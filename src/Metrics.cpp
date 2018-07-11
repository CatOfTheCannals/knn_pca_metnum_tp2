#include "Metrics.h"


string vecOfDoublesToString(std::vector<double> vec) {
    std::ostringstream oss;

    if (!vec.empty())
    {
        oss << "\"[";
        // Convert all but the last element to avoid a trailing ","
        std::copy(vec.begin(), vec.end()-1,
                  std::ostream_iterator<double>(oss, ","));

        // Now add the last element with no delimiter
        oss << vec.back();
        oss << "]\"";
    }

    return oss.str() ;
}

std::tuple<double, std::vector<double>, std::vector<double>>
allMetricsWrapper(const Matrix &groundTruth, const Matrix &estimation) {
    assert(groundTruth.rows() == estimation.rows());
    assert(groundTruth.cols() == 1 && estimation.cols() == 1);

    unsigned long amount_of_persons = 41;
    std::vector<double> recallAtK(amount_of_persons);
    std::vector<double> precisionAtK(amount_of_persons);

    std::vector<double> vecGroundTruth(amount_of_persons);
    std::vector<double> vecEstimation(amount_of_persons);

    for(int i = 1; i < amount_of_persons; i++) {
        recallAtK[i] = recallPerPerson(groundTruth, estimation, i);
        precisionAtK[i] = precisionPerPerson(groundTruth, estimation, i);

        vecGroundTruth[i] = groundTruth(i);
        vecEstimation[i] = estimation(i);
    }

    double acc = accuracy(groundTruth, estimation);

    std::cout << "accuracy: " << acc << " | recall: " << mean(recallAtK)
              << " | precision: " << mean(precisionAtK) << std::endl;

    return std::make_tuple(acc, vecGroundTruth, vecEstimation);
}

double mean(std::vector<double> series) {
    double sum = 0;
    double amount_of_defined_values = 0;
    for(auto it = series.begin(); it != series.end(); it++ ) {
        if(!isnan(*it)) {
            sum += *it;
            amount_of_defined_values ++;
        }
    }
    return sum / amount_of_defined_values;
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