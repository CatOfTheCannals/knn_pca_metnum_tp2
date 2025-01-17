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

std::vector<double> matrixToVec(const Matrix &m) {
    assert(m.cols() == 1);
    std::vector<double> v(m.rows());
    for(int i = 0; i < m.rows(); i++) {
        v[i] = m(i);
    }
    return v;
}

std::tuple<double, std::vector<double>, std::vector<double>>
allMetricsWrapper(const Matrix &groundTruth, const Matrix &estimation) {

    assert(groundTruth.rows() == estimation.rows());
    assert(groundTruth.cols() == 1 && estimation.cols() == 1);

    unsigned long amount_of_labels = 2;
    std::vector<double> recallAtK(amount_of_labels);
    std::vector<double> precisionAtK(amount_of_labels);

    std::vector<double> vecGroundTruth = matrixToVec(groundTruth);
    std::vector<double> vecEstimation = matrixToVec(estimation);

    for(int i = 0; i < amount_of_labels; i++) {
        recallAtK[i] = recallPerLabel(groundTruth, estimation, i);
        precisionAtK[i] = precisionPerLabel(groundTruth, estimation, i);

        std::cout << "recall at " << i << ":"  << recallAtK[i] << std::endl;
        std::cout << "precision at " << i << ":"  << precisionAtK[i] << std::endl;

    }

    double acc = accuracy(groundTruth, estimation);

    std::cout << "accuracy: " << acc << " | mean recall: " << mean(recallAtK)
              << " | mean precision: " << mean(precisionAtK) << std::endl;

    return std::make_tuple(acc, recallAtK, precisionAtK);
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

double recallPerLabel(const Matrix &groundTruth, const Matrix &estimation, const int person) {
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

double precisionPerLabel(const Matrix &groundTruth, const Matrix &estimation, const int person) {
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