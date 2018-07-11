#include "Dataset.h"

Matrix Dataset::getTestImages() const{
    Matrix output(_testImages);
    return output;
}

Matrix Dataset::getTestLabels() const{
    Matrix output(_testLabels);
    return output;
}

Matrix Dataset::getTrainImages() const{
    Matrix output(_trainImages);
    return output;
}

Matrix Dataset::getTrainLabels() const{
    Matrix output(_trainLabels);
    return output;
}

Matrix Dataset::getPcaVecs() const {
    Matrix output(_pcaVecs);
    return output;
}
Matrix Dataset::getPcaLambdas() const {
    Matrix output(_pcaLambdas);
    return output;
}

void Dataset::shuffle() {
    int n = _trainImages.rows();
    srand (time(NULL));
    for(int i = 0; i < n; i++) {
        int swap_index = rand() % (n - i) ;
        _trainImages.swapRows(i, i + swap_index);
        _trainLabels.swapRows(i, i + swap_index);
    }
}

void Dataset::splitTrainFromTest(double testPercentage) {
    int testRows = _trainImages.rows() * testPercentage;
    _testImages = _trainImages.subMatrix(0, testRows, 0, _trainImages.cols() - 1);
    _testLabels = _trainLabels.subMatrix(0, testRows, 0, _trainLabels.cols() - 1);
    _trainImages = _trainImages.subMatrix(testRows + 1, _trainImages.rows() - 1, 0, _trainImages.cols() - 1);
    _trainLabels = _trainLabels.subMatrix(testRows + 1, _trainLabels.rows() - 1, 0, _trainLabels.cols() - 1);
}


void Dataset::trainPca(int alpha, double epsilon) {
    assert(_trainImages.rows() > 0);
    auto pca_eigenvectors_and_eigenvalues = pca(_trainImages, alpha, epsilon);
    _pcaVecs = std::get<0>(pca_eigenvectors_and_eigenvalues);
    _pcaLambdas = std::get<1>(pca_eigenvectors_and_eigenvalues);
    _transformedTrainImages = _trainImages.multiply(_pcaVecs);
    _transformedTestImages = _testImages.multiply(_pcaVecs);
}


Matrix Dataset::pca_kNN_predict(int k, double epsilon) const {
    Matrix testLabels = Matrix(_transformedTestImages.rows(), 1);

    for(int i = 0; i < _transformedTestImages.rows(); i++) {
        int ith_label = kNN(_transformedTrainImages, _trainLabels, _transformedTestImages.getRow(i), k);
        testLabels.setIndex(i ,0 , ith_label);
    }

    return testLabels;
}


Matrix Dataset::kNN_predict(int k) const {
    Matrix testLabels = Matrix(_testImages.rows(), 1);
    for(int i = 0; i < _testImages.rows(); i++) {
        int ith_label = kNN(_trainImages, _trainLabels, _testImages.getRow(i), k);
        testLabels.setIndex(i ,0 , ith_label);
    }
    return testLabels;
}
std::vector<std::tuple<double, std::vector<double>, std::vector<double>>>
Dataset::knnEquitativeSamplingKFold(int neighbours)  {

    int amount_of_people = 41;
    int amount_of_picks = _trainImages.rows();
    int picks_per_person = amount_of_picks / amount_of_people;
    std::vector<std::tuple<double, std::vector<double>, std::vector<double>>>
        scores_per_fold;

    shuffleSamePersonPicks(amount_of_people, picks_per_person);
    for(int k = 0; k < picks_per_person; k++) {
        auto imageFold = getEquitativeSamplingFold(_trainImages, k, amount_of_people, picks_per_person);
        auto labelFold = getEquitativeSamplingFold(_trainLabels, k, amount_of_people, picks_per_person);

        Dataset d = Dataset(std::get<0>(imageFold), std::get<0>(labelFold),
                            std::get<1>(imageFold), std::get<1>(labelFold));
        scores_per_fold.push_back(allMetricsWrapper(std::get<1>(labelFold), d.kNN_predict(neighbours)));

    }

    return scores_per_fold;
}
std::vector<std::tuple<double, std::vector<double>, std::vector<double>>>
Dataset::pcaKnnEquitativeSamplingKFold(int neighbours, int alpha) {

    double epsilon = 0.0001;
    int amount_of_people = 41;
    int amount_of_picks = _trainImages.rows();
    int picks_per_person = amount_of_picks / amount_of_people;
    std::vector<std::tuple<double, std::vector<double>, std::vector<double>>>
            scores_per_fold;

    shuffleSamePersonPicks(amount_of_people, picks_per_person);

    for(int k = 0; k < 3; k++) {
        auto imageFold = getEquitativeSamplingFold(_trainImages, k, amount_of_people, picks_per_person);
        auto labelFold = getEquitativeSamplingFold(_trainLabels, k, amount_of_people, picks_per_person);

        Dataset d = Dataset(std::get<0>(imageFold), std::get<0>(labelFold),
                            std::get<1>(imageFold), std::get<1>(labelFold));
        d.trainPca(alpha, epsilon);
        scores_per_fold.push_back(allMetricsWrapper(std::get<1>(labelFold),
                                                    d.pca_kNN_predict(neighbours, epsilon)));
    }

    return scores_per_fold;
}


void Dataset::shuffleSamePersonPicks(int amount_of_people, int picks_per_person) {

    srand (time(NULL));

    for(int person = 0; person < amount_of_people; person++) {
        for(int i = 0; i < picks_per_person; i++) {
            int swap_index = rand() % (picks_per_person - i) ;
            _trainImages.swapRows(i + person * picks_per_person, i + person * picks_per_person + swap_index);
            _trainLabels.swapRows(i + person * picks_per_person, i + person * picks_per_person + swap_index);
        }
    }
}

std::tuple<Matrix, Matrix> Dataset::getEquitativeSamplingFold
        (const Matrix &input_matrix, int iteration, int amount_of_people, int picks_per_person) const {

    auto data = Matrix(input_matrix);

    for(int i = 0; i < amount_of_people; i++) {
        data.swapRows(i , i * picks_per_person + iteration * 2);
        data.swapRows(i , i * picks_per_person + 1 + iteration * 2);
    }

    auto trainSamples = data.subMatrix(amount_of_people, data.rows() - 1, 0, data.cols() - 1);
    auto testSamples = data.subMatrix(0, amount_of_people - 1, 0, data.cols() - 1);
    return std::make_tuple(trainSamples, testSamples);
}