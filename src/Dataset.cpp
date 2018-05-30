#include "Dataset.h"

Matrix Dataset::getImages() const {
    Matrix output(_images);
    return output;
}

Matrix Dataset::getTargets() const{
    Matrix output(_targets);
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
        int swap_index = rand() % n - i ;
        _trainImages.swapRows(i, i + swap_index);
        _trainLabels.swapRows(i, i + swap_index);
    }
}

void Dataset::trainPca(int alpha, double epsilon) {
    assert(_trainImages.rows() > 0);
    auto pca_eigenvectors_and_eigenvalues = pca(_trainImages, alpha, epsilon);
    _pcaVecs = std::get<0>(pca_eigenvectors_and_eigenvalues);
    _pcaLambdas = std::get<1>(pca_eigenvectors_and_eigenvalues);
}

void Dataset::splitTrainFromTest(double testPercentage) {
    int testRows = _trainImages.rows() * testPercentage;
    _testImages = _trainImages.subMatrix(0, testRows, 0, _trainImages.cols() - 1);
    _testLabels = _trainLabels.subMatrix(0, testRows, 0, _trainLabels.cols() - 1);
    _trainImages = _trainImages.subMatrix(testRows + 1, _trainImages.rows() - 1, 0, _trainImages.cols() - 1);
    _trainLabels = _trainLabels.subMatrix(testRows + 1, _trainImages.rows() - 1, 0, _trainLabels.cols() - 1);
}


Matrix Dataset::pca_kNN_predict(int k, int alpha, double epsilon) const {
    Matrix testLabels = Matrix(_testImages.rows(), 1);

    Matrix transformedTrainImages = _trainImages.multiply(_pcaVecs);
    Matrix transformedTestImages = _testImages.multiply(_pcaVecs);

    for(int i = 0; i < transformedTestImages.rows(); i++) {
        int ith_label = kNN(transformedTrainImages, _trainLabels, transformedTestImages.getRow(i), k);
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

    std::cout << "accuracy is: " << accuracy(_testLabels, testLabels) << std::endl;
    for(int i = 0; i < 1; i++) {
        std::cout << "person: " << i << std::endl;
        std::cout << "recall at " << i << " is: " << recallPerPerson(_testLabels, testLabels, i) << std::endl;
        std::cout << "precision at " << i << " is: " << precisionPerPerson(_testLabels, testLabels, i) << std::endl;
        std::cout << std::endl;
    }

    return testLabels;
}
