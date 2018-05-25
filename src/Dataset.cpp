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
    int n = _images.rows();
    srand (time(NULL));
    for(int i = 0; i < n; i++) {
        int swap_index = rand() % n - i ;
        _images.swapRows(i, i + swap_index);
        _targets.swapRows(i, i + swap_index);
    }
}

void Dataset::trainPca(int alpha, double epsilon) {

    auto pca_eigenvectors_and_eigenvalues = pca(getImages(), alpha, epsilon);

    _pcaVecs = std::get<0>(pca_eigenvectors_and_eigenvalues);
    _pcaLambdas = std::get<1>(pca_eigenvectors_and_eigenvalues);
}

void Dataset::splitTrainFromTest(double testPercentage) {
    int testRows = _images.rows() * testPercentage;
    _testImages = _images.subMatrix(0, testRows, 0, _images.cols() - 1);
    _testLabels = _targets.subMatrix(0, testRows, 0, _targets.cols() - 1);
    _trainImages = _images.subMatrix(testRows + 1, _images.rows() - 1, 0, _images.cols() - 1);
    _trainLabels = _targets.subMatrix(testRows + 1, _images.rows() - 1, 0, _targets.cols() - 1);
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
