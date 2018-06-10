#include "Dataset.h"

Matrix Dataset::getTestLabels() const{
    Matrix output(_testLabels);
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
    return testLabels;
}

    std::tuple<Matrix, Matrix> Dataset::getFold(const Matrix& input_matrix, int first_row, int second_row) const{
    auto data = Matrix(input_matrix);
    int chunk_size = second_row - first_row;
    for(int index_to_put_row_in = 0; index_to_put_row_in < chunk_size; index_to_put_row_in++) {
        for(int row_to_reposition = index_to_put_row_in + first_row; row_to_reposition > index_to_put_row_in; row_to_reposition-- ) {
            data.swapRows(row_to_reposition, row_to_reposition-1);
        }
    }
    auto trainSamples = data.subMatrix(chunk_size, data.rows() - 1, 0, data.cols() - 1);
    auto testSamples = data.subMatrix(0, chunk_size - 1, 0, data.cols() - 1);
    return std::make_tuple(trainSamples, testSamples);
}

void Dataset::kFold(int first_row, int last_row, int k) const {
    for(int i = 1; i <= k; i++) {
        first_row = (i - 1) * (_trainLabels.rows() / k);
        last_row = i * (_trainLabels.rows() / k) - 1;
        auto imageFold = (*this).getFold(_trainImages, first_row, last_row);
        auto labelFold= (*this).getFold(_trainLabels, first_row, last_row);

        Dataset d = Dataset(std::get<0>(imageFold), std::get<0>(labelFold), std::get<1>(imageFold), std::get<1>(labelFold));
        auto m = allMetricsAveraged(std::get<1>(labelFold), d.kNN_predict(5));
        std::cout << "accuracy: " << std::get<0>(m) << " | recall: " << std::get<1>(m) << " | precision: " << std::get<2>(m) << std::endl;
    }
}