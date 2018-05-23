#include "Dataset.h"

Matrix Dataset::getImages() const {
    Matrix output((*this)._images);
    return output;
}

Matrix Dataset::getTargets() const{
    Matrix output((*this)._targets);
    return output;
}

Matrix Dataset::getPcaVecs() const {
    Matrix output((*this)._pcaVecs);
    return output;
}
Matrix Dataset::getPcaLambdas() const {
    Matrix output((*this)._pcaLambdas);
    return output;
}

void Dataset::trainPca(int alpha, double epsilon) {

    auto pca_eigenvectors_and_eigenvalues = pca((*this).getImages(), alpha, epsilon);

    _pcaVecs = std::get<0>(pca_eigenvectors_and_eigenvalues);
    _pcaLambdas = std::get<1>(pca_eigenvectors_and_eigenvalues);
}