#include "Dataset.h"

Matrix Dataset::getImages() const {
    Matrix output((*this)._images);
    return output;
}