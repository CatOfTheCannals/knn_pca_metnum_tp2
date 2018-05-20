#include "Dataset.h"

Matrix Dataset::getImages() const {
    Matrix output((*this).images);
    std::cout << output << std::endl;
    //return output;
}