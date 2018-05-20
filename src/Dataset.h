#ifndef TP2_METODOS_DATASET_H
#define TP2_METODOS_DATASET_H

#include "Matrix.hpp"

class Dataset {
public:
    Dataset() : images() {}

    Matrix getImages() const;

private:
    Matrix images = Matrix();
};


#endif //TP2_METODOS_DATASET_H
