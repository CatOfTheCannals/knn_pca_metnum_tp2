#include "kNN.hpp"
#include <math.h>




Matrix distance(Matrix& data, Matrix& image) {
    assert(data.cols() == image.cols());
    Matrix res(data.rows(),1);

    Matrix aux = Matrix(data.rows(),data.cols());
    for (int i = 1; i <= data.rows(); ++i) {
        aux.setIndex(i, 1, 1);
    }
    aux.multiply(image); //creates a matrix with image in all it's rows.
    data-aux;
    for (int i = 1; i <= aux.rows(); ++i){
        int sum = 0;
        for (int j = 1; j <= aux.cols(); ++j) {
            int value = aux(i,j);
            sum = sum + value * value;
        }
        res.setIndex(i,1, sqrt(sum)); //sets in every row of res the distance between the imput image and the
    }
    return res;
}
