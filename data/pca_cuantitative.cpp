#include <iostream>
#include <iostream>
#include <fstream>
#include <sstream>
#include "chrono"
#include <stdlib.h>
#include <tuple>

#include "../src/Pca.h"
#include "../src/Dataset.h"

#define GET_TIME std::chrono::high_resolution_clock::now()
#define GET_TIME_DELTA(begin, end) \
     std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()

int main(int argc, char** argv) {
    double epsilon = 0.0001;

    std::ostringstream filename;
    filename << "../../data/results/pca_cuantitative_reduced.csv";
    ofstream file;
    file.open(filename.str());
    file << "rows" << "," << "cols" << "," << "alpha" << "," << "time" << std::endl;

    Dataset reduced = Dataset("../../test/casos_test/", "testRed.in");

    Matrix images = reduced.getTrainImages();

    for(int rows = 2; rows < images.rows(); rows += images.rows() / 10){
        for(int alpha = 1; alpha < images.cols(); alpha += images.cols() / 10){

            Matrix images_sub_sample = images.subMatrix(0, rows, 0, images.cols() - 1);

            auto begin = GET_TIME;
            auto vectors_and_lambdas = pca(images_sub_sample, alpha, epsilon);
            auto end = GET_TIME;

            file << images_sub_sample.rows() << "," << alpha
                 << "," << GET_TIME_DELTA(begin, end) << std::endl;
        }
    }
}