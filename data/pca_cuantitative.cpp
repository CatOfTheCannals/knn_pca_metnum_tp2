#include <iostream>
#include <iostream>
#include <fstream>
#include <sstream>
#include "chrono"
#include <stdlib.h>
#include <tuple>

#include "../src/Pca.h"

#define GET_TIME std::chrono::high_resolution_clock::now()
#define GET_TIME_DELTA(begin, end) \
     std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()

int main(int argc, char** argv) {
    int n = 10;
    int m = 10;
    double epsilon = 0.0001;

    std::ostringstream filename;
    filename << "../../data/results/pca_cuantitative_" << n << "_m_" << m  << ".csv";
    ofstream file;
    file.open(filename.str());
    file << "rows" << "," << "cols" << "," << "alpha" << "," << "time" << std::endl;

    for(int rows = 2; rows < n; rows++){
        for(int cols = 2; cols < m; cols++){
            for(int alpha = 1; alpha < cols; alpha++){

                Matrix random_matrix = Matrix::random(rows, cols);

                auto begin = GET_TIME;
                auto vectors_and_lambdas = pca(random_matrix, alpha, epsilon);
                auto end = GET_TIME;

                file << rows << "," << cols << "," << alpha << "," << GET_TIME_DELTA(begin, end) << std::endl;
            }
        }
    }
}