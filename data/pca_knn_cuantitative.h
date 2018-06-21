#ifndef TP2_METODOS_PCA_KNN_CUANTITATIVE_H
#define TP2_METODOS_PCA_KNN_CUANTITATIVE_H

#include <iostream>
#include <iostream>
#include <fstream>
#include <sstream>
#include "chrono"
#include <stdlib.h>
#include <tuple>

#include "../src/Dataset.h"

#define GET_TIME std::chrono::high_resolution_clock::now()
#define GET_TIME_DELTA(begin, end) \
     std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()

void pca_knn_cuantitative();

#endif //TP2_METODOS_PCA_KNN_CUANTITATIVE_H
