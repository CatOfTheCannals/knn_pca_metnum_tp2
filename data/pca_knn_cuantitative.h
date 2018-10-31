#ifndef TP2_METODOS_PCA_KNN_CUANTITATIVE_H
#define TP2_METODOS_PCA_KNN_CUANTITATIVE_H

#include <iostream>
#include <iostream>
#include <fstream>
#include <sstream>
#include "chrono"
#include <stdlib.h>
#include <tuple>

#include "../src/utils.h"
#include "../src/Dataset_tester.h"
#include "../src/Dataset.h"

void pca_knn_cuantitative();
void pca_knn_qualitative_and_quantitative(
        const vector<int> alphas,
        const vector<double> chunkPercentages,
        const vector<double> neighbourhoodPercentualSizes,
        const vector<double> frequencyThresholds);

#endif //TP2_METODOS_PCA_KNN_CUANTITATIVE_H
