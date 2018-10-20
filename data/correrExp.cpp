#include "pca_cuantitative.h"
#include "knn_cuantitative.h"
#include "pca_knn_cuantitative.h"
#include "kfold_exp.h"
#include "../src/Metrics.cpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <stdio.h>
#include <cstdlib>
#include <string>
#include <string.h>

#include "../src/Dataset.h"

int main(int argc, char** argv){

    Dataset d = Dataset::loadImdbVectorizedReviews();
    int method = 1;
    Matrix predictions;
    if(method == 1) {
        d.trainPca(25, 0.0001);
        predictions = d.pca_kNN_predict(1);
        std::cout << "k: 1, alpha: 25" << std::endl;
    } else {
    	std::cout << "entrando" << std::endl;
        predictions = d.kNN_predict(1);
        std::cout << "predict done" << std::endl;
    }

    allMetricsWrapper(d.getTestLabels(), predictions);

    std::cout << std::endl << "Finished exp." << std::endl;



    return 0;

}