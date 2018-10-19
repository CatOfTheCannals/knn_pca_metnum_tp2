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
    int method = 0;
    Matrix predictions;
    if(method == 1) {
        predictions = d.pca_kNN_predict(1, 25);
        std::cout << "k: 1, alpha: 25" << std::endl;
    } else {
    	std::cout << "entrando" << std::endl;
        predictions = d.kNN_predict(1);

        std::cout << "k: 1" << std::endl;
    }

    allMetricsWrapper(d.getTestLabels(), predictions);

    std::cout << std::endl << "Finished exp." << std::endl;



    return 0;

}