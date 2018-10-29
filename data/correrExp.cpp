#include <iostream>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <stdio.h>
#include <cstdlib>
#include <string>
#include <string.h>

#include "../src/Dataset.h"

#include "pca_cuantitative.h"
#include "knn_cuantitative.h"
#include "pca_knn_cuantitative.h"
#include "kfold_exp.h"
#include "../src/Metrics.cpp"
#include "power_method_cuantitative.h"

int main(int argc, char** argv){

    std::cout << std::endl << "Starting cuanti cuali exp." << std::endl;

    auto alphas = vector<int>({5,10,15,20,50,100, 500});
    //auto neighbourhoodPercentualSizes = vector<double>({0.002, 0.004, 0.006, 0.01, 0.02, 0.05, 0.1, 0.15});
    auto neighbourhoodPercentualSizes = vector<double>({0.1, 0.15});

    auto chunkPercentages = vector<double>({0.02, 0.08,  0.4});
    // auto chunkPercentages = vector<double>({0.04, 0.2 });

    // knn_qualitative_and_quantitative(chunkPercentages, neighbourhoodPercentualSizes);
    pca_knn_qualitative_and_quantitative(alphas, chunkPercentages, neighbourhoodPercentualSizes);
    std::cout << std::endl << "Done cuanti cuali exp." << std::endl;

    return 0;

}
