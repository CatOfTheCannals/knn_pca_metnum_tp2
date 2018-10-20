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

    std::cout << std::endl << "Starting exp." << std::endl;

    power_method_cuantitative(10, 4 , 25, 5);


    std::cout << std::endl << "Finished exp." << std::endl;

    Matrix test_m = Matrix(4,4);
    for(int i = 0; i < 4; i++){
        test_m.setIndex(i,i, (double)(1.0 * (i+1)) );
    }
    test_m.mt_times_m().show_matrix();
    
    return 0;

}
