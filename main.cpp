#include <iostream>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <stdio.h>
#include <cstdlib>
#include <string>
#include <string.h>

#include "src/Dataset.h"


int main(int argc, char** argv){

    if (argc != 7){
        std::cout << endl<<"	Unable to run program" << endl;
        std::cout << " Four parameters are expected:    -m <method> -i <train_set> -q <test_set> -o <classif>" << endl;
    return 1;
    }
    // leo el archivo de test -->la entrada i,j indica un 1 en la posicion [j][i]

    auto  method = atoi(argv[2]);
    string dataset_path (argv[4]);
    string classif(argv[6]);


    if(method != 1 && method != 2) {
        std::cout << " Wrong method number. Options are:" << std::endl;
        std::cout << "\t1: k-NN (k-nearest neighbours)." << std::endl;
        std::cout << "\t2: k-NN + PCA (k-nearest neighbours + Principal Component Analysis)." << std::endl;
    } else {

        Dataset d = Dataset::loadImdbVectorizedReviews(dataset_path);

        std::cout << d.getTestLabels().rows() << std::endl;

        Matrix predictions;
        if(method == 2) {
            int alpha = 25;
            d.trainActualPCA(alpha);
            predictions = d.pca_kNN_predict(alpha, 1);
            std::cout << "k: 1, alpha: 25" << std::endl;
        } else {
            predictions = d.kNN_predict(1);
            std::cout << "k: 1" << std::endl;
        }

        allMetricsWrapper(d.getTestLabels(), predictions);

        ofstream output_file;
        output_file.open(classif);
        Matrix test_ids = d.getTestIds();

        for(int i = 0; i < predictions.rows(); i++) {
            output_file << test_ids(i) << "," << predictions(i) <<std::endl;
        }

    }
    return 0;
}

