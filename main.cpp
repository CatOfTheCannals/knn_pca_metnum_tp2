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

    if (argc != 9){
        std::cout << endl<<"	Unable to run program" << endl;
        std::cout << " Four parameters are expected:    -m <method> -i <train_set> -q <test_set> -o <classif>" << endl;
    return 1;
    }
    // leo el archivo de test -->la entrada i,j indica un 1 en la posicion [j][i]


    double method = std::atof(argv[2]);
    string trainSet(argv[4]);
    string testSet(argv[6]);
    string classif(argv[8]);

    if(method != 0 && method != 1) {
        std::cout << method << std::endl;
    } else {
        //get train set path and filename
        string key("/");
        int pos = trainSet.rfind(key);
        std::cout << "pos is: " << pos << std::endl;
        string trainFileName = trainSet.substr(pos+1, trainSet.length());
        string trainFileDir = trainSet.substr(0, pos+1);

        //get test set path and filename
        pos = testSet.rfind(key);
        std::cout << "pos is: " << pos << std::endl;
        string testFileName = testSet.substr(pos+1, testSet.length());
        string testFileDir = testSet.substr(0, pos+1);

        Dataset d = Dataset(trainFileDir, trainFileName, testFileDir, testFileName);

        Matrix predictions;
        if(method == 1) {
            int alpha = 25;
            d.trainPca(alpha, 0.0001);
            predictions = d.pca_kNN_predict(1, alpha);
            std::cout << "k: 1, alpha: 25" << std::endl;
        } else {
            predictions = d.kNN_predict(1);
            std::cout << "k: 1" << std::endl;
        }

        allMetricsWrapper(d.getTestLabels(), predictions);

        ofstream file;
        file.open(classif);

        for(int i = 0; i < predictions.rows(); i++) {
            file << predictions(i) << std::endl;
        }

    }
    
    return 0;
}

