#include "pca_knn_cuantitative.h"

void pca_knn_cuantitative() {
    double epsilon = 0.0001;

    std::ostringstream filename;
    filename << "../../data/results/pca_knn_cuantitative_reduced.csv";
    ofstream file;
    file.open(filename.str());
    file << "alpha" << "," << "k" << "," << "time" << std::endl;

    Dataset d = Dataset("../../test/casos_test/", "testRed.in");

    d.splitTrainFromTest(0); //when we pass zero, it sets test to one row

    int rows = d.getTrainImages().rows();

    for(int alpha = 1; alpha < rows; alpha += rows / 10) {

        std::cout << std::endl << "alpha: " << alpha << std::endl;

        auto begin = GET_TIME;
        d.trainPca(alpha, epsilon);
        auto end = GET_TIME;
        auto train_time = GET_TIME_DELTA(begin, end);

        for(int k = 1; k < rows; k++){
            std::cout << std::endl << "alpha: " << alpha << std::endl;
            for(int repetitions = 0; repetitions < 20; repetitions++){

                auto begin = GET_TIME;
                d.pca_kNN_predict(k, alpha, epsilon);
                auto end = GET_TIME;
                auto predict_time = GET_TIME_DELTA(begin, end);

                file << alpha << "," << k << "," << train_time + predict_time << std::endl;
                std::cout << "time: " << train_time + predict_time << std::endl;
            }
        }
    }
}