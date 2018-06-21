#include "knn_cuantitative.h"

void knn_cuantitative() {
    double epsilon = 0.0001;

    std::ostringstream filename;
    filename << "../../data/results/knn_cuantitative_reduced.csv";
    ofstream file;
    file.open(filename.str());
    file << "k" << "," << "time" << std::endl;

    Dataset reduced = Dataset("../../test/casos_test/", "testRed.in");

    Matrix images = reduced.getTrainImages();
    Matrix labels = reduced.getTrainLabels();

    Matrix input_image = images.getRow(0);

    for(int k = 1; k < images.rows(); k++){
        for(int repetitions = 0; repetitions < 20; repetitions++){
            auto begin = GET_TIME;

            kNN(images, labels, input_image, k);

            auto end = GET_TIME;

            auto time = GET_TIME_DELTA(begin, end);

            file << k << "," << time << std::endl;

            std::cout << k << "," << time << std::endl;
        }

    }
}