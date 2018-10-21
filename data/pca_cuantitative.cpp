#include "pca_cuantitative.h"

void pca_cuantitative() {
    double epsilon = 0.0001;

    std::ostringstream filename;
    filename << "../../data/results/pca_cuantitative_imdb.csv";
    ofstream file;
    file.open(filename.str());
    file << "rows"  << "," << "alpha" << "," << "time" << std::endl;

    Dataset d = Dataset::loadImdbVectorizedReviews();
    Matrix features = d.getTrainImages();

    for(int rows = 2; rows < features.rows(); rows += features.rows() / 10){
        for(int alpha = 1; alpha < features.cols(); alpha += features.cols() / 10){

            Matrix features_sub_sample = features.subMatrix(0, rows, 0, features.cols() - 1);

            auto begin = GET_TIME;
            auto vectors_and_lambdas = pca(features_sub_sample , alpha, epsilon);
            auto end = GET_TIME;

            file << features_sub_sample.rows() << "," << alpha
                 << "," << GET_TIME_DELTA(begin, end) << std::endl;
        }
    }
}
