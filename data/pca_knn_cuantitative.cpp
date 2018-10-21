#include "pca_knn_cuantitative.h"

void pca_knn_cuantitative() {
    double epsilon = 0.001;

    std::ostringstream filename;
    filename << "../../data/results/pca_knn_benchmarks_10by10_test_is_10_200_reps.csv";
    ofstream file;

    file.open(filename.str());
    file << "alpha" << "," << "k" << "," << "time" << std::endl;

    Dataset d = Dataset::loadImdbVectorizedReviews();
    std::cout << std::endl << "dataset successfully loaded" << std::endl;
    d.splitTrainFromTest(0.1); 
    std::cout << std::endl << "dataset successfully split" << std::endl;
    int rows = d.getTrainImages().rows();
    for (int alpha = 1; alpha < rows; alpha += rows / 10) {

        std::cout << std::endl << "alpha: " << alpha << std::endl;

        d.trainPca(alpha, epsilon);

        for (int repetitions = 0; repetitions < 200; repetitions++) {
            for (int k = 1; k < rows/10; k++) {
                std::cout << std::endl << "k: " << k << std::endl;
                auto begin = GET_TIME;
                d.pca_kNN_predict(k);
                auto end = GET_TIME;
                auto predict_time = GET_TIME_DELTA(begin, end);

                file << alpha << "," << k << "," << predict_time << std::endl;
               // std::cout << "time: " << predict_time << std::endl;
            }
        }
    }
}

void pca_knn_qualitative(){
    double epsilon = 0.001;

    std::ostringstream filename;
    filename << "../../data/results/pca_knn_qualitative_changing_alpha_test_is_10.csv";
    ofstream file;

    file.open(filename.str());
    file << "alpha" << "," << "k" << "," << "time" << std::endl;

    Dataset d = Dataset::loadImdbVectorizedReviews();
    std::cout << std::endl << "dataset successfully loaded" << std::endl;
    d.splitTrainFromTest(0.1); 
    std::cout << std::endl << "dataset successfully split" << std::endl;
    int rows = d.getTrainImages().rows();
    auto test_labels = d.getTestLabels();
    vector<int> alphas = vector<int>({1,5,10,15,20,170,170*2,170*3,170*4,170*5,170*6,170*7,170*8,170*9,1700});
    for(int alpha : alphas) {

        std::cout << std::endl << "alpha: " << alpha << " out of " << rows << std::endl;

        d.trainPca(alpha, epsilon);
        
        for (int k = 1; k < rows/10; k+=5) {
            std::cout << std::endl << "k: " << k << std::endl;
            auto begin = GET_TIME;
            auto results = d.pca_kNN_predict(k);
			double acc = accuracy(test_labels, results);

            auto end = GET_TIME;
            auto predict_time = GET_TIME_DELTA(begin, end);

            file << alpha << "," << k << "," << acc << std::endl;
           // std::cout << "time: " << predict_time << std::endl;
        }
        
    }
}