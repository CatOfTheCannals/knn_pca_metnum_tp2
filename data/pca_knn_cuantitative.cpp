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
                d.pca_kNN_predict_old(k);
                auto end = GET_TIME;
                auto predict_time = GET_TIME_DELTA(begin, end);

                file << alpha << "," << k << "," << predict_time << std::endl;
               // std::cout << "time: " << predict_time << std::endl;
            }
        }
    }
}

void pca_knn_qualitative_and_quantitative(){
    double epsilon = 0.001;

    std::ostringstream filename;
    filename << "../../data/results/pca_knn_qualitative_changing_alpha_test_is_10.csv";
    ofstream file;

    file.open(filename.str());
    file << "alpha" << "," << "k" << "," << "accuracy" <<","<<"time" << std::endl;

    Dataset d = Dataset::loadImdbVectorizedReviews();
    std::cout << std::endl << "dataset successfully loaded" << std::endl;

    double test_ratio = 0.75;
    d.splitTrainFromTest(test_ratio);
    std::cout << std::endl << "dataset successfully split" << std::endl;


    int rows = d.getTrainImages().rows();
    auto test_labels = d.getTestLabels();
    int MAX_ALPHA = d.getTrainImages().cols();
    std::cout << d.getTrainImages().cols() << std::endl;
    vector<int> alphas = vector<int>({5,10,15,20,50,100,MAX_ALPHA/10,MAX_ALPHA/4,MAX_ALPHA/2,MAX_ALPHA*7/10,MAX_ALPHA});
    //vector<int> alphas = vector<int>({5});
    auto begin = GET_TIME;
    d.trainPca(MAX_ALPHA, epsilon);
    auto end = GET_TIME;
    auto train_time = GET_TIME_DELTA(begin, end);

    
    std::cout <<  "train_time : " << train_time << std::endl;
    for(int alpha : alphas) {

        
        std::cout << std::endl << "alpha: " << alpha << " out of " << MAX_ALPHA  << std::endl;
        // int iterations = rows/10;
        int iterations = 2;
        int step = 10;
        for (int k = 1; k < iterations; k+=step) {
            
            std::cout << std::endl << "k: " << k << std::endl;
            begin = GET_TIME;
            auto results = d.pca_kNN_predict_old(k, alpha);

			double acc = accuracy(test_labels, results);
			std::cout << "accuracy: " << acc << std::endl;
            end = GET_TIME;
            auto predict_time = GET_TIME_DELTA(begin, end);
            std::cout <<  "predict_time : " << predict_time << std::endl;

            file << alpha << "," << k << "," << acc << "," << predict_time << std::endl;
        }
        
    }
}
