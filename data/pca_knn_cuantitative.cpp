#include "pca_knn_cuantitative.h"


void pca_knn_qualitative_and_quantitative(){
    double epsilon = 0.001;

    std::ostringstream filename;
    filename << "../../data/results/knn_cuali_cuanti_experiment_results.csv";
    ofstream file;
    file.open(filename.str());
    file << "alpha" << "," << "k" << "," << "accuracy" <<","<<"time" << std::endl;

    Dataset d = Dataset::loadImdbVectorizedReviews();
    std::cout << std::endl << "dataset successfully loaded" << std::endl;
    double test_ratio = 0.9;
    d.shuffle();
    //d.splitTrainFromTest(test_ratio);
    std::cout << std::endl << "dataset successfully split" << std::endl;


    auto test_labels = d.getTestLabels();
    auto train_labels = d.getTrainLabels();


    int MAX_ALPHA = 500;
    vector<int> alphas = vector<int>({5,10,15,20,50,100, MAX_ALPHA});

    auto begin = GET_TIME;
    d.trainPca(MAX_ALPHA, epsilon);
    auto end = GET_TIME;
    auto train_time = GET_TIME_DELTA(begin, end);
    std::cout <<  "train_time : " << train_time << std::endl;

    for(int alpha : alphas) {
        std::cout << std::endl << "alpha: " << alpha << " out of " << MAX_ALPHA  << std::endl;
        int iterations = train_labels.rows()/10;
        //int iterations = 6;
        int step = 10;
        for (int k = 5; k < iterations; k+=step) {
            
            begin = GET_TIME;
            auto results = d.pca_kNN_predict(k, alpha);
			double acc = accuracy(test_labels, results);
			std::cout << "accuracy: " << acc << std::endl;
            end = GET_TIME;
            auto predict_time = GET_TIME_DELTA(begin, end);
            std::cout <<  "predict_time : " << predict_time << std::endl;

            if(LOG_EXP_RESULTS_SUM) {
                std::cout << "results sumapha:" << alpha << " k: " << k << std::endl;
                std::cout << results.sum() << std::endl;
            }

            file << alpha << "," << k << "," << acc << "," << predict_time << std::endl;
        }
    }

}
