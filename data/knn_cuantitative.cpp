#include "knn_cuantitative.h"

void knn_qualitative_and_quantitative(){
    double epsilon = 0.001;

    std::ostringstream filename;
    filename << "../../data/results/knn_cuali_cuanti_experiment_results.csv";
    ofstream file;
    file.open(filename.str());
    file << "k" << "," << "accuracy" <<","<<"time" << std::endl;

    Dataset d = Dataset::loadImdbVectorizedReviews();
    std::cout << std::endl << "dataset successfully loaded" << std::endl;
    double test_ratio = 0.9;
    d.shuffle();
    //d.splitTrainFromTest(test_ratio);
    std::cout << std::endl << "dataset successfully split" << std::endl;


    auto test_labels = d.getTestLabels();
    auto train_labels = d.getTrainLabels();

    auto begin = GET_TIME;
    auto end = GET_TIME;
    auto train_time = GET_TIME_DELTA(begin, end);
    std::cout <<  "train_time : " << train_time << std::endl;

    int iterations = train_labels.rows()/10;
    //int iterations = 6;
    int step = 10;
    for (int k = 5; k < iterations; k+=step) {

        begin = GET_TIME;
        auto results = d.kNN_predict(k);
        double acc = accuracy(test_labels, results);
        std::cout << "accuracy: " << acc << std::endl;
        end = GET_TIME;
        auto predict_time = GET_TIME_DELTA(begin, end);
        std::cout <<  "predict_time : " << predict_time << std::endl;

        if(LOG_EXP_RESULTS_SUM) {
            std::cout << "results sum k: " << k << std::endl;
            std::cout << results.sum() << std::endl;
        }

        file << k << "," << acc << "," << predict_time << std::endl;
    }

}
