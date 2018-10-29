#include "knn_cuantitative.h"

void knn_qualitative_and_quantitative(
        const vector<double> chunkPercentages, const vector<double> neighbourhoodPercentualSizes){

    for(double chunkPercentage : chunkPercentages ) {

        // load dataset
        Dataset d = Dataset::loadImdbVectorizedReviews();
        std::cout << std::endl << "dataset successfully loaded" << std::endl;

        // chunk it
        d.shuffle();
        d.chunkTrainSet(chunkPercentage);
        std::cout << std::endl << "dataset successfully chunked" << std::endl;

        // open output file
        int n = d.getTrainLabels().rows();
        std::ostringstream filename;
        filename << "../../data/results/knn_cuali_cuanti_experiment_results_n_" << n << ".csv";
        ofstream file;
        file.open(filename.str());
        file << "k" << "," << "accuracy" << "," << "recall_per_label" << "," << "precision_per_label" << "," << "time" << std::endl;
        std::cout << std::endl << "file opened" << std::endl;

        for (double p : neighbourhoodPercentualSizes) {
            int k = n * p;

            auto begin = GET_TIME;
            auto results = d.kNN_predict(k);
            auto metrics = allMetricsWrapper(d.getTestLabels(), results);
            double acc = std::get<0>(metrics);
            std::vector<double> recall_per_label = std::get<1>(metrics);
            std::vector<double> precision_per_label = std::get<2>(metrics);
            std::cout << "accuracy: " << acc << std::endl;
            auto end = GET_TIME;
            auto predict_time = GET_TIME_DELTA(begin, end);
            std::cout << "predict_time : " << predict_time << std::endl;

            file << k << "," << acc << "," << vecOfDoublesToString(recall_per_label) << ","
                 << vecOfDoublesToString(precision_per_label) << "," << predict_time << std::endl;
        }
    }
}
