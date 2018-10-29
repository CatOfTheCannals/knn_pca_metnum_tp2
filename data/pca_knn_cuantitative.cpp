#include "pca_knn_cuantitative.h"


void pca_knn_qualitative_and_quantitative(
        const vector<int> alphas, const vector<double> chunkPercentages, const vector<double> neighbourhoodPercentualSizes){

    for(double chunkPercentage : chunkPercentages ) {

        // load dataset
        Dataset d = Dataset::loadImdbVectorizedReviews();
        std::cout << std::endl << "dataset successfully loaded" << std::endl;
        d.shuffle();
        d.chunkTrainSet(chunkPercentage);
        std::cout << std::endl << "dataset successfully chunked" << std::endl;

        // train pca with the biggest alpha
        auto begin = GET_TIME;
        d.trainActualPCA(alphas.back());
        auto end = GET_TIME;
        auto train_time = GET_TIME_DELTA(begin, end);
        std::cout << "train_time : " << train_time << std::endl;

        // open pca_predict output file
        int n = d.getTrainLabels().rows();
        std::ostringstream pca_predict_filename;
        pca_predict_filename << "../../data/results/pca_knn_cuali_cuanti_experiment_results_n_" << n <<".csv";
        ofstream file;
        file.open(pca_predict_filename .str());
        file << "alpha" << "," << "k" << "," << "accuracy" <<","<<"time" << std::endl;

        for (int alpha : alphas) {
            std::cout << std::endl << "alpha: " << alpha << " out of " << alphas.back() << std::endl;
            for (double p : neighbourhoodPercentualSizes) {

                int k = n * p;
                begin = GET_TIME;
                Matrix results = d.pca_kNN_predict(k, alpha);
                auto metrics = allMetricsWrapper(d.getTestLabels(), results);
                double acc = std::get<0>(metrics);
                std::vector<double> recall_per_label = std::get<1>(metrics);
                std::vector<double> precision_per_label = std::get<2>(metrics);
                end = GET_TIME;
                auto predict_time = GET_TIME_DELTA(begin, end);
                std::cout << "predict_time : " << predict_time << std::endl;

                file << alpha << "," << k << "," << acc << "," << vecOfDoublesToString(recall_per_label) << ","
                     << vecOfDoublesToString(precision_per_label) << "," << predict_time << std::endl;


                if (LOG_EXP_PCA_VECS) {
                    std::cout << "getPca Vecs: " << std::endl;
                    std::cout << d.getPcaVecs() << std::endl;
                }

            }
        }

    }

}
