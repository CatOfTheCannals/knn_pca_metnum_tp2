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
        d.trainPca(alphas.back());
        auto end = GET_TIME;
        auto train_time = GET_TIME_DELTA(begin, end);
        std::cout << "train_time : " << train_time << std::endl;

        // open output file
        int n = d.getTrainLabels().rows();
        std::ostringstream filename;
        filename << "../../data/results/pca_knn_cuali_cuanti_experiment_results_n_" << n <<".csv";
        ofstream file;
        file.open(filename.str());
        file << "alpha" << "," << "k" << "," << "accuracy" <<","<<"time" << std::endl;

        for (int alpha : alphas) {
            std::cout << std::endl << "alpha: " << alpha << " out of " << alphas.back() << std::endl;
            for (double p : neighbourhoodPercentualSizes) {

                int k = n * p;
                begin = GET_TIME;
                //auto results = d.pca_kNN_predict(k, alpha);
                //double acc = accuracy(d.getTestLabels(), results);
                double acc = 0;
                std::cout << "accuracy: " << acc << std::endl;
                end = GET_TIME;
                auto predict_time = GET_TIME_DELTA(begin, end);
                std::cout << "predict_time : " << predict_time << std::endl;

                file << alpha << "," << k << "," << acc << "," << predict_time << std::endl;


                if (LOG_EXP_PCA_VECS) {
                    std::cout << "getPca Vecs: " << std::endl;
                    std::cout << d.getPcaVecs() << std::endl;
                }

            }
        }

    }

}
