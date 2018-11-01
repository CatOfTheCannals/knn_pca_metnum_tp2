#include "pca_cuantitative.h"

void pca_cuantitative(const vector<int> alphas, const vector<double> chunkPercentages) {

    std::ostringstream filename;
    filename << "../../data/results/pca_cuantitative_imdb.csv";
    ofstream file;
    file.open(filename.str());
    file << "rows"  << "," << "alpha" << "," << "time" << std::endl;

    for(double chunkPercentage : chunkPercentages ) {

        // load dataset
        Dataset d = Dataset::loadImdbVectorizedReviews();
        std::cout << std::endl << "dataset successfully loaded" << std::endl;
        d.shuffle();
        d.chunkTrainSet(chunkPercentage);
        std::cout << std::endl << "dataset successfully chunked" << std::endl;

        int n = d.getTrainLabels().rows();

        std::cout << "Benchmark pca with %" << 100 * chunkPercentage << " of train set." << std::endl;
        for (int alpha : alphas) {

            // train pca with the biggest alpha
            auto begin = GET_TIME;
            d.trainActualPCA(alpha);
            auto end = GET_TIME;
            auto train_time = GET_TIME_DELTA(begin, end);
            std::cout << "train_time : " << train_time << std::endl;

            file << n << "," << alpha << "," << train_time  << std::endl;

        }
    }
}
